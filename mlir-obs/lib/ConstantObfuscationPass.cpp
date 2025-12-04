#include "Obfuscator/Passes.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"

#include <string>

using namespace mlir;
using namespace mlir::obs;

namespace {

/// XOR encrypt for strings - same as StringEncryptPass for consistency
static std::string xorEncrypt(const std::string &input, const std::string &key) {
  std::string out = input;
  for (size_t i = 0; i < input.size(); i++) {
    out[i] = input[i] ^ key[i % key.size()];
  }
  return out;
}

/// Track which globals have been encrypted
struct EncryptedGlobalInfo {
  std::string globalName;
  size_t originalLength;
};

} // namespace

/// ConstantObfuscationPass - Obfuscate string constants with runtime decryption
///
/// NOTE: This pass ONLY obfuscates string data. Integer and float obfuscation
/// was removed because it breaks program semantics - there's no way to reverse
/// arithmetic transformations at runtime for values used in computations.
///
/// String obfuscation works because:
/// 1. Strings are stored as global data
/// 2. We can decrypt them in-place at program startup via constructors
/// 3. The decryption is transparent to the rest of the program
///
/// Integer obfuscation would break because:
/// 1. Values like "1" or "3.14" are used in arithmetic
/// 2. Transforming 1 -> XOR_MASK + OFFSET would break all computations
/// 3. There's no runtime "intercept point" to reverse the transformation
void ConstantObfuscationPass::runOnOperation() {
  ModuleOp module = getOperation();
  MLIRContext *ctx = module.getContext();
  OpBuilder builder(ctx);

  // Track encrypted globals
  std::vector<EncryptedGlobalInfo> encryptedGlobals;

  // ============================================================================
  // STEP 1: Obfuscate String Literals in LLVM GlobalOp
  // ============================================================================
  module.walk([&](LLVM::GlobalOp globalOp) {
    StringRef symName = globalOp.getSymName();

    // Skip internal and system globals
    if (symName.starts_with("__obfs_") || symName.starts_with("llvm."))
      return;

    // Check if this is a string constant
    if (auto strAttr = globalOp.getValueAttr()) {
      if (auto stringAttr = llvm::dyn_cast<StringAttr>(strAttr)) {
        std::string original = stringAttr.getValue().str();

        // Skip empty strings
        if (original.empty())
          return;

        std::string encrypted = xorEncrypt(original, key);

        // Replace the global's initializer with encrypted string
        globalOp.setValueAttr(StringAttr::get(ctx, encrypted));

        // Make writable for runtime decryption
        globalOp.setConstant(false);

        // Track for init function
        encryptedGlobals.push_back({symName.str(), original.size()});
      }
    }
  });

  // If no strings were encrypted, nothing more to do
  if (encryptedGlobals.empty())
    return;

  // ============================================================================
  // STEP 2: Create key global, decrypt function, and init constructor
  // ============================================================================
  builder.setInsertionPointToStart(module.getBody());
  Location loc = module.getLoc();

  auto i8Type = IntegerType::get(ctx, 8);
  auto i32Type = IntegerType::get(ctx, 32);
  auto i64Type = IntegerType::get(ctx, 64);
  auto i8PtrType = LLVM::LLVMPointerType::get(ctx);
  auto voidType = LLVM::LLVMVoidType::get(ctx);

  // Create key global if it doesn't exist
  if (!module.lookupSymbol<LLVM::GlobalOp>("__obfs_key")) {
    auto keyArrayType = LLVM::LLVMArrayType::get(i8Type, key.size());
    auto keyGlobal = builder.create<LLVM::GlobalOp>(
        loc,
        keyArrayType,
        /*isConstant=*/true,
        LLVM::Linkage::Private,
        "__obfs_key",
        builder.getStringAttr(key)
    );
    keyGlobal.setUnnamedAddr(LLVM::UnnamedAddr::Global);
  }

  // Create the decrypt function if it doesn't exist
  if (!module.lookupSymbol<LLVM::LLVMFuncOp>("__obfs_decrypt")) {
    auto funcType = LLVM::LLVMFunctionType::get(voidType, {i8PtrType, i32Type}, false);
    auto decryptFunc = builder.create<LLVM::LLVMFuncOp>(
        loc, "__obfs_decrypt", funcType, LLVM::Linkage::Private);
    decryptFunc.setAlwaysInline(true);

    // Create function body with XOR decryption loop
    Block *entryBlock = decryptFunc.addEntryBlock(builder);
    builder.setInsertionPointToStart(entryBlock);

    Value strPtr = entryBlock->getArgument(0);
    Value len = entryBlock->getArgument(1);
    Value keyAddr = builder.create<LLVM::AddressOfOp>(loc, i8PtrType, "__obfs_key");

    Value zero32 = builder.create<LLVM::ConstantOp>(loc, i32Type, builder.getI32IntegerAttr(0));
    Value one32 = builder.create<LLVM::ConstantOp>(loc, i32Type, builder.getI32IntegerAttr(1));
    Value keyLenVal = builder.create<LLVM::ConstantOp>(loc, i32Type, builder.getI32IntegerAttr(key.size()));

    // Allocate loop counter
    Value iPtr = builder.create<LLVM::AllocaOp>(loc, i8PtrType, i32Type, one32);
    builder.create<LLVM::StoreOp>(loc, zero32, iPtr);

    // Create loop blocks
    Block *loopCond = decryptFunc.addBlock();
    Block *loopBody = decryptFunc.addBlock();
    Block *loopEnd = decryptFunc.addBlock();

    builder.create<LLVM::BrOp>(loc, loopCond);

    // Loop condition: i < len
    builder.setInsertionPointToStart(loopCond);
    Value i = builder.create<LLVM::LoadOp>(loc, i32Type, iPtr);
    Value cond = builder.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::slt, i, len);
    builder.create<LLVM::CondBrOp>(loc, cond, loopBody, loopEnd);

    // Loop body: str[i] ^= key[i % keyLen]
    builder.setInsertionPointToStart(loopBody);
    Value iLoad = builder.create<LLVM::LoadOp>(loc, i32Type, iPtr);

    Value iExt = builder.create<LLVM::SExtOp>(loc, i64Type, iLoad);
    Value strElemPtr = builder.create<LLVM::GEPOp>(loc, i8PtrType, i8Type, strPtr, ValueRange{iExt});
    Value strChar = builder.create<LLVM::LoadOp>(loc, i8Type, strElemPtr);

    Value keyIdx = builder.create<LLVM::SRemOp>(loc, iLoad, keyLenVal);
    Value keyIdxExt = builder.create<LLVM::SExtOp>(loc, i64Type, keyIdx);
    Value keyElemPtr = builder.create<LLVM::GEPOp>(loc, i8PtrType, i8Type, keyAddr, ValueRange{keyIdxExt});
    Value keyChar = builder.create<LLVM::LoadOp>(loc, i8Type, keyElemPtr);

    Value xored = builder.create<LLVM::XOrOp>(loc, strChar, keyChar);
    builder.create<LLVM::StoreOp>(loc, xored, strElemPtr);

    Value iNext = builder.create<LLVM::AddOp>(loc, iLoad, one32);
    builder.create<LLVM::StoreOp>(loc, iNext, iPtr);
    builder.create<LLVM::BrOp>(loc, loopCond);

    builder.setInsertionPointToStart(loopEnd);
    builder.create<LLVM::ReturnOp>(loc, ValueRange{});
  }

  // Create init function that calls decrypt for each string
  builder.setInsertionPointToEnd(module.getBody());
  auto initFuncType = LLVM::LLVMFunctionType::get(voidType, {}, false);

  if (!module.lookupSymbol<LLVM::LLVMFuncOp>("__obfs_init")) {
    auto initFunc = builder.create<LLVM::LLVMFuncOp>(
        loc, "__obfs_init", initFuncType, LLVM::Linkage::Internal);

    Block *entryBlock = initFunc.addEntryBlock(builder);
    builder.setInsertionPointToStart(entryBlock);

    for (const auto &info : encryptedGlobals) {
      Value globalAddr = builder.create<LLVM::AddressOfOp>(loc, i8PtrType, info.globalName);
      Value lenVal = builder.create<LLVM::ConstantOp>(loc, i32Type,
                                                       builder.getI32IntegerAttr(info.originalLength));
      builder.create<LLVM::CallOp>(loc, TypeRange{}, "__obfs_decrypt", ValueRange{globalAddr, lenVal});
    }

    builder.create<LLVM::ReturnOp>(loc, ValueRange{});
  }

  // Register __obfs_init as a constructor via llvm.global_ctors
  if (!module.lookupSymbol<LLVM::GlobalOp>("llvm.global_ctors")) {
    builder.setInsertionPointToEnd(module.getBody());

    auto ctorStructType = LLVM::LLVMStructType::getLiteral(ctx, {i32Type, i8PtrType, i8PtrType});
    auto ctorArrayType = LLVM::LLVMArrayType::get(ctorStructType, 1);

    auto ctorsGlobal = builder.create<LLVM::GlobalOp>(
        loc,
        ctorArrayType,
        /*isConstant=*/false,
        LLVM::Linkage::Appending,
        "llvm.global_ctors",
        Attribute()
    );

    // Create initializer region
    Region &initRegion = ctorsGlobal.getInitializerRegion();
    Block *initBlock = builder.createBlock(&initRegion);
    builder.setInsertionPointToStart(initBlock);

    Value priority = builder.create<LLVM::ConstantOp>(loc, i32Type, builder.getI32IntegerAttr(65535));
    Value initFuncPtr = builder.create<LLVM::AddressOfOp>(loc, i8PtrType, "__obfs_init");
    Value nullPtr = builder.create<LLVM::ZeroOp>(loc, i8PtrType);

    Value structVal = builder.create<LLVM::UndefOp>(loc, ctorStructType);
    structVal = builder.create<LLVM::InsertValueOp>(loc, structVal, priority, ArrayRef<int64_t>{0});
    structVal = builder.create<LLVM::InsertValueOp>(loc, structVal, initFuncPtr, ArrayRef<int64_t>{1});
    structVal = builder.create<LLVM::InsertValueOp>(loc, structVal, nullPtr, ArrayRef<int64_t>{2});

    Value arrayVal = builder.create<LLVM::UndefOp>(loc, ctorArrayType);
    arrayVal = builder.create<LLVM::InsertValueOp>(loc, arrayVal, structVal, ArrayRef<int64_t>{0});

    builder.create<LLVM::ReturnOp>(loc, arrayVal);
  }
}

std::unique_ptr<Pass> mlir::obs::createConstantObfuscationPass(llvm::StringRef key) {
  return std::make_unique<ConstantObfuscationPass>(key.str());
}
