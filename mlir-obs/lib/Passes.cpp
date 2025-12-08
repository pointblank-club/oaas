#include "Obfuscator/Passes.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Builders.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "llvm/ADT/STLExtras.h"

#include <string>
#include <set>

using namespace mlir;
using namespace mlir::obs;

namespace {

static std::string xorEncrypt(const std::string &input, const std::string &key) {
  std::string out = input;
  for (size_t i = 0; i < input.size(); i++) {
    out[i] = input[i] ^ key[i % key.size()];
  }
  return out;
}

/// Track which globals have been encrypted so we can inject decryption calls
struct EncryptedGlobalInfo {
  std::string globalName;
  size_t originalLength;
};

} // namespace


void StringEncryptPass::runOnOperation() {
  ModuleOp module = getOperation();
  MLIRContext *ctx = module.getContext();
  OpBuilder builder(ctx);

  // Track encrypted globals
  std::vector<EncryptedGlobalInfo> encryptedGlobals;

  // Step 1: Encrypt string globals, make them writable, and track them
  module.walk([&](LLVM::GlobalOp globalOp) {
    StringRef symName = globalOp.getSymName();

    // Skip our internal globals and system globals
    if (symName.starts_with("__obfs_") || symName.starts_with("llvm."))
      return;

    // BUG #3 FIX: Skip C++ compiler-generated global initializers
    // These have section specifiers that must NOT be encrypted
    if (symName.starts_with("__cxx_global_var_init") ||
        symName.starts_with("_GLOBAL__sub_I_") ||
        symName.starts_with("__cxx_global_array_dtor") ||
        symName.starts_with("__dtor_") ||
        symName.starts_with("__ctor_") ||
        symName.starts_with("GCC_except_table") ||
        symName.starts_with("__func__") ||
        symName.starts_with("__PRETTY_FUNCTION__") ||
        symName.starts_with("__FUNCTION__"))
      return;

    // BUG #3 FIX: Skip globals with section attributes (section specifiers)
    // Section names like "__TEXT,__StaticInit" must not be encrypted
    if (globalOp.getSection().has_value())
      return;

    if (auto strAttr = globalOp.getValueAttr()) {
      if (auto stringAttr = llvm::dyn_cast<StringAttr>(strAttr)) {
        std::string original = stringAttr.getValue().str();

        // Skip empty strings
        if (original.empty())
          return;

        // BUG #3 FIX: Skip very short strings (likely metadata, not user data)
        if (original.size() < 2)
          return;

        std::string encrypted = xorEncrypt(original, key);

        // Update the global with encrypted value
        globalOp.setValueAttr(StringAttr::get(ctx, encrypted));

        // CRITICAL: Make the global writable so we can decrypt at runtime
        // Originally const globals become non-const for in-place decryption
        globalOp.setConstant(false);

        // Track this global for decryption injection
        encryptedGlobals.push_back({symName.str(), original.size()});
      }
    }
  });

  // If no strings were encrypted, nothing more to do
  if (encryptedGlobals.empty())
    return;

  // Step 2: Create key global
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

  // Step 3: Create the decrypt function
  // BUG #4 FIX: Use Internal linkage (not Private) to prevent dead code elimination
  // BUG #4 FIX: Use NoInline instead of AlwaysInline to prevent function removal
  if (!module.lookupSymbol<LLVM::LLVMFuncOp>("__obfs_decrypt")) {
    auto funcType = LLVM::LLVMFunctionType::get(voidType, {i8PtrType, i32Type}, false);
    auto decryptFunc = builder.create<LLVM::LLVMFuncOp>(
        loc, "__obfs_decrypt", funcType, LLVM::Linkage::Internal);
    decryptFunc.setNoInline(true);

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

    // Get str[i] pointer
    Value iExt = builder.create<LLVM::SExtOp>(loc, i64Type, iLoad);
    Value strElemPtr = builder.create<LLVM::GEPOp>(loc, i8PtrType, i8Type, strPtr, ValueRange{iExt});
    Value strChar = builder.create<LLVM::LoadOp>(loc, i8Type, strElemPtr);

    // Get key[i % keyLen]
    Value keyIdx = builder.create<LLVM::SRemOp>(loc, iLoad, keyLenVal);
    Value keyIdxExt = builder.create<LLVM::SExtOp>(loc, i64Type, keyIdx);
    Value keyElemPtr = builder.create<LLVM::GEPOp>(loc, i8PtrType, i8Type, keyAddr, ValueRange{keyIdxExt});
    Value keyChar = builder.create<LLVM::LoadOp>(loc, i8Type, keyElemPtr);

    // XOR and store back
    Value xored = builder.create<LLVM::XOrOp>(loc, strChar, keyChar);
    builder.create<LLVM::StoreOp>(loc, xored, strElemPtr);

    // Increment i
    Value iNext = builder.create<LLVM::AddOp>(loc, iLoad, one32);
    builder.create<LLVM::StoreOp>(loc, iNext, iPtr);
    builder.create<LLVM::BrOp>(loc, loopCond);

    // Loop end: return
    builder.setInsertionPointToStart(loopEnd);
    builder.create<LLVM::ReturnOp>(loc, ValueRange{});
  }

  // Step 4: Create init function that calls decrypt for each string
  builder.setInsertionPointToEnd(module.getBody());
  auto initFuncType = LLVM::LLVMFunctionType::get(voidType, {}, false);

  if (!module.lookupSymbol<LLVM::LLVMFuncOp>("__obfs_init")) {
    // BUG #4 FIX: Use External linkage to ensure linker doesn't remove this function
    // The function is referenced by global_ctors but optimizer may not see that
    auto initFunc = builder.create<LLVM::LLVMFuncOp>(
        loc, "__obfs_init", initFuncType, LLVM::Linkage::External);
    initFunc.setNoInline(true);

    Block *entryBlock = initFunc.addEntryBlock(builder);
    builder.setInsertionPointToStart(entryBlock);

    // Call decrypt for each encrypted global
    for (const auto &info : encryptedGlobals) {
      Value globalAddr = builder.create<LLVM::AddressOfOp>(loc, i8PtrType, info.globalName);
      Value lenVal = builder.create<LLVM::ConstantOp>(loc, i32Type,
                                                       builder.getI32IntegerAttr(info.originalLength));
      builder.create<LLVM::CallOp>(loc, TypeRange{}, "__obfs_decrypt", ValueRange{globalAddr, lenVal});
    }

    builder.create<LLVM::ReturnOp>(loc, ValueRange{});
  }

  // Step 5: Register __obfs_init as a constructor via llvm.global_ctors
  // Use GlobalCtorsOp AND add to @llvm.used to prevent dead code elimination
  builder.setInsertionPointToEnd(module.getBody());

  // Check if GlobalCtorsOp already exists
  LLVM::GlobalCtorsOp existingCtors = nullptr;
  for (auto &op : module.getBody()->getOperations()) {
    if (auto ctorsOp = llvm::dyn_cast<LLVM::GlobalCtorsOp>(&op)) {
      existingCtors = ctorsOp;
      break;
    }
  }

  if (existingCtors) {
    // Append our __obfs_init to existing ctors
    SmallVector<Attribute> newCtors;
    SmallVector<Attribute> newPriorities;
    SmallVector<Attribute> newData;

    for (auto attr : existingCtors.getCtors())
      newCtors.push_back(attr);
    for (auto attr : existingCtors.getPriorities())
      newPriorities.push_back(attr);
    if (auto dataAttr = existingCtors.getData()) {
      for (auto attr : dataAttr)
        newData.push_back(attr);
    }

    newCtors.push_back(FlatSymbolRefAttr::get(ctx, "__obfs_init"));
    newPriorities.push_back(builder.getI32IntegerAttr(101));
    newData.push_back(LLVM::ZeroAttr::get(ctx));

    builder.setInsertionPoint(existingCtors);
    builder.create<LLVM::GlobalCtorsOp>(
        loc,
        builder.getArrayAttr(newCtors),
        builder.getArrayAttr(newPriorities),
        builder.getArrayAttr(newData)
    );
    existingCtors.erase();
  } else {
    // Create new GlobalCtorsOp
    SmallVector<Attribute> ctors;
    SmallVector<Attribute> priorities;
    SmallVector<Attribute> data;

    ctors.push_back(FlatSymbolRefAttr::get(ctx, "__obfs_init"));
    priorities.push_back(builder.getI32IntegerAttr(101));
    data.push_back(LLVM::ZeroAttr::get(ctx));

    builder.create<LLVM::GlobalCtorsOp>(
        loc,
        builder.getArrayAttr(ctors),
        builder.getArrayAttr(priorities),
        builder.getArrayAttr(data)
    );
  }

  // Step 6: Add __obfs_init and __obfs_decrypt to @llvm.used to prevent DCE
  // This ensures the linker and optimizer cannot remove these critical functions
  auto ptrArrayType = LLVM::LLVMArrayType::get(i8PtrType, 2);

  // Check if @llvm.used already exists
  LLVM::GlobalOp existingUsed = nullptr;
  for (auto &op : module.getBody()->getOperations()) {
    if (auto globalOp = llvm::dyn_cast<LLVM::GlobalOp>(&op)) {
      if (globalOp.getSymName() == "llvm.used") {
        existingUsed = globalOp;
        break;
      }
    }
  }

  // Create @llvm.used with our functions
  // Note: We create a new one; if one exists, we'd need to merge (simplified here)
  if (!existingUsed) {
    SmallVector<Attribute> usedRefs;
    usedRefs.push_back(FlatSymbolRefAttr::get(ctx, "__obfs_init"));
    usedRefs.push_back(FlatSymbolRefAttr::get(ctx, "__obfs_decrypt"));

    auto usedGlobal = builder.create<LLVM::GlobalOp>(
        loc,
        ptrArrayType,
        /*isConstant=*/true,
        LLVM::Linkage::Appending,
        "llvm.used",
        builder.getArrayAttr(usedRefs)
    );
    // Set the section to llvm.metadata (required for @llvm.used)
    usedGlobal.setSection("llvm.metadata");
  }

  // Skip encrypting metadata attributes - only encrypt actual string data in globals
  // The second walk for metadata is kept but only skips attributes (no encryption)
  module.walk([&](Operation *op) {
    // Skip LLVM::GlobalOp as we handled them above
    if (llvm::isa<LLVM::GlobalOp>(op))
      return;

    // Just iterate through attributes and skip metadata ones
    // This preserves the skip list for any future changes
    for (auto &attr : op->getAttrs()) {
      StringRef attrName = attr.getName().getValue();

      // These attributes should never be encrypted - they are IR metadata
      if (attrName == "sym_name" ||
          attrName == "sym_visibility" ||
          attrName == "function_ref" ||
          attrName == "callee" ||
          attrName == "llvm.target_triple" ||
          attrName == "llvm.ident" ||
          attrName == "llvm.module_asm" ||
          attrName == "target_cpu" ||
          attrName == "tune_cpu" ||
          attrName == "target_features" ||
          attrName == "frame_pointer" ||
          attrName == "uwtable_kind" ||
          attrName == "linkage" ||
          attrName == "visibility" ||
          attrName == "dso_local" ||
          attrName == "addr_space" ||
          attrName == "alignment" ||
          attrName == "passthrough" ||
          attrName.starts_with("dlti.") ||
          attrName.starts_with("llvm.")) {
        // Skip - these are metadata, not user strings
        continue;
      }
    }
  });
}

std::unique_ptr<Pass> mlir::obs::createStringEncryptPass(llvm::StringRef key) {
  return std::make_unique<StringEncryptPass>(key.str());
}
