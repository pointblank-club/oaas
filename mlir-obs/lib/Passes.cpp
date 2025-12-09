#include "Obfuscator/Passes.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Builders.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

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

struct EncryptedGlobalInfo {
  std::string globalName;
  size_t originalLength;
};

}

void StringEncryptPass::runOnOperation() {
  ModuleOp module = getOperation();
  MLIRContext *ctx = module.getContext();
  OpBuilder builder(ctx);

  std::vector<EncryptedGlobalInfo> encryptedGlobals;

  module.walk([&](LLVM::GlobalOp globalOp) {
    StringRef symName = globalOp.getSymName();

    if (symName.starts_with("__obfs_") || symName.starts_with("llvm."))
      return;

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

    if (globalOp.getSection().has_value())
      return;

    if (auto strAttr = globalOp.getValueAttr()) {
      if (auto stringAttr = llvm::dyn_cast<StringAttr>(strAttr)) {
        std::string original = stringAttr.getValue().str();

        if (original.empty())
          return;

        if (original.size() < 2)
          return;

        std::string encrypted = xorEncrypt(original, key);

        globalOp.setValueAttr(StringAttr::get(ctx, encrypted));

        globalOp.setConstant(false);

        encryptedGlobals.push_back({symName.str(), original.size()});
      }
    }
  });

  if (encryptedGlobals.empty())
    return;

  builder.setInsertionPointToStart(module.getBody());
  Location loc = module.getLoc();

  auto i8Type = IntegerType::get(ctx, 8);
  auto i32Type = IntegerType::get(ctx, 32);
  auto i64Type = IntegerType::get(ctx, 64);
  auto i8PtrType = LLVM::LLVMPointerType::get(ctx);
  auto voidType = LLVM::LLVMVoidType::get(ctx);

  if (!module.lookupSymbol<LLVM::GlobalOp>("__obfs_key")) {
    auto keyArrayType = LLVM::LLVMArrayType::get(i8Type, key.size());
    auto keyGlobal = builder.create<LLVM::GlobalOp>(
        loc,
        keyArrayType,
        true,
        LLVM::Linkage::Private,
        "__obfs_key",
        builder.getStringAttr(key)
    );
    keyGlobal.setUnnamedAddr(LLVM::UnnamedAddr::Global);
  }

  if (!module.lookupSymbol<LLVM::LLVMFuncOp>("__obfs_decrypt")) {
    auto funcType = LLVM::LLVMFunctionType::get(voidType, {i8PtrType, i32Type}, false);
    auto decryptFunc = builder.create<LLVM::LLVMFuncOp>(
        loc, "__obfs_decrypt", funcType, LLVM::Linkage::Internal);
    decryptFunc.setNoInline(true);

    Block *entryBlock = decryptFunc.addEntryBlock(builder);
    builder.setInsertionPointToStart(entryBlock);

    Value strPtr = entryBlock->getArgument(0);
    Value len = entryBlock->getArgument(1);
    Value keyAddr = builder.create<LLVM::AddressOfOp>(loc, i8PtrType, "__obfs_key");

    Value zero32 = builder.create<LLVM::ConstantOp>(loc, i32Type, builder.getI32IntegerAttr(0));
    Value one32 = builder.create<LLVM::ConstantOp>(loc, i32Type, builder.getI32IntegerAttr(1));
    Value keyLenVal = builder.create<LLVM::ConstantOp>(loc, i32Type, builder.getI32IntegerAttr(key.size()));

    Value iPtr = builder.create<LLVM::AllocaOp>(loc, i8PtrType, i32Type, one32);
    builder.create<LLVM::StoreOp>(loc, zero32, iPtr);

    Block *loopCond = decryptFunc.addBlock();
    Block *loopBody = decryptFunc.addBlock();
    Block *loopEnd = decryptFunc.addBlock();

    builder.create<LLVM::BrOp>(loc, loopCond);

    builder.setInsertionPointToStart(loopCond);
    Value i = builder.create<LLVM::LoadOp>(loc, i32Type, iPtr);
    Value cond = builder.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::slt, i, len);
    builder.create<LLVM::CondBrOp>(loc, cond, loopBody, loopEnd);

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

  builder.setInsertionPointToEnd(module.getBody());
  auto initFuncType = LLVM::LLVMFunctionType::get(voidType, {}, false);

  if (!module.lookupSymbol<LLVM::LLVMFuncOp>("__obfs_init")) {
    auto initFunc = builder.create<LLVM::LLVMFuncOp>(
        loc, "__obfs_init", initFuncType, LLVM::Linkage::External);
    initFunc.setNoInline(true);

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

  builder.setInsertionPointToEnd(module.getBody());

  LLVM::GlobalCtorsOp existingCtors = nullptr;
  for (auto &op : module.getBody()->getOperations()) {
    if (auto ctorsOp = llvm::dyn_cast<LLVM::GlobalCtorsOp>(&op)) {
      existingCtors = ctorsOp;
      break;
    }
  }

  if (existingCtors) {
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

  module.walk([&](Operation *op) {
    if (llvm::isa<LLVM::GlobalOp>(op))
      return;

    for (auto &attr : op->getAttrs()) {
      StringRef attrName = attr.getName().getValue();

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
        continue;
      }
    }
  });
}

std::unique_ptr<Pass> mlir::obs::createStringEncryptPass(llvm::StringRef key) {
  return std::make_unique<StringEncryptPass>(key.str());
}
