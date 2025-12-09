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

static bool shouldSkipFunction(StringRef name) {
  if (name == "dlopen" || name == "dlsym" || name == "dlclose" || name == "dlerror")
    return true;

  if (name == "__libc_start_main" || name == "__cxa_atexit" ||
      name == "__cxa_finalize" || name == "__gmon_start__")
    return true;

  if (name.starts_with("__cxa_") || name.starts_with("__gxx_"))
    return true;

  if (name.starts_with("llvm."))
    return true;

  if (name.starts_with("__obfs_"))
    return true;

  if (name.starts_with("_Unwind") || name.starts_with("_ZSt"))
    return true;

  return false;
}

static std::string getLibraryForFunction(StringRef name) {
  static const char* mathFuncs[] = {
    "sin", "cos", "tan", "asin", "acos", "atan", "atan2",
    "sinh", "cosh", "tanh", "exp", "log", "log10", "log2",
    "pow", "sqrt", "cbrt", "ceil", "floor", "round", "fabs",
    "fmod", "remainder", "fmax", "fmin", "hypot", "erf", "erfc",
    "lgamma", "tgamma", "j0", "j1", "jn", "y0", "y1", "yn"
  };
  for (const char* f : mathFuncs) {
    if (name == f || name == (std::string(f) + "f") || name == (std::string(f) + "l"))
      return "libm.so.6";
  }

  if (name.starts_with("pthread_"))
    return "libpthread.so.0";

  return "libc.so.6";
}

}

void ImportObfuscationPass::runOnOperation() {
  ModuleOp module = getOperation();
  MLIRContext *ctx = module.getContext();
  OpBuilder builder(ctx);

  std::vector<LLVM::LLVMFuncOp> externalFuncs;

  module.walk([&](LLVM::LLVMFuncOp func) {
    if (!func.isExternal())
      return;

    StringRef name = func.getSymName();

    if (shouldSkipFunction(name))
      return;

    externalFuncs.push_back(func);
  });

  if (externalFuncs.empty())
    return;

  LLVM::LLVMFuncOp dlopenFunc = nullptr;
  LLVM::LLVMFuncOp dlsymFunc = nullptr;

  module.walk([&](LLVM::LLVMFuncOp func) {
    if (func.getSymName() == "dlopen")
      dlopenFunc = func;
    else if (func.getSymName() == "dlsym")
      dlsymFunc = func;
  });

  builder.setInsertionPointToStart(module.getBody());

  if (!dlopenFunc) {
    auto ptrType = LLVM::LLVMPointerType::get(ctx);
    auto i32Type = IntegerType::get(ctx, 32);
    auto dlopenType = LLVM::LLVMFunctionType::get(ptrType, {ptrType, i32Type}, false);
    dlopenFunc = builder.create<LLVM::LLVMFuncOp>(
        module.getLoc(), "dlopen", dlopenType, LLVM::Linkage::External);
  }

  if (!dlsymFunc) {
    auto ptrType = LLVM::LLVMPointerType::get(ctx);
    auto dlsymType = LLVM::LLVMFunctionType::get(ptrType, {ptrType, ptrType}, false);
    dlsymFunc = builder.create<LLVM::LLVMFuncOp>(
        module.getLoc(), "dlsym", dlsymType, LLVM::Linkage::External);
  }

  llvm::StringMap<LLVM::GlobalOp> libraryHandles;

  for (LLVM::LLVMFuncOp extFunc : externalFuncs) {
    StringRef funcName = extFunc.getSymName();
    std::string libName = getLibraryForFunction(funcName);
    Location loc = extFunc.getLoc();

    LLVM::GlobalOp libHandle;
    auto it = libraryHandles.find(libName);
    if (it != libraryHandles.end()) {
      libHandle = it->second;
    } else {
      builder.setInsertionPointToStart(module.getBody());
      std::string handleName = "__obfs_lib_" +
          std::to_string(std::hash<std::string>{}(libName) & 0xFFFFFF);

      auto ptrType = LLVM::LLVMPointerType::get(ctx);
      libHandle = builder.create<LLVM::GlobalOp>(
          loc, ptrType, false, LLVM::Linkage::Internal,
          handleName, Attribute());

      libraryHandles[libName] = libHandle;
    }

    builder.setInsertionPointToStart(module.getBody());
    std::string fpName = "__obfs_fp_" + funcName.str();
    auto ptrType = LLVM::LLVMPointerType::get(ctx);

    auto fpGlobal = builder.create<LLVM::GlobalOp>(
        loc, ptrType, false, LLVM::Linkage::Internal,
        fpName, Attribute());

    std::string encFuncName = encryptStrings ?
        xorEncrypt(funcName.str() + '\0', key) : (funcName.str() + '\0');
    std::string funcNameGlobalName = "__obfs_fn_" + funcName.str();

    auto i8Type = IntegerType::get(ctx, 8);
    auto strType = LLVM::LLVMArrayType::get(i8Type, encFuncName.size());

    auto funcNameGlobal = builder.create<LLVM::GlobalOp>(
        loc, strType, true, LLVM::Linkage::Internal,
        funcNameGlobalName, builder.getStringAttr(encFuncName));

    std::string libNameStr = libName + '\0';
    if (encryptStrings) {
      libNameStr = xorEncrypt(libNameStr, key);
    }
    std::string libNameGlobalName = "__obfs_ln_" +
        std::to_string(std::hash<std::string>{}(libName) & 0xFFFFFF);

    LLVM::GlobalOp libNameGlobal;
    module.walk([&](LLVM::GlobalOp g) {
      if (g.getSymName() == libNameGlobalName)
        libNameGlobal = g;
    });

    if (!libNameGlobal) {
      auto libStrType = LLVM::LLVMArrayType::get(i8Type, libNameStr.size());
      libNameGlobal = builder.create<LLVM::GlobalOp>(
          loc, libStrType, true, LLVM::Linkage::Internal,
          libNameGlobalName, builder.getStringAttr(libNameStr));
    }

    auto funcType = extFunc.getFunctionType();

    std::string wrapperName = "__obfs_wrap_" + funcName.str();

    builder.setInsertionPointToStart(module.getBody());
    auto wrapperFunc = builder.create<LLVM::LLVMFuncOp>(
        loc, wrapperName, funcType, LLVM::Linkage::Internal);

    Block *entryBlock = wrapperFunc.addEntryBlock(builder);
    builder.setInsertionPointToStart(entryBlock);

    auto i64Type = IntegerType::get(ctx, 64);
    auto i32Type = IntegerType::get(ctx, 32);

    Value fpAddr = builder.create<LLVM::AddressOfOp>(loc, ptrType, fpGlobal.getSymName());
    Value cachedFp = builder.create<LLVM::LoadOp>(loc, ptrType, fpAddr);

    Value nullPtr = builder.create<LLVM::ZeroOp>(loc, ptrType);
    Value isNull = builder.create<LLVM::ICmpOp>(
        loc, LLVM::ICmpPredicate::eq, cachedFp, nullPtr);

    Block *resolveBlock = wrapperFunc.addBlock();
    Block *callBlock = wrapperFunc.addBlock();

    builder.create<LLVM::CondBrOp>(loc, isNull, resolveBlock, callBlock);

    builder.setInsertionPointToStart(resolveBlock);

    Value libHandleAddr = builder.create<LLVM::AddressOfOp>(
        loc, ptrType, libHandle.getSymName());
    Value libHandleVal = builder.create<LLVM::LoadOp>(loc, ptrType, libHandleAddr);

    Value libIsNull = builder.create<LLVM::ICmpOp>(
        loc, LLVM::ICmpPredicate::eq, libHandleVal, nullPtr);

    Block *dlopenBlock = wrapperFunc.addBlock();
    Block *dlsymBlock = wrapperFunc.addBlock();

    builder.create<LLVM::CondBrOp>(loc, libIsNull, dlopenBlock, dlsymBlock);

    builder.setInsertionPointToStart(dlopenBlock);

    Value libNameAddr = builder.create<LLVM::AddressOfOp>(
        loc, ptrType, libNameGlobal.getSymName());

    Value rtldLazy = builder.create<LLVM::ConstantOp>(
        loc, i32Type, builder.getI32IntegerAttr(1));

    auto dlopenCall = builder.create<LLVM::CallOp>(
        loc, TypeRange{ptrType}, "dlopen", ValueRange{libNameAddr, rtldLazy});
    Value newHandle = dlopenCall.getResult();

    builder.create<LLVM::StoreOp>(loc, newHandle, libHandleAddr);

    builder.create<LLVM::BrOp>(loc, ValueRange{}, dlsymBlock);

    builder.setInsertionPointToStart(dlsymBlock);

    Value handleForDlsym = builder.create<LLVM::LoadOp>(loc, ptrType, libHandleAddr);

    Value funcNameAddr = builder.create<LLVM::AddressOfOp>(
        loc, ptrType, funcNameGlobal.getSymName());

    auto dlsymCall = builder.create<LLVM::CallOp>(
        loc, TypeRange{ptrType}, "dlsym", ValueRange{handleForDlsym, funcNameAddr});
    Value resolvedFp = dlsymCall.getResult();

    builder.create<LLVM::StoreOp>(loc, resolvedFp, fpAddr);

    builder.create<LLVM::BrOp>(loc, ValueRange{}, callBlock);

    builder.setInsertionPointToStart(callBlock);

    Value finalFp = builder.create<LLVM::LoadOp>(loc, ptrType, fpAddr);

    SmallVector<Value> args;
    for (auto arg : entryBlock->getArguments()) {
      args.push_back(arg);
    }

    auto resultType = funcType.getReturnType();
    SmallVector<Value> callArgs;
    callArgs.push_back(finalFp);
    callArgs.append(args.begin(), args.end());

    if (llvm::isa<LLVM::LLVMVoidType>(resultType)) {
      auto indirectCall = builder.create<LLVM::CallOp>(
          loc, TypeRange{}, ValueRange{callArgs});
      (void)indirectCall;
      builder.create<LLVM::ReturnOp>(loc, ValueRange{});
    } else {
      auto callResult = builder.create<LLVM::CallOp>(
          loc, TypeRange{resultType}, ValueRange{callArgs});
      builder.create<LLVM::ReturnOp>(loc, callResult.getResults());
    }

    SmallVector<LLVM::CallOp> callsToReplace;

    module.walk([&](LLVM::CallOp callOp) {
      auto callee = callOp.getCallee();
      if (callee && *callee == funcName) {
        callsToReplace.push_back(callOp);
      }
    });

    for (LLVM::CallOp callOp : callsToReplace) {
      builder.setInsertionPoint(callOp);

      auto newCall = builder.create<LLVM::CallOp>(
          callOp.getLoc(),
          callOp.getResultTypes(),
          wrapperName,
          callOp.getOperands());

      callOp.replaceAllUsesWith(newCall);
      callOp.erase();
    }
  }
}

std::unique_ptr<Pass> mlir::obs::createImportObfuscationPass(
    bool encryptStrings, llvm::StringRef key) {
  return std::make_unique<ImportObfuscationPass>(encryptStrings, key.str());
}
