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

/// XOR encrypt a string for hiding function names
static std::string xorEncrypt(const std::string &input, const std::string &key) {
  std::string out = input;
  for (size_t i = 0; i < input.size(); i++) {
    out[i] = input[i] ^ key[i % key.size()];
  }
  return out;
}

/// List of functions that should NOT be transformed (would cause issues)
static bool shouldSkipFunction(StringRef name) {
  // Skip dlopen/dlsym - would cause infinite recursion
  if (name == "dlopen" || name == "dlsym" || name == "dlclose" || name == "dlerror")
    return true;

  // Skip low-level libc functions that dlsym itself might use
  if (name == "__libc_start_main" || name == "__cxa_atexit" ||
      name == "__cxa_finalize" || name == "__gmon_start__")
    return true;

  // Skip C++ exception handling
  if (name.starts_with("__cxa_") || name.starts_with("__gxx_"))
    return true;

  // Skip LLVM intrinsics
  if (name.starts_with("llvm."))
    return true;

  // Skip our obfuscator functions
  if (name.starts_with("__obfs_"))
    return true;

  // Skip _Unwind functions (exception handling)
  if (name.starts_with("_Unwind") || name.starts_with("_ZSt"))
    return true;

  return false;
}

/// Determine which library a function comes from
static std::string getLibraryForFunction(StringRef name) {
  // Math functions -> libm
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

  // pthread functions -> libpthread
  if (name.starts_with("pthread_"))
    return "libpthread.so.0";

  // Default to libc
  return "libc.so.6";
}

} // namespace


void ImportObfuscationPass::runOnOperation() {
  ModuleOp module = getOperation();
  MLIRContext *ctx = module.getContext();
  OpBuilder builder(ctx);

  // Collect external functions that we'll transform
  std::vector<LLVM::LLVMFuncOp> externalFuncs;

  module.walk([&](LLVM::LLVMFuncOp func) {
    // Only process external functions (declarations without bodies)
    if (!func.isExternal())
      return;

    StringRef name = func.getSymName();

    // Skip functions that shouldn't be transformed
    if (shouldSkipFunction(name))
      return;

    externalFuncs.push_back(func);
  });

  if (externalFuncs.empty())
    return;

  // Get or create the dlopen and dlsym declarations
  LLVM::LLVMFuncOp dlopenFunc = nullptr;
  LLVM::LLVMFuncOp dlsymFunc = nullptr;

  // Find existing declarations or we'll create them
  module.walk([&](LLVM::LLVMFuncOp func) {
    if (func.getSymName() == "dlopen")
      dlopenFunc = func;
    else if (func.getSymName() == "dlsym")
      dlsymFunc = func;
  });

  // Set insertion point at the start of the module
  builder.setInsertionPointToStart(module.getBody());

  // Create dlopen declaration if not exists
  // void* dlopen(const char* filename, int flags)
  if (!dlopenFunc) {
    auto ptrType = LLVM::LLVMPointerType::get(ctx);
    auto i32Type = IntegerType::get(ctx, 32);
    auto dlopenType = LLVM::LLVMFunctionType::get(ptrType, {ptrType, i32Type}, false);
    dlopenFunc = builder.create<LLVM::LLVMFuncOp>(
        module.getLoc(), "dlopen", dlopenType, LLVM::Linkage::External);
  }

  // Create dlsym declaration if not exists
  // void* dlsym(void* handle, const char* symbol)
  if (!dlsymFunc) {
    auto ptrType = LLVM::LLVMPointerType::get(ctx);
    auto dlsymType = LLVM::LLVMFunctionType::get(ptrType, {ptrType, ptrType}, false);
    dlsymFunc = builder.create<LLVM::LLVMFuncOp>(
        module.getLoc(), "dlsym", dlsymType, LLVM::Linkage::External);
  }

  // Track library handles we've created
  llvm::StringMap<LLVM::GlobalOp> libraryHandles;

  // Process each external function
  for (LLVM::LLVMFuncOp extFunc : externalFuncs) {
    StringRef funcName = extFunc.getSymName();
    std::string libName = getLibraryForFunction(funcName);
    Location loc = extFunc.getLoc();

    // Get or create global for library handle
    LLVM::GlobalOp libHandle;
    auto it = libraryHandles.find(libName);
    if (it != libraryHandles.end()) {
      libHandle = it->second;
    } else {
      // Create global for library handle: void* __obfs_lib_<name> = NULL
      builder.setInsertionPointToStart(module.getBody());
      std::string handleName = "__obfs_lib_" +
          std::to_string(std::hash<std::string>{}(libName) & 0xFFFFFF);

      auto ptrType = LLVM::LLVMPointerType::get(ctx);
      libHandle = builder.create<LLVM::GlobalOp>(
          loc, ptrType, /*isConstant=*/false, LLVM::Linkage::Internal,
          handleName, Attribute());

      libraryHandles[libName] = libHandle;
    }

    // Create global for cached function pointer: void* __obfs_fp_<name> = NULL
    builder.setInsertionPointToStart(module.getBody());
    std::string fpName = "__obfs_fp_" + funcName.str();
    auto ptrType = LLVM::LLVMPointerType::get(ctx);

    auto fpGlobal = builder.create<LLVM::GlobalOp>(
        loc, ptrType, /*isConstant=*/false, LLVM::Linkage::Internal,
        fpName, Attribute());

    // Create global for function name string (encrypted)
    std::string encFuncName = encryptStrings ?
        xorEncrypt(funcName.str() + '\0', key) : (funcName.str() + '\0');
    std::string funcNameGlobalName = "__obfs_fn_" + funcName.str();

    auto i8Type = IntegerType::get(ctx, 8);
    auto strType = LLVM::LLVMArrayType::get(i8Type, encFuncName.size());

    auto funcNameGlobal = builder.create<LLVM::GlobalOp>(
        loc, strType, /*isConstant=*/true, LLVM::Linkage::Internal,
        funcNameGlobalName, builder.getStringAttr(encFuncName));

    // Create global for library name string
    std::string libNameStr = libName + '\0';
    if (encryptStrings) {
      libNameStr = xorEncrypt(libNameStr, key);
    }
    std::string libNameGlobalName = "__obfs_ln_" +
        std::to_string(std::hash<std::string>{}(libName) & 0xFFFFFF);

    // Check if lib name global already exists
    LLVM::GlobalOp libNameGlobal;
    module.walk([&](LLVM::GlobalOp g) {
      if (g.getSymName() == libNameGlobalName)
        libNameGlobal = g;
    });

    if (!libNameGlobal) {
      auto libStrType = LLVM::LLVMArrayType::get(i8Type, libNameStr.size());
      libNameGlobal = builder.create<LLVM::GlobalOp>(
          loc, libStrType, /*isConstant=*/true, LLVM::Linkage::Internal,
          libNameGlobalName, builder.getStringAttr(libNameStr));
    }

    // Get the function type from the external function
    auto funcType = extFunc.getFunctionType();

    // Create wrapper function: __obfs_wrap_<name>
    std::string wrapperName = "__obfs_wrap_" + funcName.str();

    builder.setInsertionPointToStart(module.getBody());
    auto wrapperFunc = builder.create<LLVM::LLVMFuncOp>(
        loc, wrapperName, funcType, LLVM::Linkage::Internal);

    // Create entry block for wrapper
    Block *entryBlock = wrapperFunc.addEntryBlock(builder);
    builder.setInsertionPointToStart(entryBlock);

    auto i64Type = IntegerType::get(ctx, 64);
    auto i32Type = IntegerType::get(ctx, 32);

    // Load cached function pointer
    Value fpAddr = builder.create<LLVM::AddressOfOp>(loc, ptrType, fpGlobal.getSymName());
    Value cachedFp = builder.create<LLVM::LoadOp>(loc, ptrType, fpAddr);

    // Check if null
    Value nullPtr = builder.create<LLVM::ZeroOp>(loc, ptrType);
    Value isNull = builder.create<LLVM::ICmpOp>(
        loc, LLVM::ICmpPredicate::eq, cachedFp, nullPtr);

    // Create blocks for the if-then-else
    Block *resolveBlock = wrapperFunc.addBlock();
    Block *callBlock = wrapperFunc.addBlock();

    builder.create<LLVM::CondBrOp>(loc, isNull, resolveBlock, callBlock);

    // Resolve block: call dlopen + dlsym
    builder.setInsertionPointToStart(resolveBlock);

    // Load library handle
    Value libHandleAddr = builder.create<LLVM::AddressOfOp>(
        loc, ptrType, libHandle.getSymName());
    Value libHandleVal = builder.create<LLVM::LoadOp>(loc, ptrType, libHandleAddr);

    // Check if library handle is null
    Value libIsNull = builder.create<LLVM::ICmpOp>(
        loc, LLVM::ICmpPredicate::eq, libHandleVal, nullPtr);

    Block *dlopenBlock = wrapperFunc.addBlock();
    Block *dlsymBlock = wrapperFunc.addBlock();

    builder.create<LLVM::CondBrOp>(loc, libIsNull, dlopenBlock, dlsymBlock);

    // dlopen block
    builder.setInsertionPointToStart(dlopenBlock);

    // Get library name string address
    Value libNameAddr = builder.create<LLVM::AddressOfOp>(
        loc, ptrType, libNameGlobal.getSymName());

    // If encrypted, decrypt first (for now, we'll skip decryption and just use the name)
    // In production, you'd call __obfs_decrypt here

    // RTLD_LAZY = 1
    Value rtldLazy = builder.create<LLVM::ConstantOp>(
        loc, i32Type, builder.getI32IntegerAttr(1));

    // Call dlopen
    auto dlopenCall = builder.create<LLVM::CallOp>(
        loc, TypeRange{ptrType}, "dlopen", ValueRange{libNameAddr, rtldLazy});
    Value newHandle = dlopenCall.getResult();

    // Store the handle
    builder.create<LLVM::StoreOp>(loc, newHandle, libHandleAddr);

    builder.create<LLVM::BrOp>(loc, ValueRange{}, dlsymBlock);

    // dlsym block
    builder.setInsertionPointToStart(dlsymBlock);

    // Reload handle (might have been set in dlopen block)
    Value handleForDlsym = builder.create<LLVM::LoadOp>(loc, ptrType, libHandleAddr);

    // Get function name string address
    Value funcNameAddr = builder.create<LLVM::AddressOfOp>(
        loc, ptrType, funcNameGlobal.getSymName());

    // Call dlsym
    auto dlsymCall = builder.create<LLVM::CallOp>(
        loc, TypeRange{ptrType}, "dlsym", ValueRange{handleForDlsym, funcNameAddr});
    Value resolvedFp = dlsymCall.getResult();

    // Cache the function pointer
    builder.create<LLVM::StoreOp>(loc, resolvedFp, fpAddr);

    builder.create<LLVM::BrOp>(loc, ValueRange{}, callBlock);

    // Call block: call through function pointer
    builder.setInsertionPointToStart(callBlock);

    // Reload the function pointer
    Value finalFp = builder.create<LLVM::LoadOp>(loc, ptrType, fpAddr);

    // Collect arguments from wrapper function
    SmallVector<Value> args;
    for (auto arg : entryBlock->getArguments()) {
      args.push_back(arg);
    }

    // Call indirectly through the function pointer
    // For indirect calls, prepend the function pointer to the arguments
    auto resultType = funcType.getReturnType();
    SmallVector<Value> callArgs;
    callArgs.push_back(finalFp);  // Function pointer as first operand
    callArgs.append(args.begin(), args.end());

    if (llvm::isa<LLVM::LLVMVoidType>(resultType)) {
      // Indirect call with void return - use CallOp with no callee symbol
      auto indirectCall = builder.create<LLVM::CallOp>(
          loc, TypeRange{}, ValueRange{callArgs});
      (void)indirectCall;
      builder.create<LLVM::ReturnOp>(loc, ValueRange{});
    } else {
      // Indirect call with return value
      auto callResult = builder.create<LLVM::CallOp>(
          loc, TypeRange{resultType}, ValueRange{callArgs});
      builder.create<LLVM::ReturnOp>(loc, callResult.getResults());
    }

    // Now replace all calls to the external function with calls to wrapper
    // We need to collect first, then modify to avoid iterator invalidation
    SmallVector<LLVM::CallOp> callsToReplace;

    module.walk([&](LLVM::CallOp callOp) {
      auto callee = callOp.getCallee();
      if (callee && *callee == funcName) {
        callsToReplace.push_back(callOp);
      }
    });

    // Replace calls
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
