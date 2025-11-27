#include "Obfuscator/Passes.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/Compiler.h"

using namespace mlir;

// Global pass registrations - must be outside anonymous namespace
LLVM_ATTRIBUTE_USED
static PassRegistration<mlir::obs::StringEncryptPass> stringEncryptPass;

LLVM_ATTRIBUTE_USED
static PassRegistration<mlir::obs::SymbolObfuscatePass> symbolObfuscatePass;

// Force library initialization by providing a function that mlir-opt can call
extern "C" LLVM_ATTRIBUTE_USED void mlirRegisterPasses() {
  // Nothing needed - registration happens via static constructors above
  // But this symbol ensures the library gets loaded and initialized
}
