#include "Obfuscator/Passes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "llvm/Support/Compiler.h"

using namespace mlir;

namespace {
void registerObfuscationPasses() {
  registerPass([]() -> std::unique_ptr<Pass> {
    return std::make_unique<mlir::obs::StringEncryptPass>();
  });

  registerPass([]() -> std::unique_ptr<Pass> {
    return std::make_unique<mlir::obs::SymbolObfuscatePass>();
  });
}
} // namespace

// This is the entry point that mlir-opt calls when loading the plugin
extern "C" LLVM_ATTRIBUTE_USED void mlirRegisterPasses() {
  registerObfuscationPasses();
}
