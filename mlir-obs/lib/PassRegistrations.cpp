#include "Obfuscator/Passes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "llvm/Support/Compiler.h"

// Plugin entry point for dynamic loading by mlir-opt
extern "C" LLVM_ATTRIBUTE_WEAK ::mlir::PassPluginLibraryInfo
mlirGetPassPluginInfo() {
  return {
    MLIR_PLUGIN_API_VERSION,
    "MLIRObfuscation",
    LLVM_VERSION_STRING,
    [](::mlir::PassRegistry &registry) {
      // Register passes with their factory functions
      registry.registerPass([]() -> std::unique_ptr<::mlir::Pass> {
        return std::make_unique<::mlir::obs::StringEncryptPass>();
      });
      registry.registerPass([]() -> std::unique_ptr<::mlir::Pass> {
        return std::make_unique<::mlir::obs::SymbolObfuscatePass>();
      });
    }
  };
}
