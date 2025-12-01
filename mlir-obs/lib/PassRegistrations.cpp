#include "Obfuscator/Passes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Tools/Plugins/PassPlugin.h"

namespace mlir {
namespace obs {

// Register passes with MLIR
void registerStringEncryptPass() {
  PassRegistration<StringEncryptPass>();
}

void registerSymbolObfuscatePass() {
  PassRegistration<SymbolObfuscatePass>();
}

void registerCryptoHashPass() {
  PassRegistration<CryptoHashPass>();
}

void registerConstantObfuscationPass() {
  PassRegistration<ConstantObfuscationPass>();
}

} // namespace obs
} // namespace mlir

// Plugin entry point - mlir-opt expects this exact function signature
extern "C" LLVM_ATTRIBUTE_WEAK ::mlir::PassPluginLibraryInfo
mlirGetPassPluginInfo() {
  return {MLIR_PLUGIN_API_VERSION, "MLIRObfuscation", LLVM_VERSION_STRING,
          []() {
            mlir::obs::registerStringEncryptPass();
            mlir::obs::registerSymbolObfuscatePass();
            mlir::obs::registerCryptoHashPass();
            mlir::obs::registerConstantObfuscationPass();
          }};
}
