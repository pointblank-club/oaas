#include "Obfuscator/Passes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Tools/Plugins/PassPlugin.h"

namespace mlir {
namespace obs {

inline void registerPasses() {
  registerPass([]() -> std::unique_ptr<Pass> {
    return std::make_unique<StringEncryptPass>();
  });

  registerPass([]() -> std::unique_ptr<Pass> {
    return std::make_unique<SymbolObfuscatePass>();
  });

  registerPass([]() -> std::unique_ptr<Pass> {
    return std::make_unique<SCFObfuscatePass>();
  });
}

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

void registerSCFObfuscatePass() {
  PassRegistration<SCFObfuscatePass>();
}

void registerImportObfuscationPass() {
  PassRegistration<ImportObfuscationPass>();
}

}
}

extern "C" LLVM_ATTRIBUTE_WEAK ::mlir::PassPluginLibraryInfo
mlirGetPassPluginInfo() {
  return {MLIR_PLUGIN_API_VERSION, "MLIRObfuscation", LLVM_VERSION_STRING,
          []() {
            mlir::obs::registerStringEncryptPass();
            mlir::obs::registerSymbolObfuscatePass();
            mlir::obs::registerCryptoHashPass();
            mlir::obs::registerConstantObfuscationPass();
            mlir::obs::registerSCFObfuscatePass();
            mlir::obs::registerImportObfuscationPass();
          }};
}
