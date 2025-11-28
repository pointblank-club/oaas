#include "Obfuscator/Passes.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace obs {

// Register passes - these will be automatically registered when the plugin loads
void registerStringEncryptPass() {
  PassRegistration<StringEncryptPass>();
}

void registerSymbolObfuscatePass() {
  PassRegistration<SymbolObfuscatePass>();
}

} // namespace obs
} // namespace mlir

// Plugin initialization - called when mlir-opt loads the plugin
extern "C" void mlirRegisterPasses() {
  mlir::obs::registerStringEncryptPass();
  mlir::obs::registerSymbolObfuscatePass();
}
