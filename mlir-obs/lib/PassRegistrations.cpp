#include "Obfuscator/Passes.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace mlir::obs;

namespace {
  // Register StringEncryptPass
  PassRegistration<StringEncryptPass> stringReg([]() {
    return std::make_unique<StringEncryptPass>();
  });

  // Register SymbolObfuscatePass
  PassRegistration<SymbolObfuscatePass> symbolReg([]() {
    return std::make_unique<SymbolObfuscatePass>();
  });
}

// This function must exist for MLIR plugins to register their passes
extern "C" void mlirRegisterPasses() {
  // Nothing required here — the static registration above is enough,
  // but MLIR needs this exported function symbol to locate the plugin.
}
