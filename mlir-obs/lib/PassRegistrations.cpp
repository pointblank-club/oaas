#include "Obfuscator/Passes.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

// Register passes when plugin loads
namespace {
inline void registerPasses() {
  PassRegistration<mlir::obs::StringEncryptPass>();
  PassRegistration<mlir::obs::SymbolObfuscatePass>();
}
}

// Call registration at static initialization time
static bool passesRegistered = (registerPasses(), true);
