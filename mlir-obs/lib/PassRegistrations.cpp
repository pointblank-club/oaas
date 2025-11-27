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
