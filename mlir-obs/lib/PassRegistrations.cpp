#include "Obfuscator/Passes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"

using namespace mlir;
using namespace mlir::obs;

// Global registration - must be in global namespace for plugin loading
static PassRegistration<StringEncryptPass> stringReg;
static PassRegistration<SymbolObfuscatePass> symbolReg;
