#include "Obfuscator/Passes.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/Compiler.h"

// Global static pass registrations - constructed when plugin loads
// This makes passes available as --string-encrypt and --symbol-obfuscate CLI flags
static ::mlir::PassRegistration<::mlir::obs::StringEncryptPass> stringEncryptReg;
static ::mlir::PassRegistration<::mlir::obs::SymbolObfuscatePass> symbolObfuscateReg;
