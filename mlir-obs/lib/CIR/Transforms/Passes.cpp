

#include "CIR/Passes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"

namespace mlir {
namespace cir {

/// Register all CIR dialect passes
void registerCIRPasses() {
  // Register Layer 1.5 address obfuscation pass
  PassRegistration<CIRAddressObfuscationPass>(
      "cir-address-obf",
      "Apply address-level obfuscation to CIR pointer operations",
      []() -> std::unique_ptr<Pass> {
        return createCIRAddressObfuscationPass(true);
      });

  // Register CIR to Func conversion pass
  PassRegistration<ConvertCIRToFuncPass>(
      "convert-cir-to-func",
      "Convert CIR dialect to Func dialect",
      []() -> std::unique_ptr<Pass> {
        return createConvertCIRToFuncPass();
      });
}

} // namespace cir
} // namespace mlir

static mlir::PassRegistration<mlir::cir::CIRAddressObfuscationPass> cirAddressObfPass;
static mlir::PassRegistration<mlir::cir::ConvertCIRToFuncPass> cirToFuncPass;
