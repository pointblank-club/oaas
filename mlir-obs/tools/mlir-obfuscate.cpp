#include "Obfuscator/Passes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"

int main(int argc, char **argv) {
  mlir::registerAllDialects();
  mlir::registerAllPasses();

  // Register our obfuscation passes
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return std::make_unique<mlir::obs::StringEncryptPass>();
  });

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return std::make_unique<mlir::obs::SymbolObfuscatePass>();
  });

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "MLIR Obfuscator\n", mlirGetDialectRegistrationHooks()));
}
