#include "Obfuscator/Passes.h"
#include "Obfuscator/Config.h"
#include "CIR/Passes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Conversion/Passes.h"

int main(int argc, char **argv) {
  // Register all standard MLIR dialects
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);

  // Explicitly register dialects we use (redundant but explicit)
  registry.insert<mlir::func::FuncDialect,
                  mlir::LLVM::LLVMDialect,
                  mlir::scf::SCFDialect,
                  mlir::arith::ArithDialect,
                  mlir::memref::MemRefDialect,
                  mlir::affine::AffineDialect>();


  mlir::registerAllPasses();

  // Register CIR passes (Layer 1.5)
  mlir::cir::registerCIRPasses();


  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return std::make_unique<mlir::obs::StringEncryptPass>();
  });

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return std::make_unique<mlir::obs::SymbolObfuscatePass>();
  });

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return std::make_unique<mlir::obs::CryptoHashPass>();
  });

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return std::make_unique<mlir::obs::ConstantObfuscationPass>();
  });

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return std::make_unique<mlir::obs::SCFObfuscatePass>();
  });

  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return std::make_unique<mlir::obs::ImportObfuscationPass>();
  });

  
  llvm::outs() << "MLIR Obfuscator Tool\n";
  llvm::outs() << "MLIR/LLVM Version: " << MLIR_VERSION_STRING << "\n";
  llvm::outs() << "Supported Frontend: ClangIR (LLVM 22 native)\n";
  llvm::outs() << "\n";

  
  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "MLIR Obfuscator\n", registry));
}
