#include "Obfuscator/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"

using namespace mlir;

namespace mlir {
namespace obs {

// Static registration of passes - this happens at plugin load time
void registerObfuscationPasses() {
  registerPass([]() -> std::unique_ptr<Pass> {
    return std::make_unique<StringEncryptPass>();
  });

  registerPass([]() -> std::unique_ptr<Pass> {
    return std::make_unique<SymbolObfuscatePass>();
  });
}

} // namespace obs
} // namespace mlir

// MLIR plugin entry point
extern "C" LLVM_ATTRIBUTE_WEAK void mlirRegisterPasses() {
  mlir::obs::registerObfuscationPasses();
}
