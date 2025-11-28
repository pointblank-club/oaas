#include "Obfuscator/Passes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Tools/Plugins/PassPlugin.h"
#include "llvm/Support/Compiler.h"

using namespace mlir;

namespace mlir {
namespace obs {

// Register our obfuscation passes
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

// Pass plugin registration - this is what mlir-opt calls when loading
extern "C" LLVM_ATTRIBUTE_WEAK PassPluginLibraryInfo mlirGetPassPluginInfo() {
  return {MLIR_PLUGIN_API_VERSION, "ObfuscationPasses", LLVM_VERSION_STRING,
          []() { mlir::obs::registerObfuscationPasses(); }};
}
