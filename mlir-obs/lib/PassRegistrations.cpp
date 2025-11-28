#include "Obfuscator/Passes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "llvm/Support/Compiler.h"

// Define PassPluginLibraryInfo if not available
#ifndef MLIR_PLUGIN_API_VERSION
#define MLIR_PLUGIN_API_VERSION 1
#endif

struct PassPluginLibraryInfo {
  uint32_t APIVersion;
  const char *PluginName;
  const char *PluginVersion;
  void (*RegisterPasses)();
};

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
