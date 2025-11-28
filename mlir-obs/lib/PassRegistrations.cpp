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

// Static registration using PassRegistration template
// This makes passes available as CLI flags (--string-encrypt, --symbol-obfuscate)
inline void registerObfuscationPasses() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return std::make_unique<::mlir::obs::StringEncryptPass>();
  });

  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return std::make_unique<::mlir::obs::SymbolObfuscatePass>();
  });
}

// Pass plugin registration - mlir-opt calls this when loading the plugin
extern "C" LLVM_ATTRIBUTE_WEAK PassPluginLibraryInfo mlirGetPassPluginInfo() {
  return {MLIR_PLUGIN_API_VERSION, "ObfuscationPasses", LLVM_VERSION_STRING,
          registerObfuscationPasses};
}
