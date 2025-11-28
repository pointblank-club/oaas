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

// Use PassRegistration to make passes available as CLI flags
namespace {
void registerObfuscationPasses() {
  // Register with PassRegistration to get CLI flags
  static bool initialized = false;
  if (!initialized) {
    ::mlir::PassRegistration<::mlir::obs::StringEncryptPass>();
    ::mlir::PassRegistration<::mlir::obs::SymbolObfuscatePass>();
    initialized = true;
  }
}
} // namespace

// Pass plugin registration - mlir-opt calls this when loading the plugin
extern "C" LLVM_ATTRIBUTE_WEAK PassPluginLibraryInfo mlirGetPassPluginInfo() {
  return {MLIR_PLUGIN_API_VERSION, "ObfuscationPasses", LLVM_VERSION_STRING,
          registerObfuscationPasses};
}
