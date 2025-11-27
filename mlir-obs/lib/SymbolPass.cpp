#include "Obfuscator/Passes.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

#include <random>

using namespace mlir;
using namespace mlir::obs;

namespace {

/// Utility: generate random obfuscated names (hex-based)
static std::string generateObfuscatedName(std::mt19937 &rng) {
  std::uniform_int_distribution<uint32_t> dist(0, 0xFFFFFFFF);
  uint32_t num = dist(rng);

  char buffer[16];
  snprintf(buffer, sizeof(buffer), "f_%08x", num);
  return std::string(buffer);
}

} // namespace

void SymbolObfuscatePass::runOnOperation() {
  ModuleOp module = getOperation();
  MLIRContext *ctx = module.getContext();
  SymbolTable symbolTable(module);

  // RNG seeded by key
  std::seed_seq seq(key.begin(), key.end());
  std::mt19937 rng(seq);

  // Step 1: Collect rename map for func definitions (oldName -> newName)
  llvm::StringMap<std::string> renameMap;

  module.walk([&](func::FuncOp func) {
    StringRef oldName = func.getSymName();
    // Don't rename main if you want to keep entry stable (optional)
    // if (oldName == "main")
    //   return;

    // Only assign a new name once per function
    if (renameMap.find(oldName) == renameMap.end()) {
      std::string newName = generateObfuscatedName(rng);
      renameMap[oldName] = newName;
    }
  });

  // Step 2: Update symbol references *before* renaming definitions
  module.walk([&](Operation *op) {
    SmallVector<NamedAttribute> updatedAttrs;
    bool changed = false;

    for (auto &attr : op->getAttrs()) {
      if (auto symAttr = attr.getValue().dyn_cast<SymbolRefAttr>()) {
        StringRef old = symAttr.getRootReference();
        auto it = renameMap.find(old);
        if (it != renameMap.end()) {
          auto newRef = SymbolRefAttr::get(ctx, it->second);
          updatedAttrs.emplace_back(attr.getName(), newRef);
          changed = true;
          continue;
        }
      }
      // No change -> keep original
      updatedAttrs.push_back(attr);
    }

    if (changed) {
      op->setAttrs(DictionaryAttr::get(ctx, updatedAttrs));
    }
  });

  // Step 3: Rename function definitions *after* updating uses
  module.walk([&](func::FuncOp func) {
    StringRef oldName = func.getSymName();
    auto it = renameMap.find(oldName);
    if (it != renameMap.end()) {
      symbolTable.setSymbolName(func, it->second);
    }
  });
}

std::unique_ptr<Pass> mlir::obs::createSymbolObfuscatePass(llvm::StringRef key) {
  return std::make_unique<SymbolObfuscatePass>(key.str());
}
