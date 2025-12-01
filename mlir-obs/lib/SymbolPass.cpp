#include "Obfuscator/Passes.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

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

  // Detect which dialect we're working with
  bool hasFuncDialect = false;
  bool hasLLVMDialect = false;

  module.walk([&](Operation *op) {
    if (llvm::isa<func::FuncOp>(op)) {
      hasFuncDialect = true;
    } else if (llvm::isa<LLVM::LLVMFuncOp>(op)) {
      hasLLVMDialect = true;
    }
  });

  // Process appropriate dialect(s)
  if (hasFuncDialect) {
    processFuncDialect();
  }
  if (hasLLVMDialect) {
    processLLVMDialect();
  }

  // If neither found, this is still okay - module might be empty or use other dialects
}

// Process func::FuncOp (ClangIR / high-level MLIR)
void SymbolObfuscatePass::processFuncDialect() {
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
      if (auto symAttr = llvm::dyn_cast<SymbolRefAttr>(attr.getValue())) {
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

// Process LLVM::LLVMFuncOp (post-lowering to LLVM dialect)
void SymbolObfuscatePass::processLLVMDialect() {
  ModuleOp module = getOperation();
  MLIRContext *ctx = module.getContext();
  SymbolTable symbolTable(module);

  // RNG seeded by key (use same seed for consistency)
  std::seed_seq seq(key.begin(), key.end());
  std::mt19937 rng(seq);

  // Step 1: Collect rename map for LLVM func definitions
  llvm::StringMap<std::string> renameMap;

  module.walk([&](LLVM::LLVMFuncOp func) {
    StringRef oldName = func.getSymName();
    // Don't rename main
    // if (oldName == "main")
    //   return;

    if (renameMap.find(oldName) == renameMap.end()) {
      std::string newName = generateObfuscatedName(rng);
      renameMap[oldName] = newName;
    }
  });

  // Step 2: Update symbol references
  module.walk([&](Operation *op) {
    SmallVector<NamedAttribute> updatedAttrs;
    bool changed = false;

    for (auto &attr : op->getAttrs()) {
      if (auto symAttr = llvm::dyn_cast<SymbolRefAttr>(attr.getValue())) {
        StringRef old = symAttr.getRootReference();
        auto it = renameMap.find(old);
        if (it != renameMap.end()) {
          auto newRef = SymbolRefAttr::get(ctx, it->second);
          updatedAttrs.emplace_back(attr.getName(), newRef);
          changed = true;
          continue;
        }
      }
      updatedAttrs.push_back(attr);
    }

    if (changed) {
      op->setAttrs(DictionaryAttr::get(ctx, updatedAttrs));
    }
  });

  // Step 3: Rename LLVM function definitions
  module.walk([&](LLVM::LLVMFuncOp func) {
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
