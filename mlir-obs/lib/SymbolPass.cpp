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

static std::string generateObfuscatedName(std::mt19937 &rng) {
  std::uniform_int_distribution<uint32_t> dist(0, 0xFFFFFFFF);
  uint32_t num = dist(rng);

  char buffer[16];
  snprintf(buffer, sizeof(buffer), "f_%08x", num);
  return std::string(buffer);
}

}

void SymbolObfuscatePass::runOnOperation() {
  ModuleOp module = getOperation();
  MLIRContext *ctx = module.getContext();

  bool hasFuncDialect = false;
  bool hasLLVMDialect = false;

  module.walk([&](Operation *op) {
    if (llvm::isa<func::FuncOp>(op)) {
      hasFuncDialect = true;
    } else if (llvm::isa<LLVM::LLVMFuncOp>(op)) {
      hasLLVMDialect = true;
    }
  });

  if (hasFuncDialect) {
    processFuncDialect();
  }
  if (hasLLVMDialect) {
    processLLVMDialect();
  }

}

void SymbolObfuscatePass::processFuncDialect() {
  ModuleOp module = getOperation();
  MLIRContext *ctx = module.getContext();
  SymbolTable symbolTable(module);

  std::seed_seq seq(key.begin(), key.end());
  std::mt19937 rng(seq);

  llvm::StringMap<std::string> renameMap;

  module.walk([&](func::FuncOp func) {
    StringRef oldName = func.getSymName();

    if (oldName == "main")
      return;

    if (func.isDeclaration())
      return;

    if (renameMap.find(oldName) == renameMap.end()) {
      std::string newName = generateObfuscatedName(rng);
      renameMap[oldName] = newName;
    }
  });

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

  module.walk([&](func::FuncOp func) {
    StringRef oldName = func.getSymName();
    auto it = renameMap.find(oldName);
    if (it != renameMap.end()) {
      symbolTable.setSymbolName(func, it->second);
    }
  });
}

void SymbolObfuscatePass::processLLVMDialect() {
  ModuleOp module = getOperation();
  MLIRContext *ctx = module.getContext();
  SymbolTable symbolTable(module);

  std::seed_seq seq(key.begin(), key.end());
  std::mt19937 rng(seq);

  llvm::StringMap<std::string> renameMap;

  module.walk([&](LLVM::LLVMFuncOp func) {
    StringRef oldName = func.getSymName();

    if (oldName == "main")
      return;

    if (func.isExternal())
      return;

    if (renameMap.find(oldName) == renameMap.end()) {
      std::string newName = generateObfuscatedName(rng);
      renameMap[oldName] = newName;
    }
  });

  module.walk([&](LLVM::GlobalOp globalOp) {
    StringRef oldName = globalOp.getSymName();

    if (oldName.starts_with("llvm.") || oldName.starts_with("__obfs_"))
      return;

    if (renameMap.find(oldName) == renameMap.end()) {
      std::uniform_int_distribution<uint32_t> dist(0, 0xFFFFFFFF);
      uint32_t num = dist(rng);
      char buffer[16];
      snprintf(buffer, sizeof(buffer), "g_%08x", num);
      renameMap[oldName] = std::string(buffer);
    }
  });

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
      if (auto flatSymAttr = llvm::dyn_cast<FlatSymbolRefAttr>(attr.getValue())) {
        StringRef old = flatSymAttr.getValue();
        auto it = renameMap.find(old);
        if (it != renameMap.end()) {
          auto newRef = FlatSymbolRefAttr::get(ctx, it->second);
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

  module.walk([&](LLVM::LLVMFuncOp func) {
    StringRef oldName = func.getSymName();
    auto it = renameMap.find(oldName);
    if (it != renameMap.end()) {
      symbolTable.setSymbolName(func, it->second);
    }
  });

  module.walk([&](LLVM::GlobalOp globalOp) {
    StringRef oldName = globalOp.getSymName();
    auto it = renameMap.find(oldName);
    if (it != renameMap.end()) {
      symbolTable.setSymbolName(globalOp, it->second);
    }
  });
}

std::unique_ptr<Pass> mlir::obs::createSymbolObfuscatePass(llvm::StringRef key) {
  return std::make_unique<SymbolObfuscatePass>(key.str());
}
