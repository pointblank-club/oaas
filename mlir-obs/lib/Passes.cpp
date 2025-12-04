#include "Obfuscator/Passes.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "mlir/IR/MLIRContext.h"

#include <string>

using namespace mlir;
using namespace mlir::obs;

namespace {

static std::string xorEncrypt(const std::string &input, const std::string &key) {
  std::string out = input;
  for (size_t i = 0; i < input.size(); i++) {
    out[i] = input[i] ^ key[i % key.size()];
  }
  return out;
}

} // namespace


void StringEncryptPass::runOnOperation() {
  ModuleOp module = getOperation();
  MLIRContext *ctx = module.getContext();

  module.walk([&](Operation *op) {
    bool changed = false;
    SmallVector<NamedAttribute> newAttrs;

    for (auto &attr : op->getAttrs()) {
      StringRef attrName = attr.getName().getValue();

      // Skip all metadata/structural attributes - these should never be encrypted
      if (attrName == "sym_name" ||
          attrName == "sym_visibility" ||
          attrName == "function_ref" ||
          attrName == "callee" ||
          attrName == "llvm.target_triple" ||
          attrName == "llvm.ident" ||
          attrName == "llvm.module_asm" ||
          attrName == "target_cpu" ||
          attrName == "tune_cpu" ||
          attrName == "target_features" ||
          attrName == "frame_pointer" ||
          attrName == "uwtable_kind" ||
          attrName == "linkage" ||
          attrName == "visibility" ||
          attrName == "dso_local" ||
          attrName == "addr_space" ||
          attrName == "alignment" ||
          attrName == "passthrough" ||
          attrName.starts_with("dlti.") ||
          attrName.starts_with("llvm.")) {
        newAttrs.push_back(attr);
        continue;
      }

      // Only encrypt string attributes that are actual data (like "value" for global strings)
      if (attrName == "value") {
        if (auto strAttr = llvm::dyn_cast<StringAttr>(attr.getValue())) {
          std::string original = strAttr.getValue().str();
          std::string encrypted = xorEncrypt(original, key);

          auto newValue = StringAttr::get(ctx, encrypted);
          newAttrs.emplace_back(attr.getName(), newValue);
          changed = true;
        } else {
          newAttrs.push_back(attr);
        }
      } else {
        newAttrs.push_back(attr);
      }
    }

    if (changed) {
      op->setAttrs(DictionaryAttr::get(ctx, newAttrs));
    }
  });
}

std::unique_ptr<Pass> mlir::obs::createStringEncryptPass(llvm::StringRef key) {
  return std::make_unique<StringEncryptPass>(key.str());
}
