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
      if (attrName == "sym_name" || 
          attrName == "function_ref" ||
          attrName == "callee") {
        newAttrs.push_back(attr);
        continue;  // Skip to next attribute
      }

      // Original encryption logic (unchanged)
      if (auto strAttr = llvm::dyn_cast<StringAttr>(attr.getValue())) {
        std::string original = strAttr.getValue().str();
        std::string encrypted = xorEncrypt(original, key);

        auto newValue = StringAttr::get(ctx, encrypted);
        newAttrs.emplace_back(attr.getName(), newValue);
        changed = true;
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
