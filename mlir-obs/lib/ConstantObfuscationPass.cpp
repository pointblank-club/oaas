#include "Obfuscator/Passes.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"

#include <string>
#include <cstring>
#include <random>

using namespace mlir;
using namespace mlir::obs;

namespace {

/// XOR encrypt for strings
static std::string xorEncrypt(const std::string &input, const std::string &key) {
  std::string out = input;
  for (size_t i = 0; i < input.size(); i++) {
    out[i] = input[i] ^ key[i % key.size()];
  }
  return out;
}

/// Obfuscate integer constant using XOR and arithmetic transformations
static int64_t obfuscateInteger(int64_t value, const std::string &key) {
  // Use key to generate obfuscation parameters
  std::seed_seq seq(key.begin(), key.end());
  std::mt19937 rng(seq);

  int64_t xor_mask = rng();
  int64_t add_offset = rng() % 1000;

  // Transform: value = ((original ^ xor_mask) + add_offset)
  // To recover: original = (value - add_offset) ^ xor_mask
  return (value ^ xor_mask) + add_offset;
}

/// Obfuscate float constant using bit manipulation
static double obfuscateFloat(double value, const std::string &key) {
  // Use key to generate obfuscation XOR mask
  std::seed_seq seq(key.begin(), key.end());
  std::mt19937_64 rng(seq);

  uint64_t xor_mask = rng();

  // Bit-level XOR on the float representation
  uint64_t bits;
  std::memcpy(&bits, &value, sizeof(double));
  bits ^= xor_mask;

  double obfuscated;
  std::memcpy(&obfuscated, &bits, sizeof(double));

  return obfuscated;
}

} // namespace

void ConstantObfuscationPass::runOnOperation() {
  ModuleOp module = getOperation();
  MLIRContext *ctx = module.getContext();
  OpBuilder builder(ctx);

  // ============================================================================
  // STEP 1: Obfuscate String Literals in LLVM GlobalOp (actual data)
  // ============================================================================
  module.walk([&](LLVM::GlobalOp globalOp) {
    // Check if this is a string constant
    if (auto strAttr = globalOp.getValueAttr()) {
      if (auto stringAttr = strAttr.dyn_cast<StringAttr>()) {
        std::string original = stringAttr.getValue().str();
        std::string encrypted = xorEncrypt(original, key);

        // Replace the global's initializer with encrypted string
        globalOp.setValueAttr(StringAttr::get(ctx, encrypted));
      }
    }
  });

  // ============================================================================
  // STEP 2: Obfuscate Integer and Float Constants in Func Dialect Operations
  // ============================================================================
  module.walk([&](Operation *op) {
    // Skip non-Func dialect operations to maintain compatibility
    if (!isa<func::FuncOp>(op->getParentOp()) && !isa<ModuleOp>(op->getParentOp())) {
      return;
    }

    bool changed = false;
    SmallVector<NamedAttribute> newAttrs;

    for (auto &attr : op->getAttrs()) {
      StringRef attrName = attr.getName().getValue();

      // Skip symbol names, function references, and callee (preserve semantics)
      if (attrName == "sym_name" ||
          attrName == "function_ref" ||
          attrName == "callee" ||
          attrName == "sym_visibility") {
        newAttrs.push_back(attr);
        continue;
      }

      // ========================================================================
      // Obfuscate String Attributes (metadata, not global data)
      // ========================================================================
      if (auto strAttr = attr.getValue().dyn_cast<StringAttr>()) {
        std::string original = strAttr.getValue().str();
        std::string encrypted = xorEncrypt(original, key);

        auto newValue = StringAttr::get(ctx, encrypted);
        newAttrs.emplace_back(attr.getName(), newValue);
        changed = true;
        continue;
      }

      // ========================================================================
      // Obfuscate Integer Attributes
      // ========================================================================
      if (auto intAttr = attr.getValue().dyn_cast<IntegerAttr>()) {
        int64_t original = intAttr.getInt();
        int64_t obfuscated = obfuscateInteger(original, key);

        auto newValue = IntegerAttr::get(intAttr.getType(), obfuscated);
        newAttrs.emplace_back(attr.getName(), newValue);
        changed = true;
        continue;
      }

      // ========================================================================
      // Obfuscate Float Attributes
      // ========================================================================
      if (auto floatAttr = attr.getValue().dyn_cast<FloatAttr>()) {
        double original = floatAttr.getValueAsDouble();
        double obfuscated = obfuscateFloat(original, key);

        auto newValue = FloatAttr::get(floatAttr.getType(), obfuscated);
        newAttrs.emplace_back(attr.getName(), newValue);
        changed = true;
        continue;
      }

      // ========================================================================
      // Obfuscate Dense Element Attributes (arrays of constants)
      // ========================================================================
      if (auto denseAttr = attr.getValue().dyn_cast<DenseElementsAttr>()) {
        // Check if it's an integer array
        if (denseAttr.getElementType().isInteger()) {
          SmallVector<int64_t> obfuscatedValues;
          for (auto val : denseAttr.getValues<APInt>()) {
            int64_t original = val.getSExtValue();
            int64_t obfuscated = obfuscateInteger(original, key);
            obfuscatedValues.push_back(obfuscated);
          }

          auto newValue = DenseElementsAttr::get(
              denseAttr.getType().cast<ShapedType>(),
              ArrayRef<int64_t>(obfuscatedValues)
          );
          newAttrs.emplace_back(attr.getName(), newValue);
          changed = true;
          continue;
        }

        // Check if it's a float array
        if (denseAttr.getElementType().isa<FloatType>()) {
          SmallVector<double> obfuscatedValues;
          for (auto val : denseAttr.getValues<APFloat>()) {
            double original = val.convertToDouble();
            double obfuscated = obfuscateFloat(original, key);
            obfuscatedValues.push_back(obfuscated);
          }

          auto newValue = DenseElementsAttr::get(
              denseAttr.getType().cast<ShapedType>(),
              ArrayRef<double>(obfuscatedValues)
          );
          newAttrs.emplace_back(attr.getName(), newValue);
          changed = true;
          continue;
        }
      }

      // No change -> keep original
      newAttrs.push_back(attr);
    }

    if (changed) {
      op->setAttrs(DictionaryAttr::get(ctx, newAttrs));
    }
  });

  // ============================================================================
  // STEP 3: Obfuscate Constants in LLVM Dialect Operations (Func compatible)
  // ============================================================================
  module.walk([&](LLVM::ConstantOp constOp) {
    // Obfuscate integer constants
    if (auto intAttr = constOp.getValue().dyn_cast<IntegerAttr>()) {
      int64_t original = intAttr.getInt();
      int64_t obfuscated = obfuscateInteger(original, key);

      auto newValue = IntegerAttr::get(intAttr.getType(), obfuscated);
      constOp.setValueAttr(newValue);
    }

    // Obfuscate float constants
    if (auto floatAttr = constOp.getValue().dyn_cast<FloatAttr>()) {
      double original = floatAttr.getValueAsDouble();
      double obfuscated = obfuscateFloat(original, key);

      auto newValue = FloatAttr::get(floatAttr.getType(), obfuscated);
      constOp.setValueAttr(newValue);
    }
  });
}

std::unique_ptr<Pass> mlir::obs::createConstantObfuscationPass(llvm::StringRef key) {
  return std::make_unique<ConstantObfuscationPass>(key.str());
}
