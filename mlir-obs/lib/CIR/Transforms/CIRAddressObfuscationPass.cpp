

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include <cstdint>
#include <ctime>
#include <memory>
#include <random>
#include <string>



namespace mlir {
namespace cir {

namespace CIROps {
  // Placeholder operation names - adjust based on actual CIR dialect
  static constexpr llvm::StringLiteral LoadOpName = "cir.load";
  static constexpr llvm::StringLiteral StoreOpName = "cir.store";
  static constexpr llvm::StringLiteral PtrAddOpName = "cir.ptr_add";
  static constexpr llvm::StringLiteral GetElementPtrOpName = "cir.gep";
  static constexpr llvm::StringLiteral CastOpName = "cir.cast";
}

class KeyGenerator {
public:
  // Use compile-time hash of timestamp and other constants
  static constexpr uint64_t generateKey() {
    constexpr const char* timestamp = __TIME__ __DATE__;
    return hashString(timestamp) ^ MAGIC_CONSTANT_1 ^ MAGIC_CONSTANT_2;
  }

private:
  static constexpr uint64_t MAGIC_CONSTANT_1 = 0x9E3779B97F4A7C15ULL; // Golden ratio
  static constexpr uint64_t MAGIC_CONSTANT_2 = 0x517CC1B727220A95ULL; // Random prime

  // Compile-time FNV-1a hash
  static constexpr uint64_t hashString(const char* str, uint64_t hash = 0xCBF29CE484222325ULL) {
    return (*str == '\0') ? hash : hashString(str + 1, (hash ^ static_cast<uint64_t>(*str)) * 0x100000001B3ULL);
  }
};


class CIRAddressObfuscationPass
    : public PassWrapper<CIRAddressObfuscationPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CIRAddressObfuscationPass)

  CIRAddressObfuscationPass() = default;
  CIRAddressObfuscationPass(bool enabled) : obfuscationEnabled(enabled) {}

  StringRef getArgument() const final { return "cir-address-obf"; }
  StringRef getDescription() const final {
    return "Apply address-level obfuscation to CIR pointer operations";
  }

  void runOnOperation() override {
    // Check if obfuscation is enabled (controlled by frontend toggle)
    if (!obfuscationEnabled) {
      // No-op when disabled
      return;
    }

    ModuleOp module = getOperation();
    MLIRContext* context = &getContext();

    obfuscationKey = KeyGenerator::generateKey();

    // Walk all operations and apply obfuscation
    module.walk([&](Operation* op) {
      StringRef opName = op->getName().getStringRef();

      if (opName == CIROps::LoadOpName) {
        obfuscateLoadOp(op);
      } else if (opName == CIROps::StoreOpName) {
        obfuscateStoreOp(op);
      } else if (opName == CIROps::PtrAddOpName) {
        obfuscatePtrArithmetic(op);
      } else if (opName == CIROps::GetElementPtrOpName) {
        obfuscateGetElementPtr(op);
      }
    });
  }

  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<arith::ArithDialect>();
  }

private:
  bool obfuscationEnabled = true;
  uint64_t obfuscationKey = 0;

  Value createKeyConstant(OpBuilder& builder, Location loc, Type indexType) {
    // Create an arith.constant with the obfuscation key
    IntegerAttr keyAttr = builder.getIntegerAttr(indexType, obfuscationKey);
    return builder.create<arith::ConstantOp>(loc, indexType, keyAttr);
  }

  void obfuscateLoadOp(Operation* loadOp) {
    OpBuilder builder(loadOp);
    Location loc = loadOp->getLoc();

    if (loadOp->getNumOperands() >= 2) {
      Value indexOperand = loadOp->getOperand(1); // Index/offset operand
      Type indexType = indexOperand.getType();

      // Create key constant
      Value keyConst = createKeyConstant(builder, loc, indexType);

      // Create XOR operation: masked_index = index XOR key
      Value maskedIndex = builder.create<arith::XOrIOp>(loc, indexOperand, keyConst);

      // Replace the index operand with masked version
      loadOp->setOperand(1, maskedIndex);
    }
  }

  void obfuscateStoreOp(Operation* storeOp) {
    OpBuilder builder(storeOp);
    Location loc = storeOp->getLoc();

    if (storeOp->getNumOperands() >= 3) {
      Value indexOperand = storeOp->getOperand(2); // Index/offset operand
      Type indexType = indexOperand.getType();

      // Create key constant
      Value keyConst = createKeyConstant(builder, loc, indexType);

      // Create XOR operation
      Value maskedIndex = builder.create<arith::XOrIOp>(loc, indexOperand, keyConst);

      // Replace the index operand
      storeOp->setOperand(2, maskedIndex);
    }
  }

  void obfuscatePtrArithmetic(Operation* ptrAddOp) {
    OpBuilder builder(ptrAddOp);
    Location loc = ptrAddOp->getLoc();

    if (ptrAddOp->getNumOperands() >= 2) {
      Value offsetOperand = ptrAddOp->getOperand(1);
      Type offsetType = offsetOperand.getType();

      // Create key constant
      Value keyConst = createKeyConstant(builder, loc, offsetType);

      // Create XOR operation
      Value maskedOffset = builder.create<arith::XOrIOp>(loc, offsetOperand, keyConst);

      // Replace offset operand
      ptrAddOp->setOperand(1, maskedOffset);
    }
  }

  void obfuscateGetElementPtr(Operation* gepOp) {
    OpBuilder builder(gepOp);
    Location loc = gepOp->getLoc();

    for (unsigned i = 1; i < gepOp->getNumOperands(); ++i) {
      Value indexOperand = gepOp->getOperand(i);
      Type indexType = indexOperand.getType();

      // Create key constant
      Value keyConst = createKeyConstant(builder, loc, indexType);

      // Create XOR operation
      Value maskedIndex = builder.create<arith::XOrIOp>(loc, indexOperand, keyConst);

      // Replace index operand
      gepOp->setOperand(i, maskedIndex);
    }
  }
};

/// Factory function to create the pass with configuration
std::unique_ptr<Pass> createCIRAddressObfuscationPass(bool enabled = true) {
  return std::make_unique<CIRAddressObfuscationPass>(enabled);
}

} // namespace cir
} // namespace mlir
