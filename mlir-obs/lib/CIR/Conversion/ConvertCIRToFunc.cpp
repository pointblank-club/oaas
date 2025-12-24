

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include <memory>



namespace mlir {
namespace cir {


class CIRToFuncTypeConverter : public TypeConverter {
public:
  CIRToFuncTypeConverter() {
    
    addConversion([](Type type) { return type; });

    
    addConversion([](Type type) -> std::optional<Type> {
      

      
      if (auto cirPtrType = type.dyn_cast<Type>()) {
        
        
        if (auto intType = type.dyn_cast<IntegerType>()) {
          return MemRefType::get({ShapedType::kDynamic}, intType);
        }
      }
      return std::nullopt;
    });

    
    addConversion([this](FunctionType type) -> std::optional<Type> {
      SmallVector<Type> inputs;
      SmallVector<Type> results;

      if (failed(convertTypes(type.getInputs(), inputs)))
        return std::nullopt;
      if (failed(convertTypes(type.getResults(), results)))
        return std::nullopt;

      return FunctionType::get(type.getContext(), inputs, results);
    });

    
    addTargetMaterialization([](OpBuilder& builder, Type resultType,
                                ValueRange inputs, Location loc) -> Value {
      if (inputs.size() == 1)
        return inputs[0];
      return nullptr;
    });

    addSourceMaterialization([](OpBuilder& builder, Type resultType,
                                ValueRange inputs, Location loc) -> Value {
      if (inputs.size() == 1)
        return inputs[0];
      return nullptr;
    });
  }
};

class CIRLoadOpConversion : public OpConversionPattern<Operation> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(Operation* op, ArrayRef<Value> operands,
                                ConversionPatternRewriter& rewriter) const override {
    
    if (op->getName().getStringRef() != "cir.load")
      return failure();

    Location loc = op->getLoc();

    
    if (operands.size() < 2)
      return failure();

    Value basePtr = operands[0]; 
    Value index = operands[1];    

    
    Value loadResult = rewriter.create<memref::LoadOp>(loc, basePtr, ValueRange{index});

   
    rewriter.replaceOp(op, loadResult);

    return success();
  }
};


class CIRStoreOpConversion : public OpConversionPattern<Operation> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(Operation* op, ArrayRef<Value> operands,
                                ConversionPatternRewriter& rewriter) const override {
    
    if (op->getName().getStringRef() != "cir.store")
      return failure();

    Location loc = op->getLoc();

    
    if (operands.size() < 3)
      return failure();

    Value valueToStore = operands[0];
    Value basePtr = operands[1];     
    Value index = operands[2];       

    
    rewriter.create<memref::StoreOp>(loc, valueToStore, basePtr, ValueRange{index});

    
    rewriter.eraseOp(op);

    return success();
  }
};


class CIRPtrAddOpConversion : public OpConversionPattern<Operation> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(Operation* op, ArrayRef<Value> operands,
                                ConversionPatternRewriter& rewriter) const override {
    if (op->getName().getStringRef() != "cir.ptr_add")
      return failure();

    Location loc = op->getLoc();

    if (operands.size() < 2)
      return failure();

    Value basePtr = operands[0];
    Value offset = operands[1];

    
    auto memrefType = basePtr.getType().dyn_cast<MemRefType>();
    if (!memrefType)
      return failure();

    
    SmallVector<OpFoldResult> offsets = {offset};
    SmallVector<OpFoldResult> sizes = {rewriter.getIndexAttr(ShapedType::kDynamic)};
    SmallVector<OpFoldResult> strides = {rewriter.getIndexAttr(1)};

    Value subview = rewriter.create<memref::SubViewOp>(
        loc, basePtr, offsets, sizes, strides);

    rewriter.replaceOp(op, subview);

    return success();
  }
};


class CIRGetElementPtrOpConversion : public OpConversionPattern<Operation> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(Operation* op, ArrayRef<Value> operands,
                                ConversionPatternRewriter& rewriter) const override {
    if (op->getName().getStringRef() != "cir.gep")
      return failure();

    Location loc = op->getLoc();

    if (operands.empty())
      return failure();

   
    Value base = operands[0];

    
    SmallVector<Value> indices(operands.begin() + 1, operands.end());

   

    auto memrefType = base.getType().dyn_cast<MemRefType>();
    if (!memrefType)
      return failure();

    
    if (indices.size() == 1) {
      
      SmallVector<OpFoldResult> offsets = {indices[0]};
      SmallVector<OpFoldResult> sizes = {rewriter.getIndexAttr(ShapedType::kDynamic)};
      SmallVector<OpFoldResult> strides = {rewriter.getIndexAttr(1)};

      Value subview = rewriter.create<memref::SubViewOp>(
          loc, base, offsets, sizes, strides);

      rewriter.replaceOp(op, subview);
    } else {
      
      Value linearOffset = nullptr;

      for (size_t i = 0; i < indices.size(); ++i) {
        
        if (linearOffset) {
          linearOffset = rewriter.create<arith::AddIOp>(loc, linearOffset, indices[i]);
        } else {
          linearOffset = indices[i];
        }
      }

      
      SmallVector<OpFoldResult> offsets = {linearOffset};
      SmallVector<OpFoldResult> sizes = {rewriter.getIndexAttr(ShapedType::kDynamic)};
      SmallVector<OpFoldResult> strides = {rewriter.getIndexAttr(1)};

      Value subview = rewriter.create<memref::SubViewOp>(
          loc, base, offsets, sizes, strides);

      rewriter.replaceOp(op, subview);
    }

    return success();
  }
};

class CIRFuncOpConversion : public OpConversionPattern<Operation> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(Operation* op, ArrayRef<Value> operands,
                                ConversionPatternRewriter& rewriter) const override {
    if (op->getName().getStringRef() != "cir.func")
      return failure();

    Location loc = op->getLoc();

   
    auto funcOp = dyn_cast<func::FuncOp>(op);
    if (!funcOp)
      return failure();

    
    TypeConverter* typeConverter = getTypeConverter();
    auto convertedType = typeConverter->convertType(funcOp.getFunctionType());
    if (!convertedType)
      return failure();

    auto newFuncOp = rewriter.create<func::FuncOp>(
        loc, funcOp.getName(), convertedType.cast<FunctionType>());

    
    for (auto namedAttr : funcOp->getAttrs()) {
      if (namedAttr.getName() != "type" && namedAttr.getName() != "function_type")
        newFuncOp->setAttr(namedAttr.getName(), namedAttr.getValue());
    }

    
    Region& oldRegion = funcOp.getBody();
    Region& newRegion = newFuncOp.getBody();

    if (!oldRegion.empty()) {
      rewriter.inlineRegionBefore(oldRegion, newRegion, newRegion.end());

      
      TypeConverter* converter = getTypeConverter();
      if (failed(rewriter.convertRegionTypes(&newRegion, *converter)))
        return failure();
    }

    rewriter.replaceOp(op, newFuncOp);

    return success();
  }
};

/// Pattern to convert cir.return → func.return
class CIRReturnOpConversion : public OpConversionPattern<Operation> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(Operation* op, ArrayRef<Value> operands,
                                ConversionPatternRewriter& rewriter) const override {
    if (op->getName().getStringRef() != "cir.return")
      return failure();

    // Create func.return with converted operands
    rewriter.replaceOpWithNewOp<func::ReturnOp>(op, operands);

    return success();
  }
};


class ConvertCIRToFuncPass
    : public PassWrapper<ConvertCIRToFuncPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertCIRToFuncPass)

  StringRef getArgument() const final { return "convert-cir-to-func"; }
  StringRef getDescription() const final {
    return "Convert CIR dialect to Func dialect";
  }

  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<func::FuncDialect, arith::ArithDialect,
                    cf::ControlFlowDialect, memref::MemRefDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext* context = &getContext();

    // Step 1: Create type converter
    CIRToFuncTypeConverter typeConverter;

    // Step 2: Define conversion target
    ConversionTarget target(*context);

    // Mark target dialects as legal
    target.addLegalDialect<func::FuncDialect, arith::ArithDialect,
                          cf::ControlFlowDialect, memref::MemRefDialect,
                          BuiltinDialect>();

    // Mark CIR dialect and operations as illegal (must be converted)
    target.addIllegalDialect<DenseStringElementsAttr>();  // Placeholder for CIR dialect

    // Mark specific CIR operations as illegal
    target.addDynamicallyLegalOp<Operation>([](Operation* op) {
      StringRef opName = op->getName().getStringRef();
      return !opName.starts_with("cir.");
    });

    // Step 3: Register conversion patterns
    RewritePatternSet patterns(context);

    patterns.add<CIRLoadOpConversion>(typeConverter, context);
    patterns.add<CIRStoreOpConversion>(typeConverter, context);
    patterns.add<CIRPtrAddOpConversion>(typeConverter, context);
    patterns.add<CIRGetElementPtrOpConversion>(typeConverter, context);
    patterns.add<CIRFuncOpConversion>(typeConverter, context);
    patterns.add<CIRReturnOpConversion>(typeConverter, context);

    // Step 4: Apply partial conversion
    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
      return;
    }
  }
};

/// Factory function to create the conversion pass
std::unique_ptr<Pass> createConvertCIRToFuncPass() {
  return std::make_unique<ConvertCIRToFuncPass>();
}

} // namespace cir
} // namespace mlir
