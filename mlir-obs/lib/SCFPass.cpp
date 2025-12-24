#include "Obfuscator/Passes.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"

#include <random>

using namespace mlir;
using namespace mlir::obs;

namespace {

void insertOpaquePredicates(scf::IfOp ifOp, OpBuilder &builder) {
  Value condition = ifOp.getCondition();
  Location loc = ifOp.getLoc();

  builder.setInsertionPoint(ifOp);

  auto c1 = builder.create<arith::ConstantIntOp>(loc, 1, 32);
  auto c2 = builder.create<arith::ConstantIntOp>(loc, 2, 32);

  auto mul = builder.create<arith::MulIOp>(loc, c1, c2);
  auto div = builder.create<arith::DivSIOp>(loc, mul, c2);
  auto opaquePred = builder.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::eq, div, c1);

  auto newCond = builder.create<arith::AndIOp>(loc, condition, opaquePred);

  ifOp.getConditionMutable().assign(newCond);
}

void obfuscateLoop(scf::ForOp forOp, OpBuilder &builder) {
}

}

void SCFObfuscatePass::runOnOperation() {
  ModuleOp module = getOperation();
  MLIRContext *ctx = module.getContext();
  OpBuilder builder(ctx);

  module.walk([&](scf::IfOp ifOp) {
    insertOpaquePredicates(ifOp, builder);
  });

  module.walk([&](scf::ForOp forOp) {
    obfuscateLoop(forOp, builder);
  });

  module.walk([&](scf::WhileOp whileOp) {
  });
}

std::unique_ptr<Pass> mlir::obs::createSCFObfuscatePass() {
  return std::make_unique<SCFObfuscatePass>();
}
