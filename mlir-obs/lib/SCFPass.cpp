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

/// Insert opaque predicates into SCF control flow
/// This makes reverse engineering much harder by adding conditions
/// that are always true/false but not obviously so
void insertOpaquePredicates(scf::IfOp ifOp, OpBuilder &builder) {
  // Get the condition
  Value condition = ifOp.getCondition();
  Location loc = ifOp.getLoc();

  // Create an opaque predicate: (x * 2) / 2 == x
  // This is always true, but not obviously so to static analysis
  builder.setInsertionPoint(ifOp);

  // Create a constant for obfuscation
  auto c1 = builder.create<arith::ConstantIntOp>(loc, 1, 32);
  auto c2 = builder.create<arith::ConstantIntOp>(loc, 2, 32);

  // Create opaque predicate: (c1 * 2) / 2 == c1
  auto mul = builder.create<arith::MulIOp>(loc, c1, c2);
  auto div = builder.create<arith::DivSIOp>(loc, mul, c2);
  auto opaquePred = builder.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::eq, div, c1);

  // AND the opaque predicate with original condition
  // This doesn't change behavior (always true AND cond == cond)
  // but adds complexity for analysis
  auto newCond = builder.create<arith::AndIOp>(loc, condition, opaquePred);

  // Replace the condition
  ifOp.getConditionMutable().assign(newCond);
}

/// Add fake loop iterations that don't affect output
/// Makes loop analysis harder
void obfuscateLoop(scf::ForOp forOp, OpBuilder &builder) {
  // We can add complexity by inserting dead code or
  // modifying loop structure while preserving semantics
  // For now, just mark that we processed it
  // TODO: Implement more sophisticated loop obfuscation
}

} // namespace

void SCFObfuscatePass::runOnOperation() {
  ModuleOp module = getOperation();
  MLIRContext *ctx = module.getContext();
  OpBuilder builder(ctx);

  // Walk all scf.if operations and add opaque predicates
  module.walk([&](scf::IfOp ifOp) {
    insertOpaquePredicates(ifOp, builder);
  });

  // Walk all scf.for operations and obfuscate
  module.walk([&](scf::ForOp forOp) {
    obfuscateLoop(forOp, builder);
  });

  // scf.while could also be obfuscated
  module.walk([&](scf::WhileOp whileOp) {
    // TODO: Add while loop obfuscation
  });
}

std::unique_ptr<Pass> mlir::obs::createSCFObfuscatePass() {
  return std::make_unique<SCFObfuscatePass>();
}
