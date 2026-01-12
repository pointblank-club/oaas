//===- NopInsertion.cpp - NOP/Dead Code Insertion Pass -------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the NOP insertion pass. It inserts meaningless
// operations that have no semantic effect to increase code complexity.
//
// Types of dead code inserted:
// - Dead arithmetic chains: (x + 0) * 1 - 0
// - Dead XOR operations: x ^ 0
// - Dead AND operations: x & -1
// - Dead OR operations: x | 0
// - Useless variable copies
//
// SAFE FOR BINARY OBFUSCATION MODE (McSema lifted IR):
// - Does NOT modify control flow
// - Only adds dead/neutral instructions
// - CFG-neutral, works purely on instruction sequences
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Obfuscation/NopInsertion.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Obfuscation/Utils.h"
#include "llvm/IR/Intrinsics.h"

#define DEBUG_TYPE "nop-insertion"

// Command line options
static cl::opt<int> NopInsertRate(
    "nop_rate",
    cl::desc("Probability [%] of inserting NOPs after each instruction"),
    cl::value_desc("rate"), cl::init(50), cl::Optional);

static cl::opt<int> NopInsertCount(
    "nop_count",
    cl::desc("Maximum number of NOP sequences to insert per instruction"),
    cl::value_desc("count"), cl::init(2), cl::Optional);

// Binary-safe mode
static cl::opt<bool> BinarySafeMode(
    "nop_binary_safe",
    cl::desc("Enable binary-safe mode for McSema-lifted IR"),
    cl::init(false), cl::Optional);

// Stats
STATISTIC(NumNopsInserted, "Number of NOP sequences inserted");
STATISTIC(NumDeadArithmetic, "Number of dead arithmetic operations");
STATISTIC(NumDeadBitwise, "Number of dead bitwise operations");

namespace {

struct NopInsertion : public FunctionPass {
  static char ID;
  bool flag;

  NopInsertion() : FunctionPass(ID), flag(false) {}
  NopInsertion(bool flag) : FunctionPass(ID), flag(flag) {}

  // Helper: Check if this is a McSema-generated function
  bool isMcSemaFunction(Function *F) {
    StringRef Name = F->getName();
    return Name.starts_with("sub_") ||
           Name.starts_with("callback_") ||
           Name.starts_with("data_") ||
           Name.starts_with("ext_") ||
           Name.starts_with("__mcsema") ||
           Name.starts_with("__remill");
  }

  bool runOnFunction(Function &F) override;
  bool insertNops(Function *F);

  // Different NOP insertion strategies
  void insertDeadArithmetic(Instruction *I, IRBuilder<> &Builder);
  void insertDeadBitwise(Instruction *I, IRBuilder<> &Builder);
  void insertDeadCopy(Instruction *I, IRBuilder<> &Builder);
};

} // end anonymous namespace

char NopInsertion::ID = 0;
static RegisterPass<NopInsertion> X("nop-insertion", "NOP/Dead Code Insertion Pass");

Pass *llvm::createNopInsertion() {
  return new NopInsertion(false);
}

Pass *llvm::createNopInsertion(bool flag) {
  return new NopInsertion(flag);
}

bool NopInsertion::runOnFunction(Function &F) {
  // Binary-safe mode: Skip McSema-generated functions
  if (BinarySafeMode && isMcSemaFunction(&F)) {
    return false;
  }

  Function *Func = &F;

  if (toObfuscate(flag, Func, "nop")) {
    return insertNops(Func);
  }

  return false;
}

bool NopInsertion::insertNops(Function *F) {
  bool Changed = false;

  // Collect insertion points to avoid iterator invalidation
  std::vector<Instruction*> InsertionPoints;

  for (BasicBlock &BB : *F) {
    // Skip exception handling blocks
    if (BB.isLandingPad()) continue;

    for (Instruction &I : BB) {
      // Skip PHI nodes and terminators
      if (isa<PHINode>(&I)) continue;
      if (I.isTerminator()) continue;

      // Skip exception handling instructions
      if (isa<LandingPadInst>(&I) || isa<ResumeInst>(&I) ||
          isa<CatchPadInst>(&I) || isa<CleanupPadInst>(&I)) continue;

      // Randomly decide whether to insert NOPs
      if ((int)llvm::cryptoutils->get_range(100) < NopInsertRate) {
        InsertionPoints.push_back(&I);
      }
    }
  }

  // Insert NOPs at selected points
  for (Instruction *I : InsertionPoints) {
    // Insert after the instruction
    BasicBlock::iterator InsertPt = I->getIterator();
    ++InsertPt;

    if (InsertPt == I->getParent()->end()) continue;

    IRBuilder<> Builder(&*InsertPt);

    // Randomly select NOP type and insert multiple times
    int insertCount = llvm::cryptoutils->get_range(NopInsertCount) + 1;
    for (int i = 0; i < insertCount; ++i) {
      switch (llvm::cryptoutils->get_range(3)) {
        case 0:
          insertDeadArithmetic(I, Builder);
          break;
        case 1:
          insertDeadBitwise(I, Builder);
          break;
        case 2:
          insertDeadCopy(I, Builder);
          break;
      }
      ++NumNopsInserted;
      Changed = true;
    }
  }

  return Changed;
}

// Insert dead arithmetic: x + 0, x * 1, x - 0
void NopInsertion::insertDeadArithmetic(Instruction *I, IRBuilder<> &Builder) {
  // Find an integer value to work with
  Value *V = nullptr;
  for (unsigned i = 0; i < I->getNumOperands(); ++i) {
    if (I->getOperand(i)->getType()->isIntegerTy()) {
      V = I->getOperand(i);
      break;
    }
  }

  if (!V) return;

  Type *Ty = V->getType();

  // Create dead arithmetic chain: ((V + 0) * 1) - 0
  Value *Zero = ConstantInt::get(Ty, 0);
  Value *One = ConstantInt::get(Ty, 1);

  Value *Add = Builder.CreateAdd(V, Zero, "nop.add");
  Value *Mul = Builder.CreateMul(Add, One, "nop.mul");
  Value *Sub = Builder.CreateSub(Mul, Zero, "nop.sub");

  // The result is unused (dead code)
  (void)Sub;

  ++NumDeadArithmetic;
}

// Insert dead bitwise: x ^ 0, x & -1, x | 0
void NopInsertion::insertDeadBitwise(Instruction *I, IRBuilder<> &Builder) {
  Value *V = nullptr;
  for (unsigned i = 0; i < I->getNumOperands(); ++i) {
    if (I->getOperand(i)->getType()->isIntegerTy()) {
      V = I->getOperand(i);
      break;
    }
  }

  if (!V) return;

  Type *Ty = V->getType();

  Value *Zero = ConstantInt::get(Ty, 0);
  Value *AllOnes = ConstantInt::getAllOnesValue(Ty);

  // Create dead bitwise chain: ((V ^ 0) & -1) | 0
  Value *Xor = Builder.CreateXor(V, Zero, "nop.xor");
  Value *And = Builder.CreateAnd(Xor, AllOnes, "nop.and");
  Value *Or = Builder.CreateOr(And, Zero, "nop.or");

  (void)Or;

  ++NumDeadBitwise;
}

// Insert dead copy operations
void NopInsertion::insertDeadCopy(Instruction *I, IRBuilder<> &Builder) {
  Value *V = nullptr;
  for (unsigned i = 0; i < I->getNumOperands(); ++i) {
    if (I->getOperand(i)->getType()->isIntegerTy()) {
      V = I->getOperand(i);
      break;
    }
  }

  if (!V) return;

  Type *Ty = V->getType();

  // Create useless transformations that result in the original value
  // V -> (V << 0) -> (V >> 0) -> V
  Value *Zero = ConstantInt::get(Ty, 0);
  Value *Shl = Builder.CreateShl(V, Zero, "nop.shl");
  Value *Lshr = Builder.CreateLShr(Shl, Zero, "nop.lshr");

  (void)Lshr;

  ++NumDeadArithmetic;
}
