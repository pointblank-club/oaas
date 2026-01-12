//===- IndirectBranch.cpp - Indirect Branch Obfuscation Pass -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the indirect branch obfuscation pass. It converts
// direct branches to indirect branches via blockaddress.
//
// Example (unconditional):
//   Before: br label %bb2
//   After:  %target = blockaddress(@func, %bb2)
//           indirectbr i8* %target, [label %bb2]
//
// Example (conditional):
//   Before: br i1 %cond, label %true_block, label %false_block
//   After:  %true_target = blockaddress(@func, %true_block)
//           %false_target = blockaddress(@func, %false_block)
//           %target = select i1 %cond, i8* %true_target, i8* %false_target
//           indirectbr i8* %target, [label %true_block, label %false_block]
//
// MEDIUM RISK FOR BINARY OBFUSCATION MODE (McSema lifted IR):
// - IR remains structurally correct
// - Must not apply to exception handling blocks
// - Backend support varies (especially Mingw-lld)
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Obfuscation/IndirectBranch.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Obfuscation/Utils.h"
#include <vector>

#define DEBUG_TYPE "indirect-branch"

// Command line options
static cl::opt<int> IndirectBranchRate(
    "ibr_rate",
    cl::desc("Probability [%] of converting a direct branch to indirect"),
    cl::value_desc("rate"), cl::init(50), cl::Optional);

// Binary-safe mode
static cl::opt<bool> BinarySafeMode(
    "ibr_binary_safe",
    cl::desc("Enable binary-safe mode for McSema-lifted IR"),
    cl::init(false), cl::Optional);

// Maximum number of indirect branches per function (to avoid linker issues)
static cl::opt<int> MaxIndirectBranches(
    "ibr_max",
    cl::desc("Maximum indirect branches per function"),
    cl::value_desc("max"), cl::init(30), cl::Optional);

// Stats
STATISTIC(NumIndirectBranches, "Number of branches converted to indirect");
STATISTIC(NumSkippedEH, "Number of EH-related branches skipped");

namespace {

struct IndirectBranch : public FunctionPass {
  static char ID;
  bool flag;

  IndirectBranch() : FunctionPass(ID), flag(false) {}
  IndirectBranch(bool flag) : FunctionPass(ID), flag(flag) {}

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

  // Helper: Check if block is involved in exception handling
  bool isExceptionBlock(BasicBlock *BB) {
    if (BB->isLandingPad()) return true;
    for (Instruction &I : *BB) {
      if (isa<LandingPadInst>(&I) || isa<ResumeInst>(&I) ||
          isa<CatchPadInst>(&I) || isa<CleanupPadInst>(&I) ||
          isa<CatchReturnInst>(&I) || isa<CleanupReturnInst>(&I) ||
          isa<CatchSwitchInst>(&I)) {
        return true;
      }
    }
    return false;
  }

  // Helper: Check if any successor is an exception block
  bool hasExceptionSuccessor(BranchInst *BI) {
    for (unsigned i = 0; i < BI->getNumSuccessors(); ++i) {
      if (isExceptionBlock(BI->getSuccessor(i))) {
        return true;
      }
    }
    return false;
  }

  bool runOnFunction(Function &F) override;
  bool convertToIndirect(Function *F);
};

} // end anonymous namespace

char IndirectBranch::ID = 0;
static RegisterPass<IndirectBranch> X("indirect-branch", "Indirect Branch Obfuscation Pass");

Pass *llvm::createIndirectBranch() {
  return new IndirectBranch(false);
}

Pass *llvm::createIndirectBranch(bool flag) {
  return new IndirectBranch(flag);
}

bool IndirectBranch::runOnFunction(Function &F) {
  // Binary-safe mode: Skip McSema-generated functions entirely
  if (BinarySafeMode && isMcSemaFunction(&F)) {
    return false;
  }

  Function *Func = &F;

  if (toObfuscate(flag, Func, "ibr")) {
    return convertToIndirect(Func);
  }

  return false;
}

bool IndirectBranch::convertToIndirect(Function *F) {
  bool Changed = false;
  int ConvertedCount = 0;

  // Collect branch instructions to convert
  std::vector<BranchInst*> BranchesToConvert;

  for (BasicBlock &BB : *F) {
    // Skip exception handling blocks
    if (isExceptionBlock(&BB)) {
      ++NumSkippedEH;
      continue;
    }

    Instruction *Term = BB.getTerminator();
    if (!Term) continue;

    // Only handle BranchInst (not switch, invoke, etc.)
    BranchInst *BI = dyn_cast<BranchInst>(Term);
    if (!BI) continue;

    // Skip if any successor is an exception block
    if (hasExceptionSuccessor(BI)) {
      ++NumSkippedEH;
      continue;
    }

    // Randomly decide whether to convert
    if ((int)llvm::cryptoutils->get_range(100) < IndirectBranchRate) {
      BranchesToConvert.push_back(BI);
    }
  }

  // Convert collected branches (up to max limit)
  for (BranchInst *BI : BranchesToConvert) {
    if (ConvertedCount >= MaxIndirectBranches) break;

    LLVMContext &Ctx = F->getContext();
    Type *I8PtrTy = Type::getInt8PtrTy(Ctx);

    if (BI->isUnconditional()) {
      // Unconditional branch: br label %dest
      BasicBlock *Dest = BI->getSuccessor(0);

      // Create blockaddress
      BlockAddress *BA = BlockAddress::get(F, Dest);
      Value *Target = ConstantExpr::getBitCast(BA, I8PtrTy);

      // Create indirect branch
      IndirectBrInst *IBr = IndirectBrInst::Create(Target, 1, BI->getIterator());
      IBr->addDestination(Dest);

      // Remove old branch
      BI->eraseFromParent();

    } else {
      // Conditional branch: br i1 %cond, label %true, label %false
      Value *Cond = BI->getCondition();
      BasicBlock *TrueDest = BI->getSuccessor(0);
      BasicBlock *FalseDest = BI->getSuccessor(1);

      // Create blockaddresses
      BlockAddress *TrueBA = BlockAddress::get(F, TrueDest);
      BlockAddress *FalseBA = BlockAddress::get(F, FalseDest);

      Value *TrueTarget = ConstantExpr::getBitCast(TrueBA, I8PtrTy);
      Value *FalseTarget = ConstantExpr::getBitCast(FalseBA, I8PtrTy);

      // Create select to choose target
      IRBuilder<> Builder(BI);
      Value *Target = Builder.CreateSelect(Cond, TrueTarget, FalseTarget, "ibr.target");

      // Create indirect branch with both destinations
      IndirectBrInst *IBr = IndirectBrInst::Create(Target, 2, BI->getIterator());
      IBr->addDestination(TrueDest);
      IBr->addDestination(FalseDest);

      // Remove old branch
      BI->eraseFromParent();
    }

    ++NumIndirectBranches;
    ++ConvertedCount;
    Changed = true;
  }

  return Changed;
}
