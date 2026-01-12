//===- BlockReordering.cpp - Basic Block Reordering Pass -----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the basic block reordering pass. It rearranges the
// order of basic blocks without changing control flow edges.
//
// SAFE FOR BINARY OBFUSCATION MODE (McSema lifted IR):
// - Does NOT modify control flow edges (labels still point to correct blocks)
// - STATE struct + PC update logic unaffected
// - No PHI node corruption
// - Disassemblers get confused (good obfuscation)
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Obfuscation/BlockReordering.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Obfuscation/Utils.h"
#include <algorithm>
#include <vector>

#define DEBUG_TYPE "block-reordering"

// Binary-safe mode
static cl::opt<bool> BinarySafeMode(
    "reorder_binary_safe",
    cl::desc("Enable binary-safe mode for McSema-lifted IR"),
    cl::init(false), cl::Optional);

// Stats
STATISTIC(NumFunctionsReordered, "Number of functions with reordered blocks");
STATISTIC(NumBlocksMoved, "Number of basic blocks moved");

namespace {

struct BlockReordering : public FunctionPass {
  static char ID;
  bool flag;

  BlockReordering() : FunctionPass(ID), flag(false) {}
  BlockReordering(bool flag) : FunctionPass(ID), flag(flag) {}

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
    if (BB->getTerminator() && isa<InvokeInst>(BB->getTerminator())) return true;
    for (Instruction &I : *BB) {
      if (isa<LandingPadInst>(&I) || isa<ResumeInst>(&I) ||
          isa<CatchPadInst>(&I) || isa<CleanupPadInst>(&I) ||
          isa<CatchReturnInst>(&I) || isa<CleanupReturnInst>(&I)) {
        return true;
      }
    }
    return false;
  }

  bool runOnFunction(Function &F) override;
  bool reorderBlocks(Function *F);
  void shuffle(std::vector<BasicBlock*> &vec);
};

} // end anonymous namespace

char BlockReordering::ID = 0;
static RegisterPass<BlockReordering> X("block-reordering", "Basic Block Reordering Pass");

Pass *llvm::createBlockReordering() {
  return new BlockReordering(false);
}

Pass *llvm::createBlockReordering(bool flag) {
  return new BlockReordering(flag);
}

bool BlockReordering::runOnFunction(Function &F) {
  // Binary-safe mode: Skip McSema-generated functions
  if (BinarySafeMode && isMcSemaFunction(&F)) {
    return false;
  }

  Function *Func = &F;

  if (toObfuscate(flag, Func, "reorder")) {
    return reorderBlocks(Func);
  }

  return false;
}

void BlockReordering::shuffle(std::vector<BasicBlock*> &vec) {
  int n = vec.size();
  for (int i = n - 1; i > 0; --i) {
    int j = llvm::cryptoutils->get_uint32_t() % (i + 1);
    std::swap(vec[i], vec[j]);
  }
}

bool BlockReordering::reorderBlocks(Function *F) {
  // Need at least 3 blocks to reorder meaningfully
  // (entry block stays first, need at least 2 others)
  if (F->size() < 3) return false;

  // Collect all basic blocks except entry
  std::vector<BasicBlock*> Blocks;
  std::vector<BasicBlock*> ExceptionBlocks;

  bool First = true;
  for (BasicBlock &BB : *F) {
    if (First) {
      First = false;
      continue; // Skip entry block
    }

    // Keep exception blocks separate - they need special handling
    if (isExceptionBlock(&BB)) {
      ExceptionBlocks.push_back(&BB);
    } else {
      Blocks.push_back(&BB);
    }
  }

  // Need at least 2 regular blocks to shuffle
  if (Blocks.size() < 2) return false;

  // Shuffle the regular blocks
  shuffle(Blocks);

  // Move blocks to new positions
  // The entry block stays first, then we insert shuffled blocks
  BasicBlock *InsertPoint = &F->front();

  for (BasicBlock *BB : Blocks) {
    BB->moveAfter(InsertPoint);
    InsertPoint = BB;
    ++NumBlocksMoved;
  }

  // Exception blocks go at the end (keeping their relative order)
  for (BasicBlock *BB : ExceptionBlocks) {
    BB->moveAfter(InsertPoint);
    InsertPoint = BB;
  }

  ++NumFunctionsReordered;
  return true;
}
