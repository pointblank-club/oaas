//===- Flattening.cpp - Flattening Obfuscation pass------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the flattening pass
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Obfuscation/Flattening.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/IR/Constants.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/LowerSwitch.h"
#include "llvm/CryptoUtils.h"
#include <cstring>

// Forward declarations (ProcessSwitchInst is in LowerSwitch.h)
namespace llvm {
  class LazyValueInfo;
  class AssumptionCache;
}

#define DEBUG_TYPE "flattening"

using namespace llvm;

// Stats
STATISTIC(Flattened, "Functions flattened");

// Binary-safe mode for McSema-lifted IR compatibility
static cl::opt<bool>
BinarySafeMode("fla_binary_safe",
    cl::desc("Enable binary-safe mode for McSema-lifted IR (skips sub_* functions, reduces complexity)"),
    cl::init(false), cl::Optional);

// Maximum number of basic blocks to flatten in binary-safe mode
static cl::opt<int>
MaxBBsInBinarySafe("fla_max_bbs",
    cl::desc("Maximum basic blocks to flatten per function in binary-safe mode"),
    cl::value_desc("max blocks"), cl::init(50), cl::Optional);

namespace {
struct Flattening : public FunctionPass {
  static char ID;  // Pass identification, replacement for typeid
  bool flag;

  Flattening() : FunctionPass(ID) {}
  Flattening(bool flag) : FunctionPass(ID) { this->flag = flag; }

  // Helper: Check if this is a McSema-generated function (sub_*)
  bool isMcSemaFunction(Function *F) {
    StringRef Name = F->getName();
    // McSema generates functions like sub_140001000, sub_*, callback_*, etc.
    return Name.starts_with("sub_") ||
           Name.starts_with("callback_") ||
           Name.starts_with("data_") ||
           Name.starts_with("ext_") ||
           Name.starts_with("__mcsema") ||
           Name.starts_with("__remill");
  }

  bool runOnFunction(Function &F) override;
  bool flatten(Function *f);
};
}

char Flattening::ID = 0;
static RegisterPass<Flattening> X("flattening", "Call graph flattening");
Pass *llvm::createFlattening(bool flag) { return new Flattening(flag); }

bool Flattening::runOnFunction(Function &F) {
  Function *tmp = &F;

  // Binary-safe mode: Skip McSema-generated functions entirely
  // These functions are lifted from binary and modifying their CFG
  // can corrupt the state machine architecture
  if (BinarySafeMode && isMcSemaFunction(tmp)) {
    DEBUG_WITH_TYPE("opt", errs() << "fla: Skipping McSema function in binary-safe mode: "
        << F.getName() << "\n");
    return false;
  }

  // Binary-safe mode: Skip functions with too many basic blocks
  // Large functions are more likely to have complex state machine logic
  if (BinarySafeMode) {
    size_t bbCount = 0;
    for (auto &BB : F) {
      (void)BB;
      bbCount++;
    }
    if (bbCount > (size_t)MaxBBsInBinarySafe) {
      DEBUG_WITH_TYPE("opt", errs() << "fla: Skipping large function in binary-safe mode: "
          << F.getName() << " (BBs: " << bbCount << ")\n");
      return false;
    }
  }

  // Do we obfuscate
  if (toObfuscate(flag, tmp, "fla")) {
    if (flatten(tmp)) {
      ++Flattened;
    }
  }

  return false;
}

bool Flattening::flatten(Function *f) {
  errs() << "DEBUG: flatten() called for function: " << f->getName() << "\n";
  vector<BasicBlock *> origBB;
  BasicBlock *loopEntry;
  BasicBlock *loopEnd;
  LoadInst *load;
  SwitchInst *switchI;
  AllocaInst *switchVar;

  errs() << "DEBUG: About to initialize cryptoutils scrambler\n";
  // SCRAMBLER
  char scrambling_key[16];
  bool use_scrambler = false;
  // Force initialization of cryptoutils if not already done
  if (cryptoutils.isConstructed()) {
    errs() << "DEBUG: cryptoutils is constructed, getting scrambling key\n";
    llvm::cryptoutils->get_bytes(scrambling_key, 16);
    use_scrambler = true;
  } else {
    errs() << "DEBUG: cryptoutils not constructed, using passthrough\n";
    // Fallback: use zeros if cryptoutils not available
    memset(scrambling_key, 0, 16);
  }

  // Helper lambda to scramble values or pass through
  auto scramble = [&](uint32_t val) -> uint32_t {
    return use_scrambler ? llvm::cryptoutils->scramble32(val, scrambling_key) : val;
  };
  // END OF SCRAMBLER

  errs() << "DEBUG: About to lower switch instructions\n";
  // Lower switch - using modern LLVM utility
  SmallPtrSet<BasicBlock *, 16> DeleteList;
  for (BasicBlock &BB : *f) {
    errs() << "DEBUG: Checking BB: " << BB.getName() << "\n";
    if (SwitchInst *SI = dyn_cast<SwitchInst>(BB.getTerminator())) {
      errs() << "DEBUG: Found switch instruction, processing...\n";
      ProcessSwitchInst(SI, DeleteList, nullptr, nullptr);
    }
  }
  errs() << "DEBUG: Deleting " << DeleteList.size() << " basic blocks\n";
  for (BasicBlock *BB : DeleteList) {
    BB->eraseFromParent();
  }

  errs() << "DEBUG: About to save all original BB\n";
  // Save all original BB
  for (Function::iterator i = f->begin(); i != f->end(); ++i) {
    BasicBlock *tmp = &*i;
    origBB.push_back(tmp);

    BasicBlock *bb = &*i;
    errs() << "DEBUG: Checking BB terminator for: " << bb->getName() << "\n";
    if (!bb->getTerminator()) {
      errs() << "DEBUG: BB has no terminator, skipping\n";
      return false;
    }
    if (isa<InvokeInst>(bb->getTerminator())) {
      errs() << "DEBUG: BB has invoke instruction, bailing out\n";
      return false;
    }
  }

  // Nothing to flatten
  errs() << "DEBUG: origBB.size() = " << origBB.size() << "\n";
  if (origBB.size() <= 1) {
    errs() << "DEBUG: Nothing to flatten, returning\n";
    return false;
  }

  errs() << "DEBUG: Removing first BB\n";
  // Remove first BB
  origBB.erase(origBB.begin());

  errs() << "DEBUG: Getting pointer to first BB\n";
  // Get a pointer on the first BB
  Function::iterator tmp = f->begin();  //++tmp;
  BasicBlock *insert = &*tmp;

  errs() << "DEBUG: Checking if main begins with an if\n";
  // If main begin with an if
  BranchInst *br = NULL;
  if (isa<BranchInst>(insert->getTerminator())) {
    br = cast<BranchInst>(insert->getTerminator());
  }

  if ((br != NULL && br->isConditional()) ||
      insert->getTerminator()->getNumSuccessors() > 1) {
    errs() << "DEBUG: Conditional branch found, need to split basic block\n";
    errs() << "DEBUG: insert->size() = " << insert->size() << "\n";

    // The goal: If the first BB has multiple instructions before a conditional branch,
    // split it so that there's a clean entry point.
    // In LLVM 22, splitBasicBlock(iterator, name, Before) has a Before parameter.
    // If Before=false (default): splits AT iterator, moving it to new block
    // If Before=true: splits BEFORE iterator, keeping it in original block

    Instruction *terminator = insert->getTerminator();

    // We want to split BEFORE the terminator if there are multiple instructions
    if (insert->size() > 1) {
      errs() << "DEBUG: Splitting before terminator\n";
      // With Before=false (default): Split AT terminator, moves it and everything after to new block
      // With Before=true: Split BEFORE terminator, keeps it in original, moves everything before to new block
      // We want the terminator to stay so we can erase it, so use Before=false (default)
      BasicBlock *tmpBB = insert->splitBasicBlock(terminator, "first");
      origBB.insert(origBB.begin(), tmpBB);
      errs() << "DEBUG: Split complete, tmpBB has " << tmpBB->size() << " instructions\n";
      errs() << "DEBUG: insert now has " << insert->size() << " instructions\n";
    }
  }

  errs() << "DEBUG: About to remove jump from insert block\n";
  // Remove jump - insert should still have its terminator
  if (insert->getTerminator()) {
    errs() << "DEBUG: Erasing terminator from insert\n";
    insert->getTerminator()->eraseFromParent();
  } else {
    errs() << "DEBUG: ERROR: No terminator in insert block!\n";
    return false;
  }

  errs() << "DEBUG: Creating switch variable\n";
  // Create switch variable and set as it
  // NOTE: insert has no terminator now, so we need to insert at end()
  switchVar =
      new AllocaInst(Type::getInt32Ty(f->getContext()), 0, "switchVar", insert->end());
  errs() << "DEBUG: Creating store instruction\n";
  new StoreInst(
      ConstantInt::get(Type::getInt32Ty(f->getContext()),
                       scramble(0)),
      switchVar, insert->end());

  errs() << "DEBUG: Creating main loop blocks\n";
  // Create main loop
  loopEntry = BasicBlock::Create(f->getContext(), "loopEntry", f, insert);
  loopEnd = BasicBlock::Create(f->getContext(), "loopEnd", f, insert);

  errs() << "DEBUG: Creating load instruction in loopEntry\n";
  // loopEntry has no terminator yet, use end()
  load = new LoadInst(Type::getInt32Ty(f->getContext()), switchVar, "switchVar", loopEntry->end());

  errs() << "DEBUG: Moving insert block and creating branch\n";
  // Move first BB on top
  insert->moveBefore(loopEntry);
  // insert still has no terminator, use end()
  BranchInst::Create(loopEntry, insert->end());

  errs() << "DEBUG: Creating loopEnd branch\n";
  // loopEnd jump to loopEntry
  BranchInst::Create(loopEntry, loopEnd->end());

  errs() << "DEBUG: Creating switch default block\n";
  BasicBlock *swDefault =
      BasicBlock::Create(f->getContext(), "switchDefault", f, loopEnd);
  BranchInst::Create(loopEnd, swDefault->end());

  errs() << "DEBUG: Creating switch instruction\n";
  // Create switch instruction itself and set condition
  switchI = SwitchInst::Create(&*f->begin(), swDefault, 0, loopEntry->end());
  switchI->setCondition(load);

  errs() << "DEBUG: Fixing first BB branch\n";
  // Remove branch jump from 1st BB and make a jump to the while
  f->begin()->getTerminator()->eraseFromParent();

  BranchInst::Create(loopEntry, f->begin()->end());

  // Put all BB in the switch
  for (vector<BasicBlock *>::iterator b = origBB.begin(); b != origBB.end();
       ++b) {
    BasicBlock *i = *b;
    ConstantInt *numCase = NULL;

    // Move the BB inside the switch (only visual, no code logic)
    i->moveBefore(loopEnd);

    // Add case to switch
    numCase = cast<ConstantInt>(ConstantInt::get(
        switchI->getCondition()->getType(),
        scramble(switchI->getNumCases())));
    switchI->addCase(numCase, i);
  }

  // Recalculate switchVar
  for (vector<BasicBlock *>::iterator b = origBB.begin(); b != origBB.end();
       ++b) {
    BasicBlock *i = *b;
    ConstantInt *numCase = NULL;

    // Ret BB
    if (i->getTerminator()->getNumSuccessors() == 0) {
      continue;
    }

    // If it's a non-conditional jump
    if (i->getTerminator()->getNumSuccessors() == 1) {
      // Get successor and delete terminator
      BasicBlock *succ = i->getTerminator()->getSuccessor(0);
      i->getTerminator()->eraseFromParent();

      // Get next case
      numCase = switchI->findCaseDest(succ);

      // If next case == default case (switchDefault)
      if (numCase == NULL) {
        numCase = cast<ConstantInt>(
            ConstantInt::get(switchI->getCondition()->getType(),
                             scramble(switchI->getNumCases() - 1)));
      }

      // Update switchVar and jump to the end of loop
      // i has no terminator now (erased above), use end()
      new StoreInst(numCase, load->getPointerOperand(), i->end());
      BranchInst::Create(loopEnd, i->end());
      continue;
    }

    // If it's a conditional jump
    if (i->getTerminator()->getNumSuccessors() == 2) {
      // Get next cases
      ConstantInt *numCaseTrue =
          switchI->findCaseDest(i->getTerminator()->getSuccessor(0));
      ConstantInt *numCaseFalse =
          switchI->findCaseDest(i->getTerminator()->getSuccessor(1));

      // Check if next case == default case (switchDefault)
      if (numCaseTrue == NULL) {
        numCaseTrue = cast<ConstantInt>(
            ConstantInt::get(switchI->getCondition()->getType(),
                             scramble(switchI->getNumCases() - 1)));
      }

      if (numCaseFalse == NULL) {
        numCaseFalse = cast<ConstantInt>(
            ConstantInt::get(switchI->getCondition()->getType(),
                             scramble(switchI->getNumCases() - 1)));
      }

      // Create a SelectInst
      BranchInst *br = cast<BranchInst>(i->getTerminator());
      SelectInst *sel =
          SelectInst::Create(br->getCondition(), numCaseTrue, numCaseFalse, "",
                             i->getTerminator()->getIterator());

      // Erase terminator
      i->getTerminator()->eraseFromParent();

      // Update switchVar and jump to the end of loop
      // i has no terminator now (erased above), use end()
      new StoreInst(sel, load->getPointerOperand(), i->end());
      BranchInst::Create(loopEnd, i->end());
      continue;
    }
  }

  fixStack(f);

  return true;
}
