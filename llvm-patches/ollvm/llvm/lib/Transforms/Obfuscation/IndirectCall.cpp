//===- IndirectCall.cpp - Indirect Call Obfuscation Pass -----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the indirect call obfuscation pass. It converts
// direct function calls to indirect calls via function pointers.
//
// Example:
//   Before: call void @foo()
//   After:  %fptr = bitcast @foo to void (...)*
//           call void (...) %fptr()
//
// MEDIUM RISK FOR BINARY OBFUSCATION MODE (McSema lifted IR):
// - Does NOT alter CFG topology
// - Must filter out McSema sub_* functions (state machine stubs)
// - Safe for user-level functions only
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Obfuscation/IndirectCall.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Obfuscation/Utils.h"
#include <vector>

#define DEBUG_TYPE "indirect-call"

// Command line options
static cl::opt<int> IndirectCallRate(
    "icall_rate",
    cl::desc("Probability [%] of converting a direct call to indirect"),
    cl::value_desc("rate"), cl::init(80), cl::Optional);

// Binary-safe mode - MUST skip McSema functions
static cl::opt<bool> BinarySafeMode(
    "icall_binary_safe",
    cl::desc("Enable binary-safe mode for McSema-lifted IR (skips sub_* targets)"),
    cl::init(false), cl::Optional);

// Stats
STATISTIC(NumIndirectCalls, "Number of calls converted to indirect");
STATISTIC(NumSkippedIntrinsics, "Number of intrinsic calls skipped");
STATISTIC(NumSkippedMcSema, "Number of McSema calls skipped");

namespace {

struct IndirectCall : public FunctionPass {
  static char ID;
  bool flag;

  IndirectCall() : FunctionPass(ID), flag(false) {}
  IndirectCall(bool flag) : FunctionPass(ID), flag(flag) {}

  // Helper: Check if this is a McSema-generated function
  bool isMcSemaFunction(Function *F) {
    if (!F) return false;
    StringRef Name = F->getName();
    return Name.starts_with("sub_") ||
           Name.starts_with("callback_") ||
           Name.starts_with("data_") ||
           Name.starts_with("ext_") ||
           Name.starts_with("__mcsema") ||
           Name.starts_with("__remill");
  }

  // Helper: Check if call target should be skipped
  bool shouldSkipTarget(Function *Target) {
    if (!Target) return true; // Already indirect
    if (Target->isIntrinsic()) return true;
    if (Target->isDeclaration() && Target->hasFnAttribute(Attribute::NoUnwind)) {
      // Skip certain runtime functions
      StringRef Name = Target->getName();
      if (Name.starts_with("llvm.") || Name.starts_with("__")) {
        return true;
      }
    }
    // In binary-safe mode, skip McSema functions
    if (BinarySafeMode && isMcSemaFunction(Target)) {
      ++NumSkippedMcSema;
      return true;
    }
    return false;
  }

  bool runOnFunction(Function &F) override;
  bool convertToIndirect(Function *F);
};

} // end anonymous namespace

char IndirectCall::ID = 0;
static RegisterPass<IndirectCall> X("indirect-call", "Indirect Call Obfuscation Pass");

Pass *llvm::createIndirectCall() {
  return new IndirectCall(false);
}

Pass *llvm::createIndirectCall(bool flag) {
  return new IndirectCall(flag);
}

bool IndirectCall::runOnFunction(Function &F) {
  // Binary-safe mode: Skip McSema-generated functions entirely
  if (BinarySafeMode && isMcSemaFunction(&F)) {
    return false;
  }

  Function *Func = &F;

  if (toObfuscate(flag, Func, "icall")) {
    return convertToIndirect(Func);
  }

  return false;
}

bool IndirectCall::convertToIndirect(Function *F) {
  bool Changed = false;

  // Collect call instructions to convert
  std::vector<CallInst*> CallsToConvert;

  for (BasicBlock &BB : *F) {
    // Skip exception handling blocks
    if (BB.isLandingPad()) continue;

    for (Instruction &I : BB) {
      if (CallInst *CI = dyn_cast<CallInst>(&I)) {
        // Skip invoke instructions (handled separately for EH)
        if (isa<InvokeInst>(&I)) continue;

        // Get the called function
        Function *Target = CI->getCalledFunction();

        // Skip if should not be converted
        if (shouldSkipTarget(Target)) {
          if (Target && Target->isIntrinsic()) ++NumSkippedIntrinsics;
          continue;
        }

        // Randomly decide whether to convert
        if ((int)llvm::cryptoutils->get_range(100) < IndirectCallRate) {
          CallsToConvert.push_back(CI);
        }
      }
    }
  }

  // Convert collected calls
  for (CallInst *CI : CallsToConvert) {
    Function *Target = CI->getCalledFunction();
    if (!Target) continue;

    // Create function pointer type
    FunctionType *FTy = Target->getFunctionType();
    PointerType *FPtrTy = PointerType::getUnqual(FTy);

    // Create the indirect call
    IRBuilder<> Builder(CI);

    // Cast function to pointer
    Value *FPtr = Builder.CreateBitCast(Target, FPtrTy, "icall.fptr");

    // Collect arguments
    std::vector<Value*> Args;
    for (unsigned i = 0; i < CI->arg_size(); ++i) {
      Args.push_back(CI->getArgOperand(i));
    }

    // Create indirect call
    CallInst *NewCall = Builder.CreateCall(FTy, FPtr, Args);

    // Copy attributes
    NewCall->setCallingConv(CI->getCallingConv());
    NewCall->setAttributes(CI->getAttributes());

    // Replace uses and remove old call
    if (!CI->getType()->isVoidTy()) {
      CI->replaceAllUsesWith(NewCall);
    }
    CI->eraseFromParent();

    ++NumIndirectCalls;
    Changed = true;
  }

  return Changed;
}
