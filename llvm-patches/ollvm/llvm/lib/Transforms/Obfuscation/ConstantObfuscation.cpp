//===- ConstantObfuscation.cpp - Constant Obfuscation Pass ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the constant obfuscation pass. It encodes literal
// constants into computed forms using various encoding strategies:
// - Linear encoding: value = (a * b) + c
// - XOR encoding: value = encrypted ^ key
// - Polynomial encoding: value = a*x^2 + b*x + c
//
// SAFE FOR BINARY OBFUSCATION MODE (McSema lifted IR):
// - Does NOT modify control flow
// - Only transforms constant values into computed sequences
// - CFG-neutral, works purely on SSA values
// - McSema state machine dispatch completely unaffected
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Obfuscation/ConstantObfuscation.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Obfuscation/Utils.h"
#include "llvm/IR/Intrinsics.h"

#define DEBUG_TYPE "constant-obfuscation"

// Command line options
static cl::opt<int> ObfConstLoop(
    "const_loop",
    cl::desc("Number of times the constant obfuscation pass loops on a function"),
    cl::value_desc("number of times"), cl::init(1), cl::Optional);

static cl::opt<int> ObfConstThreshold(
    "const_threshold",
    cl::desc("Only obfuscate constants with absolute value >= threshold"),
    cl::value_desc("threshold value"), cl::init(2), cl::Optional);

// Stats
STATISTIC(NumObfuscatedConstants, "Number of constants obfuscated");
STATISTIC(NumLinearEncodings, "Number of linear encodings applied");
STATISTIC(NumXorEncodings, "Number of XOR encodings applied");
STATISTIC(NumPolyEncodings, "Number of polynomial encodings applied");

namespace {

struct ConstantObfuscation : public FunctionPass {
  static char ID;
  bool flag;

  // Encoding methods
  enum EncodingMethod {
    LINEAR_ENCODING = 0,    // value = (a * b) + c
    XOR_ENCODING = 1,       // value = encrypted ^ key
    POLY_ENCODING = 2,      // value = a*x^2 + b*x + c
    ADDITIVE_CHAIN = 3,     // value = a + b - c + d
    NUM_ENCODINGS = 4
  };

  ConstantObfuscation() : FunctionPass(ID), flag(false) {}
  ConstantObfuscation(bool flag) : FunctionPass(ID), flag(flag) {}

  bool runOnFunction(Function &F) override;
  bool obfuscateConstants(Function *F);

  // Encoding implementations
  Value *linearEncode(ConstantInt *CI, Instruction *InsertPt);
  Value *xorEncode(ConstantInt *CI, Instruction *InsertPt);
  Value *polyEncode(ConstantInt *CI, Instruction *InsertPt);
  Value *additiveChainEncode(ConstantInt *CI, Instruction *InsertPt);

  // Helper to check if constant should be obfuscated
  bool shouldObfuscate(ConstantInt *CI);
};

} // end anonymous namespace

char ConstantObfuscation::ID = 0;
static RegisterPass<ConstantObfuscation> X("constant-obfuscate",
                                            "Constant Obfuscation Pass");

Pass *llvm::createConstantObfuscation() {
  return new ConstantObfuscation(false);
}

Pass *llvm::createConstantObfuscation(bool flag) {
  return new ConstantObfuscation(flag);
}

bool ConstantObfuscation::shouldObfuscate(ConstantInt *CI) {
  if (!CI) return false;

  // Don't obfuscate boolean values (0 and 1)
  if (CI->getBitWidth() == 1) return false;

  // Get the absolute value
  int64_t val = CI->getSExtValue();
  int64_t absVal = val < 0 ? -val : val;

  // Skip small constants based on threshold
  if (absVal < ObfConstThreshold) return false;

  return true;
}

bool ConstantObfuscation::runOnFunction(Function &F) {
  if (ObfConstLoop <= 0) {
    errs() << "Constant obfuscation loop count must be > 0\n";
    return false;
  }

  Function *Func = &F;

  // Check if we should obfuscate this function
  if (toObfuscate(flag, Func, "constobf")) {
    return obfuscateConstants(Func);
  }

  return false;
}

bool ConstantObfuscation::obfuscateConstants(Function *F) {
  bool Changed = false;

  int times = ObfConstLoop;
  do {
    // Collect instructions with constant operands to avoid iterator invalidation
    std::vector<std::pair<Instruction*, unsigned>> ToObfuscate;

    for (BasicBlock &BB : *F) {
      for (Instruction &I : BB) {
        // Skip PHI nodes - they need special handling
        if (isa<PHINode>(&I)) continue;

        // Skip terminators - don't mess with branch conditions
        if (I.isTerminator()) continue;

        // Check each operand
        for (unsigned i = 0; i < I.getNumOperands(); ++i) {
          if (ConstantInt *CI = dyn_cast<ConstantInt>(I.getOperand(i))) {
            if (shouldObfuscate(CI)) {
              ToObfuscate.push_back({&I, i});
            }
          }
        }
      }
    }

    // Now apply obfuscation
    for (auto &Pair : ToObfuscate) {
      Instruction *I = Pair.first;
      unsigned OpIdx = Pair.second;
      ConstantInt *CI = dyn_cast<ConstantInt>(I->getOperand(OpIdx));

      if (!CI || !shouldObfuscate(CI)) continue;

      // Select random encoding method
      EncodingMethod method = static_cast<EncodingMethod>(
          llvm::cryptoutils->get_range(NUM_ENCODINGS));

      Value *NewVal = nullptr;

      switch (method) {
        case LINEAR_ENCODING:
          NewVal = linearEncode(CI, I);
          ++NumLinearEncodings;
          break;
        case XOR_ENCODING:
          NewVal = xorEncode(CI, I);
          ++NumXorEncodings;
          break;
        case POLY_ENCODING:
          NewVal = polyEncode(CI, I);
          ++NumPolyEncodings;
          break;
        case ADDITIVE_CHAIN:
          NewVal = additiveChainEncode(CI, I);
          ++NumLinearEncodings;
          break;
        default:
          NewVal = linearEncode(CI, I);
          ++NumLinearEncodings;
          break;
      }

      if (NewVal) {
        I->setOperand(OpIdx, NewVal);
        ++NumObfuscatedConstants;
        Changed = true;
      }
    }
  } while (--times > 0);

  return Changed;
}

// Linear encoding: value = (a * b) + c
// where a and b are random, c is computed to make equation hold
Value *ConstantObfuscation::linearEncode(ConstantInt *CI, Instruction *InsertPt) {
  Type *Ty = CI->getType();
  int64_t value = CI->getSExtValue();

  // Generate random multipliers
  // Use small random values to avoid overflow issues
  int64_t a = (llvm::cryptoutils->get_range(250) + 5);  // 5-255
  int64_t b = (llvm::cryptoutils->get_range(250) + 5);  // 5-255

  // Compute c such that value = (a * b) + c
  int64_t product = a * b;
  int64_t c = value - product;

  IRBuilder<> Builder(InsertPt);

  // Create: (a * b) + c
  Value *A = ConstantInt::get(Ty, a);
  Value *B = ConstantInt::get(Ty, b);
  Value *C = ConstantInt::get(Ty, c);

  Value *Mul = Builder.CreateMul(A, B, "const.mul");
  Value *Result = Builder.CreateAdd(Mul, C, "const.add");

  return Result;
}

// XOR encoding: value = encrypted ^ key
// where key is random, encrypted = value ^ key
Value *ConstantObfuscation::xorEncode(ConstantInt *CI, Instruction *InsertPt) {
  Type *Ty = CI->getType();
  uint64_t value = CI->getZExtValue();

  // Generate random key
  uint64_t key = llvm::cryptoutils->get_uint64_t();

  // Compute encrypted value
  uint64_t encrypted = value ^ key;

  IRBuilder<> Builder(InsertPt);

  // Create: encrypted ^ key
  Value *Encrypted = ConstantInt::get(Ty, encrypted);
  Value *Key = ConstantInt::get(Ty, key);

  Value *Result = Builder.CreateXor(Encrypted, Key, "const.xor");

  return Result;
}

// Polynomial encoding: value = a*x^2 + b*x + c
// We use x=2 for simplicity: value = 4a + 2b + c
Value *ConstantObfuscation::polyEncode(ConstantInt *CI, Instruction *InsertPt) {
  Type *Ty = CI->getType();
  int64_t value = CI->getSExtValue();

  // Use x = 2, so value = 4a + 2b + c
  // Generate random a and b
  int64_t a = (int64_t)(llvm::cryptoutils->get_range(100)) - 50;  // -50 to 49
  int64_t b = (int64_t)(llvm::cryptoutils->get_range(100)) - 50;  // -50 to 49

  // Compute c such that value = 4a + 2b + c
  int64_t c = value - (4 * a) - (2 * b);

  IRBuilder<> Builder(InsertPt);

  // Create x = 2
  Value *X = ConstantInt::get(Ty, 2);
  Value *A = ConstantInt::get(Ty, a);
  Value *B = ConstantInt::get(Ty, b);
  Value *C = ConstantInt::get(Ty, c);

  // x^2
  Value *X2 = Builder.CreateMul(X, X, "const.x2");

  // a * x^2
  Value *AX2 = Builder.CreateMul(A, X2, "const.ax2");

  // b * x
  Value *BX = Builder.CreateMul(B, X, "const.bx");

  // a*x^2 + b*x
  Value *Partial = Builder.CreateAdd(AX2, BX, "const.partial");

  // a*x^2 + b*x + c
  Value *Result = Builder.CreateAdd(Partial, C, "const.poly");

  return Result;
}

// Additive chain encoding: value = a + b - c + d
Value *ConstantObfuscation::additiveChainEncode(ConstantInt *CI, Instruction *InsertPt) {
  Type *Ty = CI->getType();
  int64_t value = CI->getSExtValue();

  // Generate random components
  int64_t a = (int64_t)llvm::cryptoutils->get_uint32_t();
  int64_t b = (int64_t)llvm::cryptoutils->get_uint32_t();
  int64_t c = (int64_t)llvm::cryptoutils->get_uint32_t();

  // Compute d such that value = a + b - c + d
  int64_t d = value - a - b + c;

  IRBuilder<> Builder(InsertPt);

  Value *A = ConstantInt::get(Ty, a);
  Value *B = ConstantInt::get(Ty, b);
  Value *C = ConstantInt::get(Ty, c);
  Value *D = ConstantInt::get(Ty, d);

  // a + b
  Value *AB = Builder.CreateAdd(A, B, "const.ab");

  // (a + b) - c
  Value *ABC = Builder.CreateSub(AB, C, "const.abc");

  // (a + b - c) + d
  Value *Result = Builder.CreateAdd(ABC, D, "const.abcd");

  return Result;
}
