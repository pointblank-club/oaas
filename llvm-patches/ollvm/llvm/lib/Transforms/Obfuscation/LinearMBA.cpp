// LinearMBA.cpp
// Simple LLVM Pass Plugin: Linear-MBA-style replacement of bitwise ops
//
// Build: compile as part of your LLVM plugin; example CMake snippet below
//
// This pass replaces `and`, `or`, `xor` binary operators with a
// per-bit reconstruction using shifts, truncation and selects.
// The result is semantically identical but increases IR complexity
// and produces less recognizable patterns after optimization.
//
// NOTE: This implementation focuses on integer bitwidths that
// are a power of two and <= 64 bits for speed. For larger widths
// it still works but you may want to chunk by 64-bit lanes.

#include "llvm/Transforms/Obfuscation/LinearMBA.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/CommandLine.h"
#include <random>

using namespace llvm;

// Command-line option for binary-safe mode
static cl::opt<bool>
MBABinarySafeMode("mba_binary_safe",
    cl::desc("Enable binary-safe mode for McSema-lifted IR (skips sub_* functions)"),
    cl::init(false), cl::Optional);

LinearMBAPass::LinearMBAPass(unsigned Cycles, uint64_t Seed, bool BinarySafe)
    : Cycles(Cycles), Seed(Seed), BinarySafe(BinarySafe || MBABinarySafeMode) {}

// Replace a single binary op with a per-bit reconstruction
Value* LinearMBAPass::replaceBitwiseWithMBA(BinaryOperator *BO, unsigned bitWidth, IRBuilder<> &B, std::mt19937_64 &R) {
    Value *L = BO->getOperand(0);
    Value *RHS = BO->getOperand(1);

    // Ensure operands are integer and same width (caller guarantees)
    LLVMContext &Ctx = BO->getContext();
    IntegerType *IntTy = cast<IntegerType>(L->getType());

    // We'll build the new result by computing each bit as an i1 and
    // then OR-ing them back shifted to their position. This increases
    // IR complexity and mixes operations; it's slower but obfuscated.
    Value *accum = ConstantInt::get(IntTy, 0);

    for (unsigned i = 0; i < bitWidth; ++i) {
      // compute (L >> i) & 1  --> via lshr then trunc to i1
      Value *shiftL = B.CreateLShr(L, ConstantInt::get(IntTy, i));
      Value *bitL = B.CreateTrunc(shiftL, Type::getInt1Ty(Ctx));

      Value *shiftR = B.CreateLShr(RHS, ConstantInt::get(IntTy, i));
      Value *bitR = B.CreateTrunc(shiftR, Type::getInt1Ty(Ctx));

      // Mix bits via small MBA template depending on opcode
      Value *bitRes = nullptr;
      switch (BO->getOpcode()) {
        case Instruction::And: {
          // AND: bitL & bitR -> implement as select(bitL, bitR, false)
          bitRes = B.CreateSelect(bitL, bitR, ConstantInt::getFalse(Ctx));
          break;
        }
        case Instruction::Or: {
          // OR: bitL | bitR -> select(bitL, true, bitR) -> equivalent to (bitL ? 1 : bitR)
          bitRes = B.CreateSelect(bitL, ConstantInt::getTrue(Ctx), bitR);
          break;
        }
        case Instruction::Xor: {
          // XOR: (bitL != bitR). Implement as select(bitL, not bitR, bitR)
          Value *notBitR = B.CreateNot(bitR);
          bitRes = B.CreateSelect(bitL, notBitR, bitR);
          break;
        }
        default:
          llvm_unreachable("Unhandled bitwise opcode");
      }

      // inject randomized nonsense to obfuscate i1 slightly:
      // e.g., XOR with a deterministic low-entropy bit generated from seed,
      // then invert back after shifting — this creates extra ops but preserves semantics.
      if ((R() & 0xF) == 0) { // low probability extra obf
        Value *randConst = ConstantInt::get(Type::getInt1Ty(Ctx), (R() & 1));
        bitRes = B.CreateXor(bitRes, randConst);
        // immediately XOR back before embedding (no semantic change)
        bitRes = B.CreateXor(bitRes, randConst);
      }

      // Promote bitRes (i1) back to IntTy and shift left i, then OR to accum
      Value *ext = B.CreateZExt(bitRes, IntTy);
      Value *shiftBack = B.CreateShl(ext, ConstantInt::get(IntTy, i));
      accum = B.CreateOr(accum, shiftBack);
    }

    return accum;
  }

PreservedAnalyses LinearMBAPass::run(Function &F, FunctionAnalysisManager &AM) {
    // Binary-safe mode: Skip McSema-generated functions entirely
    if ((BinarySafe || MBABinarySafeMode) && isMcSemaFunction(&F)) {
      return PreservedAnalyses::all();
    }

    // Use str() to get hash for StringRef
    std::mt19937_64 rng(Seed ^ std::hash<std::string>{}(F.getName().str()));
    bool changed = false;

    SmallVector<Instruction*, 64> toReplace;

    // collect binary bitwise ops first (we mutate while iterating)
    for (Instruction &I : instructions(F)) {
      if (auto *BO = dyn_cast<BinaryOperator>(&I)) {
        if (BO->getOpcode() == Instruction::And ||
            BO->getOpcode() == Instruction::Or ||
            BO->getOpcode() == Instruction::Xor) {
          // only integer bitwise ops
          if (BO->getType()->isIntegerTy()) {
            toReplace.push_back(&I);
          }
        }
      }
    }

    // Apply replacements; allow multiple cycles to increase obf
    for (unsigned c = 0; c < std::max(1u, Cycles); ++c) {
      for (Instruction *I : toReplace) {
        if (!I->use_empty() && I->getParent()) {
          if (auto *BO = dyn_cast<BinaryOperator>(I)) {
            IRBuilder<> B(BO);
            unsigned bitWidth = cast<IntegerType>(BO->getType())->getBitWidth();
            // limit per-bit loops to avoid extreme compile time; chunk 128->64
            if (bitWidth > 128) bitWidth = 128;
            Value *newVal = replaceBitwiseWithMBA(BO, bitWidth, B, rng);
            // if type bigger than bitWidth, zero-extend accordingly
            if (bitWidth < cast<IntegerType>(BO->getType())->getBitWidth()) {
              // extend result to original width
              newVal = B.CreateZExt(newVal, BO->getType());
            }
            BO->replaceAllUsesWith(newVal);
            BO->eraseFromParent();
            changed = true;
          }
        }
      }
      // second cycle: recollect newly introduced bitwise ops optionally
      // (this simple implementation keeps the original list to avoid runaway compile-time)
    }

    if (changed)
      return PreservedAnalyses::none();
    return PreservedAnalyses::all();
}

// Note: LinearMBA pass registration is handled in Plugin/PluginRegistration.cpp
// along with other obfuscation passes (flattening, substitution, boguscf, split)

