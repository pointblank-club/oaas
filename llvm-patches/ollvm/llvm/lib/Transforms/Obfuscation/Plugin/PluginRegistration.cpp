//===- PluginRegistration.cpp - Register obfuscation passes as plugin ----===//
//
// Unified registration for all OLLVM obfuscation passes
//
//===----------------------------------------------------------------------===//

#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Function.h"
#include "llvm/Pass.h"
#include "llvm/Transforms/Obfuscation/Flattening.h"
#include "llvm/Transforms/Obfuscation/Split.h"
#include "llvm/Transforms/Obfuscation/Substitution.h"
#include "llvm/Transforms/Obfuscation/LinearMBA.h"
#include "llvm/Transforms/Obfuscation/ConstantObfuscation.h"
#include "llvm/Transforms/Obfuscation/NopInsertion.h"
#include "llvm/Transforms/Obfuscation/BlockReordering.h"
#include "llvm/Transforms/Obfuscation/IndirectCall.h"
#include "llvm/Transforms/Obfuscation/IndirectBranch.h"

// Forward declare to avoid including heavy BogusControlFlow.h
namespace llvm {
  Pass *createBogus(bool flag);
}

using namespace llvm;

// Wrapper passes for new pass manager
namespace {

struct FlatteningPassWrapper : public PassInfoMixin<FlatteningPassWrapper> {
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM) {
    Pass *LegacyPass = createFlattening(true);
    FunctionPass *FP = static_cast<FunctionPass*>(LegacyPass);
    bool Changed = FP->runOnFunction(F);
    delete LegacyPass;
    return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
  }
  static bool isRequired() { return true; }
};

struct SubstitutionPassWrapper : public PassInfoMixin<SubstitutionPassWrapper> {
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM) {
    Pass *LegacyPass = createSubstitution(true);
    FunctionPass *FP = static_cast<FunctionPass*>(LegacyPass);
    bool Changed = FP->runOnFunction(F);
    delete LegacyPass;
    return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
  }
  static bool isRequired() { return true; }
};

struct SplitBasicBlockPassWrapper : public PassInfoMixin<SplitBasicBlockPassWrapper> {
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM) {
    Pass *LegacyPass = createSplitBasicBlock(true);
    FunctionPass *FP = static_cast<FunctionPass*>(LegacyPass);
    bool Changed = FP->runOnFunction(F);
    delete LegacyPass;
    return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
  }
  static bool isRequired() { return true; }
};

struct BogusControlFlowPassWrapper : public PassInfoMixin<BogusControlFlowPassWrapper> {
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM) {
    Pass *LegacyPass = createBogus(true);
    FunctionPass *FP = static_cast<FunctionPass*>(LegacyPass);
    bool Changed = FP->runOnFunction(F);
    delete LegacyPass;
    return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
  }
  static bool isRequired() { return true; }
};

struct ConstantObfuscationPassWrapper : public PassInfoMixin<ConstantObfuscationPassWrapper> {
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM) {
    Pass *LegacyPass = createConstantObfuscation(true);
    FunctionPass *FP = static_cast<FunctionPass*>(LegacyPass);
    bool Changed = FP->runOnFunction(F);
    delete LegacyPass;
    return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
  }
  static bool isRequired() { return true; }
};

struct NopInsertionPassWrapper : public PassInfoMixin<NopInsertionPassWrapper> {
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM) {
    Pass *LegacyPass = createNopInsertion(true);
    FunctionPass *FP = static_cast<FunctionPass*>(LegacyPass);
    bool Changed = FP->runOnFunction(F);
    delete LegacyPass;
    return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
  }
  static bool isRequired() { return true; }
};

struct BlockReorderingPassWrapper : public PassInfoMixin<BlockReorderingPassWrapper> {
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM) {
    Pass *LegacyPass = createBlockReordering(true);
    FunctionPass *FP = static_cast<FunctionPass*>(LegacyPass);
    bool Changed = FP->runOnFunction(F);
    delete LegacyPass;
    return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
  }
  static bool isRequired() { return true; }
};

struct IndirectCallPassWrapper : public PassInfoMixin<IndirectCallPassWrapper> {
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM) {
    Pass *LegacyPass = createIndirectCall(true);
    FunctionPass *FP = static_cast<FunctionPass*>(LegacyPass);
    bool Changed = FP->runOnFunction(F);
    delete LegacyPass;
    return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
  }
  static bool isRequired() { return true; }
};

struct IndirectBranchPassWrapper : public PassInfoMixin<IndirectBranchPassWrapper> {
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM) {
    Pass *LegacyPass = createIndirectBranch(true);
    FunctionPass *FP = static_cast<FunctionPass*>(LegacyPass);
    bool Changed = FP->runOnFunction(F);
    delete LegacyPass;
    return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
  }
  static bool isRequired() { return true; }
};

} // end anonymous namespace

// Plugin registration for opt
extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
  return {
    LLVM_PLUGIN_API_VERSION, "ObfuscationPasses", LLVM_VERSION_STRING,
    [](PassBuilder &PB) {
      PB.registerPipelineParsingCallback(
        [](StringRef Name, FunctionPassManager &FPM,
           ArrayRef<PassBuilder::PipelineElement>) {
          if (Name == "flattening") {
            FPM.addPass(FlatteningPassWrapper());
            return true;
          }
          if (Name == "substitution") {
            FPM.addPass(SubstitutionPassWrapper());
            return true;
          }
          if (Name == "split") {
            FPM.addPass(SplitBasicBlockPassWrapper());
            return true;
          }
          if (Name == "boguscf") {
            FPM.addPass(BogusControlFlowPassWrapper());
            return true;
          }
          if (Name == "linear-mba") {
            FPM.addPass(LinearMBAPass(1, 0xC0FFEE));
            return true;
          }
          if (Name == "constant-obfuscate") {
            FPM.addPass(ConstantObfuscationPassWrapper());
            return true;
          }
          if (Name == "nop-insertion") {
            FPM.addPass(NopInsertionPassWrapper());
            return true;
          }
          if (Name == "block-reordering") {
            FPM.addPass(BlockReorderingPassWrapper());
            return true;
          }
          if (Name == "indirect-call") {
            FPM.addPass(IndirectCallPassWrapper());
            return true;
          }
          if (Name == "indirect-branch") {
            FPM.addPass(IndirectBranchPassWrapper());
            return true;
          }
          return false;
        });
    }
  };
}
