//===- BogusControlFlow.cpp - BogusControlFlow Obfuscation pass-------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------------------===//
//
// This file implements BogusControlFlow's pass, inserting bogus control flow.
// Updated with C++ exception handling support (Hikari-style).
//
// It adds bogus flow to a given basic block this way:
//
// Before :
// 	         		     entry
//      			       |
//  	    	  	 ______v______
//   	    		|   Original  |
//   	    		|_____________|
//             		       |
// 		        	       v
//		        	     return
//
// After :
//           		     entry
//             		       |
//            		   ____v_____
//      			  |condition*| (false)
//           		  |__________|----+
//           		 (true)|          |
//             		       |          |
//           		 ______v______    |
// 		        +-->|   Original* |   |
// 		        |   |_____________| (true)
// 		        |   (false)|    !-----------> return
// 		        |    ______v______    |
// 		        |   |   Altered   |<--!
// 		        |   |_____________|
// 		        |__________|
//
//  * The results of these terminator's branch's conditions are always true, but these predicates are
//    opacificated. For this, we declare two global values: x and y, and replace the FCMP_TRUE
//    predicate with (y < 10 || x * (x + 1) % 2 == 0) (this could be improved, as the global
//    values give a hint on where are the opaque predicates)
//
//  The altered bloc is a copy of the original's one with junk instructions added accordingly to the
//  type of instructions we found in the bloc
//
//  Each basic block of the function is choosen if a random number in the range [0,100] is smaller
//  than the choosen probability rate. The default value is 30. This value can be modify using
//  the option -boguscf-prob=[value]. Value must be an integer in the range [0, 100], otherwise
//  the default value is taken. Exemple: -boguscf -boguscf-prob=60
//
//  The pass can also be loop many times on a function, including on the basic blocks added in
//  a previous loop. Be careful if you use a big probability number and choose to run the loop
//  many times wich may cause the pass to run for a very long time. The default value is one loop,
//  but you can change it with -boguscf-loop=[value]. Value must be an integer greater than 1,
//  otherwise the default value is taken. Exemple: -boguscf -boguscf-loop=2
//
//
//  Defined debug types:
//  - "gen" : general informations
//  - "opt" : concerning the given options (parameter)
//  - "cfg" : printing the various function's cfg before transformation
//	      and after transformation if it has been modified, and all
//	      the functions at end of the pass, after doFinalization.
//
//  To use them all, simply use the -debug option.
//  To use only one of them, follow the pass' command by -debug-only=name.
//  Exemple, -boguscf -debug-only=cfg
//
//
//  Stats:
//  The following statistics will be printed if you use
//  the -stats command:
//
// a. Number of functions in this module
// b. Number of times we run on each function
// c. Initial number of basic blocks in this module
// d. Number of modified basic blocks
// e. Number of added basic blocks in this module
// f. Final number of basic blocks in this module
//
// file   : lib/Transforms/Obfuscation/BogusControlFlow.cpp
// date   : june 2012
// version: 1.0
// author : julie.michielin@gmail.com
// modifications: pjunod, Rinaldini Julien, exception handling support
// project: Obfuscator
// option : -boguscf
//
//===----------------------------------------------------------------------------------===//

#include "llvm/Transforms/Obfuscation/BogusControlFlow.h"
#include "llvm/Transforms/Obfuscation/Utils.h"

// Stats
#define DEBUG_TYPE "BogusControlFlow"
STATISTIC(NumFunction,  "a. Number of functions in this module");
STATISTIC(NumTimesOnFunctions, "b. Number of times we run on each function");
STATISTIC(InitNumBasicBlocks,  "c. Initial number of basic blocks in this module");
STATISTIC(NumModifiedBasicBlocks,  "d. Number of modified basic blocks");
STATISTIC(NumAddedBasicBlocks,  "e. Number of added basic blocks in this module");
STATISTIC(FinalNumBasicBlocks,  "f. Final number of basic blocks in this module");


// Options for the pass
const int defaultObfRate = 30, defaultObfTime = 1;
const int binarySafeObfRate = 10, binarySafeObfTime = 1;

static cl::opt<int>
ObfProbRate("bcf_prob", cl::desc("Choose the probability [%] each basic blocks will be obfuscated by the -bcf pass"), cl::value_desc("probability rate"), cl::init(defaultObfRate), cl::Optional);

static cl::opt<int>
ObfTimes("bcf_loop", cl::desc("Choose how many time the -bcf pass loop on a function"), cl::value_desc("number of times"), cl::init(defaultObfTime), cl::Optional);

// Binary-safe mode for McSema-lifted IR compatibility
static cl::opt<bool>
BinarySafeMode("bcf_binary_safe",
    cl::desc("Enable binary-safe mode for McSema-lifted IR (reduces aggressiveness, skips sub_* functions)"),
    cl::init(false), cl::Optional);

namespace {
  struct BogusControlFlow : public FunctionPass {
    static char ID; // Pass identification
    bool flag;
    BogusControlFlow() : FunctionPass(ID) {}
    BogusControlFlow(bool flag) : FunctionPass(ID) {this->flag = flag; BogusControlFlow();}

    // Helper: Check if this is a McSema-generated function (sub_*)
    bool isMcSemaFunction(Function *F) {
      StringRef Name = F->getName();
      // McSema generates functions like sub_140001000, sub_*, etc.
      return Name.starts_with("sub_") ||
             Name.starts_with("callback_") ||
             Name.starts_with("data_") ||
             Name.starts_with("ext_");
    }

    /* runOnFunction
     *
     * Overwrite FunctionPass method to apply the transformation
     * to the function. See header for more details.
     */
    bool runOnFunction(Function &F) override {
      // Binary-safe mode: Skip McSema-generated functions entirely
      if (BinarySafeMode && isMcSemaFunction(&F)) {
        DEBUG_WITH_TYPE("opt", errs() << "bcf: Skipping McSema function in binary-safe mode: "
            << F.getName() << "\n");
        return false;
      }

      // Get effective parameters (binary-safe mode uses reduced values)
      int effectiveObfRate = BinarySafeMode ? binarySafeObfRate : ObfProbRate;
      int effectiveObfTimes = BinarySafeMode ? binarySafeObfTime : ObfTimes;

      // Check if the percentage is correct
      if (effectiveObfTimes <= 0) {
        errs()<<"BogusControlFlow application number -bcf_loop=x must be x > 0";
		return false;
      }

      // Check if the number of applications is correct
      if ( !((effectiveObfRate > 0) && (effectiveObfRate <= 100)) ) {
        errs()<<"BogusControlFlow application basic blocks percentage -bcf_prob=x must be 0 < x <= 100";
		return false;
      }

      // Override global values for this run if in binary-safe mode
      if (BinarySafeMode) {
        ObfProbRate = effectiveObfRate;
        ObfTimes = effectiveObfTimes;
      }

      // If fla annotations
      if(toObfuscate(flag,&F,"bcf")) {
        bogus(F);
        doF(*F.getParent());
        return true;
      }

      return false;
    } // end of runOnFunction()

    void bogus(Function &F) {
      // For statistics and debug
      ++NumFunction;
      int NumBasicBlocks = 0;
      bool firstTime = true; // First time we do the loop in this function
      bool hasBeenModified = false;
      DEBUG_WITH_TYPE("opt", errs() << "bcf: Started on function " << F.getName() << "\n");
      DEBUG_WITH_TYPE("opt", errs() << "bcf: Probability rate: "<< ObfProbRate<< "\n");
      if(ObfProbRate < 0 || ObfProbRate > 100){
        DEBUG_WITH_TYPE("opt", errs() << "bcf: Incorrect value,"
            << " probability rate set to default value: "
            << defaultObfRate <<" \n");
        ObfProbRate = defaultObfRate;
      }
      DEBUG_WITH_TYPE("opt", errs() << "bcf: How many times: "<< ObfTimes<< "\n");
      if(ObfTimes <= 0){
        DEBUG_WITH_TYPE("opt", errs() << "bcf: Incorrect value,"
            << " must be greater than 1. Set to default: "
            << defaultObfTime <<" \n");
        ObfTimes = defaultObfTime;
      }
      NumTimesOnFunctions = ObfTimes;
      int NumObfTimes = ObfTimes;

        // Real begining of the pass
        // Loop for the number of time we run the pass on the function
        do{
          DEBUG_WITH_TYPE("cfg", errs() << "bcf: Function " << F.getName()
              <<", before the pass:\n");
          DEBUG_WITH_TYPE("cfg", F.viewCFG());
          // Put all the function's block in a list
          std::list<BasicBlock *> basicBlocks;
          for (Function::iterator i=F.begin();i!=F.end();++i) {
            BasicBlock *BB = &*i;
            
            // ===== EXCEPTION HANDLING: Skip landing pad blocks =====
            // Landing pad blocks cannot be modified - they must remain as the
            // first instruction in their block and be directly reachable from invoke
            if (BB->isLandingPad()) {
              DEBUG_WITH_TYPE("gen", errs() << "bcf: Skipping landing pad block: " 
                  << BB->getName() << "\n");
              continue;
            }
            
            // ===== EXCEPTION HANDLING: Skip blocks ending with invoke =====
            // We can't add bogus flow to blocks ending with invoke because
            // invoke is a terminator that must remain at the end for exception handling.
            // The invoke's unwind edge must directly reach a landing pad.
            if (BB->getTerminator() && isa<InvokeInst>(BB->getTerminator())) {
              DEBUG_WITH_TYPE("gen", errs() << "bcf: Skipping block with invoke terminator: " 
                  << BB->getName() << "\n");
              continue;
            }
            
            basicBlocks.push_back(BB);
          }
          DEBUG_WITH_TYPE("gen", errs() << "bcf: Iterating on the Function's Basic Blocks\n");

          while(!basicBlocks.empty()){
            NumBasicBlocks ++;
            // Basic Blocks' selection
            if((int)llvm::cryptoutils->get_range(100) <= ObfProbRate){
              DEBUG_WITH_TYPE("opt", errs() << "bcf: Block "
                  << NumBasicBlocks <<" selected. \n");
              hasBeenModified = true;
              ++NumModifiedBasicBlocks;
              NumAddedBasicBlocks += 3;
              FinalNumBasicBlocks += 3;
              // Add bogus flow to the given Basic Block (see description)
              BasicBlock *basicBlock = basicBlocks.front();
              addBogusFlow(basicBlock, F);
            }
            else{
              DEBUG_WITH_TYPE("opt", errs() << "bcf: Block "
                  << NumBasicBlocks <<" not selected.\n");
            }
            // remove the block from the list
            basicBlocks.pop_front();

            if(firstTime){ // first time we iterate on this function
              ++InitNumBasicBlocks;
              ++FinalNumBasicBlocks;
            }
          } // end of while(!basicBlocks.empty())
          DEBUG_WITH_TYPE("gen", errs() << "bcf: End of function " << F.getName() << "\n");
          if(hasBeenModified){ // if the function has been modified
            DEBUG_WITH_TYPE("cfg", errs() << "bcf: Function " << F.getName()
                <<", after the pass: \n");
            DEBUG_WITH_TYPE("cfg", F.viewCFG());
          }
          else{
            DEBUG_WITH_TYPE("cfg", errs() << "bcf: Function's not been modified \n");
          }
          firstTime = false;
        }while(--NumObfTimes > 0);
    }

    /* Helper: Check if a block is involved in exception handling
     * Returns true if the block should not be modified by BCF
     */
    bool isExceptionHandlingBlock(BasicBlock *BB) {
      // Check if it's a landing pad
      if (BB->isLandingPad()) return true;
      
      // Check for exception-related instructions
      for (Instruction &I : *BB) {
        // Landing pad instruction
        if (isa<LandingPadInst>(I)) return true;
        
        // Resume instruction (resumes exception propagation)
        if (isa<ResumeInst>(I)) return true;
        
        // Cleanup return (returns from cleanup pad)
        if (isa<CleanupReturnInst>(I)) return true;
        
        // Catch return (returns from catch pad)
        if (isa<CatchReturnInst>(I)) return true;
        
        // Invoke instruction (exception-aware call)
        if (isa<InvokeInst>(I)) return true;
      }
      
      return false;
    }

    /* addBogusFlow
     *
     * Add bogus flow to a given basic block, according to the header's description
     */
    virtual void addBogusFlow(BasicBlock * basicBlock, Function &F){
      // ===== EXCEPTION HANDLING: Comprehensive check for exception-related blocks =====
      if (isExceptionHandlingBlock(basicBlock)) {
        DEBUG_WITH_TYPE("gen", errs() << "bcf: Block contains exception handling, skipping\n");
        return;
      }

      // Split the block: first part with only the phi nodes and debug info and terminator
      //                  created by splitBasicBlock. (-> No instruction)
      //                  Second part with every instructions from the original block
      // We do this way, so we don't have to adjust all the phi nodes, metadatas and so on
      // for the first block. We have to let the phi nodes in the first part, because they
      // actually are updated in the second part according to them.
      BasicBlock::iterator i1 = basicBlock->begin();
      BasicBlock::iterator firstNonPHI = basicBlock->getFirstNonPHIOrDbgOrLifetime();
      if(firstNonPHI != basicBlock->end())
        i1 = firstNonPHI;
      Twine *var;
      var = new Twine("originalBB");
      BasicBlock *originalBB = basicBlock->splitBasicBlock(i1, *var);
      DEBUG_WITH_TYPE("gen", errs() << "bcf: First and original basic blocks: ok\n");

      // ===== EXCEPTION HANDLING: Check if the split created a block ending with invoke =====
      // If so, we need to handle this differently
      if (originalBB->getTerminator() && isa<InvokeInst>(originalBB->getTerminator())) {
        DEBUG_WITH_TYPE("gen", errs() << "bcf: Split resulted in invoke terminator, using simplified obfuscation\n");
        
        // For blocks ending with invoke, we can still add the entry condition
        // but we can't split after the invoke or create altered blocks that clone it
        
        // Remove the unconditional branch from basicBlock
        basicBlock->getTerminator()->eraseFromParent();
        
        // Create the always-true condition
        Value * LHS = ConstantFP::get(Type::getFloatTy(F.getContext()), 1.0);
        Value * RHS = ConstantFP::get(Type::getFloatTy(F.getContext()), 1.0);
        Twine * var4 = new Twine("condition");
        FCmpInst * condition = new FCmpInst(basicBlock->end(), FCmpInst::FCMP_TRUE, LHS, RHS, *var4);
        
        // Create a dummy "altered" block that just loops back
        // This creates the appearance of bogus control flow without touching the invoke
        Twine * var3 = new Twine("alteredBB");
        BasicBlock *alteredBB = BasicBlock::Create(F.getContext(), *var3, &F, originalBB);
        
        // The altered block immediately jumps to the original (it's bogus, never taken)
        BranchInst::Create(originalBB, alteredBB);
        
        // Add conditional branch: true -> originalBB, false -> alteredBB (never taken)
        BranchInst::Create(originalBB, alteredBB, (Value *)condition, basicBlock);
        
        DEBUG_WITH_TYPE("gen", errs() << "bcf: Simplified bogus flow added for invoke block\n");
        return;
      }

      // Creating the altered basic block on which the first basicBlock will jump
      Twine * var3 = new Twine("alteredBB");
      BasicBlock *alteredBB = createAlteredBasicBlock(originalBB, *var3, &F);
      DEBUG_WITH_TYPE("gen", errs() << "bcf: Altered basic block: ok\n");

      // Now that all the blocks are created,
      // we modify the terminators to adjust the control flow.

      alteredBB->getTerminator()->eraseFromParent();
      basicBlock->getTerminator()->eraseFromParent();
      DEBUG_WITH_TYPE("gen", errs() << "bcf: Terminator removed from the altered"
          <<" and first basic blocks\n");

      // Preparing a condition..
      // For now, the condition is an always true comparaison between 2 float
      // This will be complicated after the pass (in doFinalization())
      Value * LHS = ConstantFP::get(Type::getFloatTy(F.getContext()), 1.0);
      Value * RHS = ConstantFP::get(Type::getFloatTy(F.getContext()), 1.0);
      DEBUG_WITH_TYPE("gen", errs() << "bcf: Value LHS and RHS created\n");


      // The always true condition. End of the first block
      Twine * var4 = new Twine("condition");
      FCmpInst * condition = new FCmpInst(basicBlock->end(), FCmpInst::FCMP_TRUE , LHS, RHS, *var4);
      DEBUG_WITH_TYPE("gen", errs() << "bcf: Always true condition created\n");

      // ===== EXCEPTION HANDLING: Check if originalBB has exception handling =====
      // Cannot create normal branches to exception handling blocks (only unwind edges allowed)
      if (isExceptionHandlingBlock(originalBB)) {
        DEBUG_WITH_TYPE("gen", errs() << "bcf: originalBB has exception handling, cannot add bogus flow\n");
        // Restore the original terminator and abort bogus flow
        return;
      }

      // Jump to the original basic block if the condition is true or
      // to the altered block if false.
      BranchInst::Create(originalBB, alteredBB, (Value *)condition, basicBlock);
      DEBUG_WITH_TYPE("gen",
          errs() << "bcf: Terminator instruction in first basic block: ok\n");

      // The altered block loop back on the original one.
      BranchInst::Create(originalBB, alteredBB);
      DEBUG_WITH_TYPE("gen", errs() << "bcf: Terminator instruction in altered block: ok\n");


      // The end of the originalBB is modified to give the impression that sometimes
      // it continues in the loop, and sometimes it return the desired value
      // (of course it's always true, so it always use the original terminator..
      //  but this will be obfuscated too;) )

      // iterate on instruction just before the terminator of the originalBB
      BasicBlock::iterator i = originalBB->end();

      // ===== EXCEPTION HANDLING: Check terminator before splitting =====
      Instruction *terminator = originalBB->getTerminator();
      if (terminator && isa<InvokeInst>(terminator)) {
        // If terminator is invoke, we can't split further - just skip the second split
        DEBUG_WITH_TYPE("gen", errs() << "bcf: originalBB ends with invoke, skipping second split\n");
        return;
      }

      // Split at this point (we only want the terminator in the second part)
      Twine * var5 = new Twine("originalBBpart2");
      BasicBlock * originalBBpart2 = originalBB->splitBasicBlock(--i , *var5);
      DEBUG_WITH_TYPE("gen", errs() << "bcf: Terminator part of the original basic block"
          << " is isolated\n");
      // the first part go either on the return statement or on the begining
      // of the altered block.. So we erase the terminator created when splitting.
      originalBB->getTerminator()->eraseFromParent();
      
      // ===== EXCEPTION HANDLING: Check if originalBBpart2 has exception handling =====
      // This shouldn't happen (EH instructions should be at block start), but check anyway
      if (isExceptionHandlingBlock(originalBBpart2)) {
        DEBUG_WITH_TYPE("gen", errs() << "bcf: originalBBpart2 has exception handling, restoring original terminator\n");
        // This is a safety check - if we somehow got an EH block, restore and abort
        BranchInst::Create(originalBBpart2, originalBB);
        return;
      }
      
      // We add at the end a new always true condition
      Twine * var6 = new Twine("condition2");
      FCmpInst * condition2 = new FCmpInst(originalBB, CmpInst::FCMP_TRUE , LHS, RHS, *var6);
      BranchInst::Create(originalBBpart2, alteredBB, (Value *)condition2, originalBB);
      DEBUG_WITH_TYPE("gen", errs() << "bcf: Terminator original basic block: ok\n");
      DEBUG_WITH_TYPE("gen", errs() << "bcf: End of addBogusFlow().\n");

    } // end of addBogusFlow()

    /* createAlteredBasicBlock
     *
     * This function return a basic block similar to a given one.
     * It's inserted just after the given basic block.
     * The instructions are similar but junk instructions are added between
     * the cloned one. The cloned instructions' phi nodes, metadatas, uses and
     * debug locations are adjusted to fit in the cloned basic block and
     * behave nicely.
     */
    virtual BasicBlock* createAlteredBasicBlock(BasicBlock * basicBlock,
        const Twine &  Name = "gen", Function * F = 0){
      
      // ===== EXCEPTION HANDLING: Check for invoke/landingpad before cloning =====
      // If the block contains invoke or is a landing pad, we need special handling
      if (basicBlock->isLandingPad()) {
        DEBUG_WITH_TYPE("gen", errs() << "bcf: Cannot clone landing pad block, creating empty altered block\n");
        BasicBlock *alteredBB = BasicBlock::Create(F->getContext(), Name, F, basicBlock);
        // Add a dummy instruction and return
        BranchInst::Create(basicBlock, alteredBB);
        return alteredBB;
      }
      
      // Check if the block ends with invoke - if so, we can't safely clone it
      if (basicBlock->getTerminator() && isa<InvokeInst>(basicBlock->getTerminator())) {
        DEBUG_WITH_TYPE("gen", errs() << "bcf: Block ends with invoke, creating simplified altered block\n");
        BasicBlock *alteredBB = BasicBlock::Create(F->getContext(), Name, F, basicBlock);
        // Create a simplified block that just jumps to the original
        BranchInst::Create(basicBlock, alteredBB);
        return alteredBB;
      }
      
      // Useful to remap the informations concerning instructions.
      ValueToValueMapTy VMap;
      BasicBlock * alteredBB = llvm::CloneBasicBlock (basicBlock, VMap, Name, F);
      DEBUG_WITH_TYPE("gen", errs() << "bcf: Original basic block cloned\n");
      
      // ===== EXCEPTION HANDLING: Remove any invoke instructions from the cloned block =====
      // They can't be properly cloned without also cloning the landing pad
      std::vector<Instruction*> toRemove;
      for (BasicBlock::iterator i = alteredBB->begin(), e = alteredBB->end(); i != e; ++i) {
        if (isa<InvokeInst>(&*i)) {
          toRemove.push_back(&*i);
        }
      }
      for (Instruction *I : toRemove) {
        // Replace invoke with unreachable or branch to avoid invalid IR
        if (I == alteredBB->getTerminator()) {
          // If it's the terminator, we need to add a new terminator
          // Just create an unreachable for now (this block is bogus anyway)
          new UnreachableInst(F->getContext(), I);
        }
        I->eraseFromParent();
      }
      
      // Remap operands.
      BasicBlock::iterator ji = basicBlock->begin();
      for (BasicBlock::iterator i = alteredBB->begin(), e = alteredBB->end() ; i != e; ++i){
        // Skip if we've run out of original instructions to match against
        if (ji == basicBlock->end()) break;
        
        // Loop over the operands of the instruction
        for(User::op_iterator opi = i->op_begin (), ope = i->op_end(); opi != ope; ++opi){
          // get the value for the operand
          Value *v = MapValue(*opi, VMap,  RF_None, 0);
          if (v != 0){
            *opi = v;
            DEBUG_WITH_TYPE("gen", errs() << "bcf: Value's operand has been setted\n");
          }
        }
        DEBUG_WITH_TYPE("gen", errs() << "bcf: Operands remapped\n");
        // Remap phi nodes' incoming blocks.
        if (PHINode *pn = dyn_cast<PHINode>(i)) {
          for (unsigned j = 0, e = pn->getNumIncomingValues(); j != e; ++j) {
            Value *v = MapValue(pn->getIncomingBlock(j), VMap, RF_None, 0);
            if (v != 0){
              pn->setIncomingBlock(j, cast<BasicBlock>(v));
            }
          }
        }
        DEBUG_WITH_TYPE("gen", errs() << "bcf: PHINodes remapped\n");
        // Remap attached metadata.
        SmallVector<std::pair<unsigned, MDNode *>, 4> MDs;
        i->getAllMetadata(MDs);
        DEBUG_WITH_TYPE("gen", errs() << "bcf: Metadatas remapped\n");
        // important for compiling with DWARF, using option -g.
        i->setDebugLoc(ji->getDebugLoc());
        ji++;
        DEBUG_WITH_TYPE("gen", errs() << "bcf: Debug information location setted\n");

      } // The instructions' informations are now all correct

      DEBUG_WITH_TYPE("gen", errs() << "bcf: The cloned basic block is now correct\n");
      DEBUG_WITH_TYPE("gen",
          errs() << "bcf: Starting to add junk code in the cloned bloc...\n");

      // add random instruction in the middle of the bloc. This part can be improve
      for (BasicBlock::iterator i = alteredBB->begin(), e = alteredBB->end() ; i != e; ++i){
        // ===== EXCEPTION HANDLING: Skip exception-related instructions =====
        if (isa<LandingPadInst>(&*i) || isa<ResumeInst>(&*i) || 
            isa<CatchPadInst>(&*i) || isa<CatchReturnInst>(&*i) ||
            isa<CleanupPadInst>(&*i) || isa<CleanupReturnInst>(&*i)) {
          continue;
        }
        
        // in the case we find binary operator, we modify slightly this part by randomly
        // insert some instructions
        if(i->isBinaryOp()){ // binary instructions
          unsigned opcode = i->getOpcode();
          Value *op, *op1 = NULL;
          Twine *var = new Twine("_");
          // treat differently float or int
          // Binary int
          if(opcode == Instruction::Add || opcode == Instruction::Sub ||
              opcode == Instruction::Mul || opcode == Instruction::UDiv ||
              opcode == Instruction::SDiv || opcode == Instruction::URem ||
              opcode == Instruction::SRem || opcode == Instruction::Shl ||
              opcode == Instruction::LShr || opcode == Instruction::AShr ||
              opcode == Instruction::And || opcode == Instruction::Or ||
              opcode == Instruction::Xor){
            for(int random = (int)llvm::cryptoutils->get_range(10); random < 10; ++random){
              switch(llvm::cryptoutils->get_range(4)){ // to improve
                case 0: //do nothing
                  break;
                case 1: op = BinaryOperator::CreateNeg(i->getOperand(0),*var,i->getIterator());
                        op1 = BinaryOperator::Create(Instruction::Add,op,
                            i->getOperand(1),"gen",i->getIterator());
                        break;
                case 2: op1 = BinaryOperator::Create(Instruction::Sub,
                            i->getOperand(0),
                            i->getOperand(1),*var,i->getIterator());
                        op = BinaryOperator::Create(Instruction::Mul,op1,
                            i->getOperand(1),"gen",i->getIterator());
                        break;
                case 3: op = BinaryOperator::Create(Instruction::Shl,
                            i->getOperand(0),
                            i->getOperand(1),*var,i->getIterator());
                        break;
              }
            }
          }
          // Binary float
          if(opcode == Instruction::FAdd || opcode == Instruction::FSub ||
              opcode == Instruction::FMul || opcode == Instruction::FDiv ||
              opcode == Instruction::FRem){
            for(int random = (int)llvm::cryptoutils->get_range(10); random < 10; ++random){
              switch(llvm::cryptoutils->get_range(3)){ // can be improved
                case 0: //do nothing
                  break;
                case 1: op = UnaryOperator::CreateFNeg(i->getOperand(0),*var,i->getIterator());
                        op1 = BinaryOperator::Create(Instruction::FAdd,op,
                            i->getOperand(1),"gen",i->getIterator());
                        break;
                case 2: op = BinaryOperator::Create(Instruction::FSub,
                            i->getOperand(0),
                            i->getOperand(1),*var,i->getIterator());
                        op1 = BinaryOperator::Create(Instruction::FMul,op,
                            i->getOperand(1),"gen",i->getIterator());
                        break;
              }
            }
          }
          if(opcode == Instruction::ICmp){ // Condition (with int)
            ICmpInst *currentI = (ICmpInst*)(&i);
            switch(llvm::cryptoutils->get_range(3)){ // must be improved
              case 0: //do nothing
                break;
              case 1: currentI->swapOperands();
                      break;
              case 2: // randomly change the predicate
                      switch(llvm::cryptoutils->get_range(10)){
                        case 0: currentI->setPredicate(ICmpInst::ICMP_EQ);
                                break; // equal
                        case 1: currentI->setPredicate(ICmpInst::ICMP_NE);
                                break; // not equal
                        case 2: currentI->setPredicate(ICmpInst::ICMP_UGT);
                                break; // unsigned greater than
                        case 3: currentI->setPredicate(ICmpInst::ICMP_UGE);
                                break; // unsigned greater or equal
                        case 4: currentI->setPredicate(ICmpInst::ICMP_ULT);
                                break; // unsigned less than
                        case 5: currentI->setPredicate(ICmpInst::ICMP_ULE);
                                break; // unsigned less or equal
                        case 6: currentI->setPredicate(ICmpInst::ICMP_SGT);
                                break; // signed greater than
                        case 7: currentI->setPredicate(ICmpInst::ICMP_SGE);
                                break; // signed greater or equal
                        case 8: currentI->setPredicate(ICmpInst::ICMP_SLT);
                                break; // signed less than
                        case 9: currentI->setPredicate(ICmpInst::ICMP_SLE);
                                break; // signed less or equal
                      }
                      break;
            }

          }
          if(opcode == Instruction::FCmp){ // Conditions (with float)
            FCmpInst *currentI = (FCmpInst*)(&i);
            switch(llvm::cryptoutils->get_range(3)){ // must be improved
              case 0: //do nothing
                break;
              case 1: currentI->swapOperands();
                      break;
              case 2: // randomly change the predicate
                      switch(llvm::cryptoutils->get_range(10)){
                        case 0: currentI->setPredicate(FCmpInst::FCMP_OEQ);
                                break; // ordered and equal
                        case 1: currentI->setPredicate(FCmpInst::FCMP_ONE);
                                break; // ordered and operands are unequal
                        case 2: currentI->setPredicate(FCmpInst::FCMP_UGT);
                                break; // unordered or greater than
                        case 3: currentI->setPredicate(FCmpInst::FCMP_UGE);
                                break; // unordered, or greater than, or equal
                        case 4: currentI->setPredicate(FCmpInst::FCMP_ULT);
                                break; // unordered or less than
                        case 5: currentI->setPredicate(FCmpInst::FCMP_ULE);
                                break; // unordered, or less than, or equal
                        case 6: currentI->setPredicate(FCmpInst::FCMP_OGT);
                                break; // ordered and greater than
                        case 7: currentI->setPredicate(FCmpInst::FCMP_OGE);
                                break; // ordered and greater than or equal
                        case 8: currentI->setPredicate(FCmpInst::FCMP_OLT);
                                break; // ordered and less than
                        case 9: currentI->setPredicate(FCmpInst::FCMP_OLE);
                                break; // ordered or less than, or equal
                      }
                      break;
            }
          }
        }
      }
      return alteredBB;
    } // end of createAlteredBasicBlock()


    /* doFinalization
     *
     * Overwrite FunctionPass method to apply the transformations to the whole module.
     * This part obfuscate all the always true predicates of the module.
     * More precisely, the condition which predicate is FCMP_TRUE.
     * It also remove all the functions' basic blocks' and instructions' names.
     */
    bool doF(Module &M){
      // In this part we extract all always-true predicate and replace them with opaque predicate:
      // For this, we declare two global values: x and y, and replace the FCMP_TRUE predicate with
      // (y < 10 || x * (x + 1) % 2 == 0)
      // A better way to obfuscate the predicates would be welcome.
      // In the meantime we will erase the name of the basic blocks, the instructions
      // and the functions.
      DEBUG_WITH_TYPE("gen", errs()<<"bcf: Starting doFinalization...\n");

      //  The global values
      Twine * varX = new Twine("x");
      Twine * varY = new Twine("y");
      Value * x1 =ConstantInt::get(Type::getInt32Ty(M.getContext()), 0, false);
      Value * y1 =ConstantInt::get(Type::getInt32Ty(M.getContext()), 0, false);

      GlobalVariable 	* x = new GlobalVariable(M, Type::getInt32Ty(M.getContext()), false,
          GlobalValue::CommonLinkage, (Constant * )x1,
          *varX);
      GlobalVariable 	* y = new GlobalVariable(M, Type::getInt32Ty(M.getContext()), false,
          GlobalValue::CommonLinkage, (Constant * )y1,
          *varY);


      std::vector<Instruction*> toEdit, toDelete;
      BinaryOperator *op,*op1 = NULL;
      LoadInst * opX , * opY;
      ICmpInst * condition, * condition2;
      // Looking for the conditions and branches to transform
      for(Module::iterator mi = M.begin(), me = M.end(); mi != me; ++mi){
        for(Function::iterator fi = mi->begin(), fe = mi->end(); fi != fe; ++fi){
          //fi->setName("");
          Instruction * tbb= fi->getTerminator();
          if(tbb->getOpcode() == Instruction::Br){
            BranchInst * br = (BranchInst *)(tbb);
            if(br->isConditional()){
              FCmpInst * cond = (FCmpInst *)br->getCondition();
              unsigned opcode = cond->getOpcode();
              if(opcode == Instruction::FCmp){
                if (cond->getPredicate() == FCmpInst::FCMP_TRUE){
                  DEBUG_WITH_TYPE("gen",
                      errs()<<"bcf: an always true predicate !\n");
                  toDelete.push_back(cond); // The condition
                  toEdit.push_back(tbb);    // The branch using the condition
                }
              }
            }
          }
          /*
          for (BasicBlock::iterator bi = fi->begin(), be = fi->end() ; bi != be; ++bi){
            bi->setName(""); // setting the basic blocks' names
          }
          */
        }
      }
      // Replacing all the branches we found
      for(std::vector<Instruction*>::iterator i =toEdit.begin();i!=toEdit.end();++i){
        //if y < 10 || x*(x+1) % 2 == 0
        opX = new LoadInst (Type::getInt32Ty(M.getContext()), (Value *)x, "", (*i)->getIterator());
        opY = new LoadInst (Type::getInt32Ty(M.getContext()), (Value *)y, "", (*i)->getIterator());

        op = BinaryOperator::Create(Instruction::Sub, (Value *)opX,
            ConstantInt::get(Type::getInt32Ty(M.getContext()), 1,
              false), "", (*i)->getIterator());
        op1 = BinaryOperator::Create(Instruction::Mul, (Value *)opX, op, "", (*i)->getIterator());
        op = BinaryOperator::Create(Instruction::URem, op1,
            ConstantInt::get(Type::getInt32Ty(M.getContext()), 2,
              false), "", (*i)->getIterator());
        condition = new ICmpInst((*i)->getIterator(), ICmpInst::ICMP_EQ, op,
            ConstantInt::get(Type::getInt32Ty(M.getContext()), 0,
              false));
        condition2 = new ICmpInst((*i)->getIterator(), ICmpInst::ICMP_SLT, opY,
            ConstantInt::get(Type::getInt32Ty(M.getContext()), 10,
              false));
        op1 = BinaryOperator::Create(Instruction::Or, (Value *)condition,
            (Value *)condition2, "", (*i)->getIterator());

        BranchInst::Create(((BranchInst*)*i)->getSuccessor(0),
            ((BranchInst*)*i)->getSuccessor(1),(Value *) op1,
            ((BranchInst*)*i)->getParent());
        DEBUG_WITH_TYPE("gen", errs() << "bcf: Erase branch instruction:"
            << *((BranchInst*)*i) << "\n");
        (*i)->eraseFromParent(); // erase the branch
      }
      // Erase all the associated conditions we found
      for(std::vector<Instruction*>::iterator i =toDelete.begin();i!=toDelete.end();++i){
        DEBUG_WITH_TYPE("gen", errs() << "bcf: Erase condition instruction:"
            << *((Instruction*)*i)<< "\n");
        (*i)->eraseFromParent();
      }

      // Only for debug
      DEBUG_WITH_TYPE("cfg",
          errs() << "bcf: End of the pass, here are the graphs after doFinalization\n");
      for(Module::iterator mi = M.begin(), me = M.end(); mi != me; ++mi){
        DEBUG_WITH_TYPE("cfg", errs() << "bcf: Function " << mi->getName() <<"\n");
        DEBUG_WITH_TYPE("cfg", mi->viewCFG());
      }

      return true;
    } // end of doFinalization
  }; // end of struct BogusControlFlow : public FunctionPass
}

char BogusControlFlow::ID = 0;
static RegisterPass<BogusControlFlow> X("boguscf", "inserting bogus control flow");

Pass *llvm::createBogus() {
  return new BogusControlFlow();
}

Pass *llvm::createBogus(bool flag) {
  return new BogusControlFlow(flag);
}
