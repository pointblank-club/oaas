//===- Substitution.cpp - Substitution Obfuscation
// pass-------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements operators substitution's pass
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Obfuscation/Substitution.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Obfuscation/Utils.h"
#include "llvm/IR/Intrinsics.h"

#define DEBUG_TYPE "substitution"

#define NUMBER_ADD_SUBST 4
#define NUMBER_SUB_SUBST 3
#define NUMBER_AND_SUBST 2
#define NUMBER_OR_SUBST 2
#define NUMBER_XOR_SUBST 2

static cl::opt<int>
ObfTimes("sub_loop",
         cl::desc("Choose how many time the -sub pass loops on a function"),
         cl::value_desc("number of times"), cl::init(1), cl::Optional);

// Binary-safe mode for McSema-lifted IR compatibility
static cl::opt<bool>
BinarySafeMode("sub_binary_safe",
    cl::desc("Enable binary-safe mode for McSema-lifted IR (skips sub_* functions)"),
    cl::init(false), cl::Optional);


// Stats
STATISTIC(Add, "Add substitued");
STATISTIC(Sub, "Sub substitued");
// STATISTIC(Mul,  "Mul substitued");
// STATISTIC(Div,  "Div substitued");
// STATISTIC(Rem,  "Rem substitued");
// STATISTIC(Shi,  "Shift substitued");
STATISTIC(And, "And substitued");
STATISTIC(Or, "Or substitued");
STATISTIC(Xor, "Xor substitued");

namespace {

struct Substitution : public FunctionPass {
  static char ID; // Pass identification, replacement for typeid
  void (Substitution::*funcAdd[NUMBER_ADD_SUBST])(BinaryOperator *bo);
  void (Substitution::*funcSub[NUMBER_SUB_SUBST])(BinaryOperator *bo);
  void (Substitution::*funcAnd[NUMBER_AND_SUBST])(BinaryOperator *bo);
  void (Substitution::*funcOr[NUMBER_OR_SUBST])(BinaryOperator *bo);
  void (Substitution::*funcXor[NUMBER_XOR_SUBST])(BinaryOperator *bo);
  bool flag;

  Substitution() : FunctionPass(ID) {}

  Substitution(bool flag) : FunctionPass(ID) {
    this->flag = flag;
    funcAdd[0] = &Substitution::addNeg;
    funcAdd[1] = &Substitution::addDoubleNeg;
    funcAdd[2] = &Substitution::addRand;
    funcAdd[3] = &Substitution::addRand2;

    funcSub[0] = &Substitution::subNeg;
    funcSub[1] = &Substitution::subRand;
    funcSub[2] = &Substitution::subRand2;

    funcAnd[0] = &Substitution::andSubstitution;
    funcAnd[1] = &Substitution::andSubstitutionRand;

    funcOr[0] = &Substitution::orSubstitution;
    funcOr[1] = &Substitution::orSubstitutionRand;

    funcXor[0] = &Substitution::xorSubstitution;
    funcXor[1] = &Substitution::xorSubstitutionRand;
  }

  bool runOnFunction(Function &F) override;
  bool substitute(Function *f);

  void addNeg(BinaryOperator *bo);
  void addDoubleNeg(BinaryOperator *bo);
  void addRand(BinaryOperator *bo);
  void addRand2(BinaryOperator *bo);

  void subNeg(BinaryOperator *bo);
  void subRand(BinaryOperator *bo);
  void subRand2(BinaryOperator *bo);

  void andSubstitution(BinaryOperator *bo);
  void andSubstitutionRand(BinaryOperator *bo);

  void orSubstitution(BinaryOperator *bo);
  void orSubstitutionRand(BinaryOperator *bo);

  void xorSubstitution(BinaryOperator *bo);
  void xorSubstitutionRand(BinaryOperator *bo);

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
};
}

char Substitution::ID = 0;
static RegisterPass<Substitution> X("substitution", "operators substitution");
Pass *llvm::createSubstitution(bool flag) { return new Substitution(flag); }

bool Substitution::runOnFunction(Function &F) {
  Function *tmp = &F;

  // Binary-safe mode: Skip McSema-generated functions entirely
  // Note: Substitution is generally safe, but we still skip for consistency
  if (BinarySafeMode && isMcSemaFunction(tmp)) {
    DEBUG_WITH_TYPE("sub", errs() << "sub: Skipping McSema function in binary-safe mode: "
        << F.getName() << "\n");
    return false;
  }

  // Check if the percentage is correct
  if (ObfTimes <= 0) {
    errs()<<"Substitution application number -sub_loop=x must be x > 0";
    return false;
  }

  // Do we obfuscate
  if (toObfuscate(flag, tmp, "sub")) {
    substitute(tmp);
    return true;
  }

  return false;
}

bool Substitution::substitute(Function *f) {
  Function *tmp = f;

  // Loop for the number of time we run the pass on the function
  int times = ObfTimes;
  do {
    for (Function::iterator bb = tmp->begin(); bb != tmp->end(); ++bb) {
      for (BasicBlock::iterator inst = bb->begin(); inst != bb->end(); ++inst) {
        if (inst->isBinaryOp()) {
          switch (inst->getOpcode()) {
          case BinaryOperator::Add:
            // case BinaryOperator::FAdd:
            // Substitute with random add operation
            (this->*funcAdd[llvm::cryptoutils->get_range(NUMBER_ADD_SUBST)])(
                cast<BinaryOperator>(inst));
            ++Add;
            break;
          case BinaryOperator::Sub:
            // case BinaryOperator::FSub:
            // Substitute with random sub operation
            (this->*funcSub[llvm::cryptoutils->get_range(NUMBER_SUB_SUBST)])(
                cast<BinaryOperator>(inst));
            ++Sub;
            break;
          case BinaryOperator::Mul:
          case BinaryOperator::FMul:
            //++Mul;
            break;
          case BinaryOperator::UDiv:
          case BinaryOperator::SDiv:
          case BinaryOperator::FDiv:
            //++Div;
            break;
          case BinaryOperator::URem:
          case BinaryOperator::SRem:
          case BinaryOperator::FRem:
            //++Rem;
            break;
          case Instruction::Shl:
            //++Shi;
            break;
          case Instruction::LShr:
            //++Shi;
            break;
          case Instruction::AShr:
            //++Shi;
            break;
          case Instruction::And:
            (this->*
             funcAnd[llvm::cryptoutils->get_range(2)])(cast<BinaryOperator>(inst));
            ++And;
            break;
          case Instruction::Or:
            (this->*
             funcOr[llvm::cryptoutils->get_range(2)])(cast<BinaryOperator>(inst));
            ++Or;
            break;
          case Instruction::Xor:
            (this->*
             funcXor[llvm::cryptoutils->get_range(2)])(cast<BinaryOperator>(inst));
            ++Xor;
            break;
          default:
            break;
          }              // End switch
        }                // End isBinaryOp
      }                  // End for basickblock
    }                    // End for Function
  } while (--times > 0); // for times
  return false;
}

// Implementation of a = b - (-c)
void Substitution::addNeg(BinaryOperator *bo) {
  BinaryOperator *op = NULL;

  // Create sub
  if (bo->getOpcode() == Instruction::Add) {
    op = BinaryOperator::CreateNeg(bo->getOperand(1), "", bo->getIterator()->getIterator());
    op =
        BinaryOperator::Create(Instruction::Sub, bo->getOperand(0), op, "", bo->getIterator()->getIterator());

    // Check signed wrap
    //op->setHasNoSignedWrap(bo->hasNoSignedWrap());
    //op->setHasNoUnsignedWrap(bo->hasNoUnsignedWrap());

    bo->replaceAllUsesWith(op);
  }/* else {
    op = BinaryOperator::CreateFNeg(bo->getOperand(1), "", bo->getIterator());
    op = BinaryOperator::Create(Instruction::FSub, bo->getOperand(0), op, "",
                                bo);
  }*/
}

// Implementation of a = -(-b + (-c))
void Substitution::addDoubleNeg(BinaryOperator *bo) {
  Value *op, *op2 = NULL;

  if (bo->getOpcode() == Instruction::Add) {
    op = BinaryOperator::CreateNeg(bo->getOperand(0), "", bo->getIterator());
    op2 = BinaryOperator::CreateNeg(bo->getOperand(1), "", bo->getIterator());
    op = BinaryOperator::Create(Instruction::Add, op, op2, "", bo->getIterator());
    op = BinaryOperator::CreateNeg(op, "", bo->getIterator());

    // Check signed wrap
    //op->setHasNoSignedWrap(bo->hasNoSignedWrap());
    //op->setHasNoUnsignedWrap(bo->hasNoUnsignedWrap());
  } else {
    op = UnaryOperator::CreateFNeg(bo->getOperand(0), "", bo->getIterator());
    op2 = UnaryOperator::CreateFNeg(bo->getOperand(1), "", bo->getIterator());
    op = BinaryOperator::Create(Instruction::FAdd, op, op2, "", bo->getIterator());
    op = UnaryOperator::CreateFNeg(op, "", bo->getIterator());
  }

  bo->replaceAllUsesWith(op);
}

// Implementation of  r = rand (); a = b + r; a = a + c; a = a - r
void Substitution::addRand(BinaryOperator *bo) {
  BinaryOperator *op = NULL;

  if (bo->getOpcode() == Instruction::Add) {
    Type *ty = bo->getType();
    ConstantInt *co =
        (ConstantInt *)ConstantInt::get(ty, llvm::cryptoutils->get_uint64_t());
    op =
        BinaryOperator::Create(Instruction::Add, bo->getOperand(0), co, "", bo->getIterator());
    op =
        BinaryOperator::Create(Instruction::Add, op, bo->getOperand(1), "", bo->getIterator());
    op = BinaryOperator::Create(Instruction::Sub, op, co, "", bo->getIterator());

    // Check signed wrap
    //op->setHasNoSignedWrap(bo->hasNoSignedWrap());
    //op->setHasNoUnsignedWrap(bo->hasNoUnsignedWrap());

    bo->replaceAllUsesWith(op);
  }
  /* else {
      Type *ty = bo->getType();
      ConstantFP *co =
  (ConstantFP*)ConstantFP::get(ty,(float)llvm::cryptoutils->get_uint64_t());
      op = BinaryOperator::Create(Instruction::FAdd,bo->getOperand(0),co,"",bo);
      op = BinaryOperator::Create(Instruction::FAdd,op,bo->getOperand(1),"",bo);
      op = BinaryOperator::Create(Instruction::FSub,op,co,"",bo);
  } */
}

// Implementation of r = rand (); a = b - r; a = a + b; a = a + r
void Substitution::addRand2(BinaryOperator *bo) {
  BinaryOperator *op = NULL;

  if (bo->getOpcode() == Instruction::Add) {
    Type *ty = bo->getType();
    ConstantInt *co =
        (ConstantInt *)ConstantInt::get(ty, llvm::cryptoutils->get_uint64_t());
    op =
        BinaryOperator::Create(Instruction::Sub, bo->getOperand(0), co, "", bo->getIterator());
    op =
        BinaryOperator::Create(Instruction::Add, op, bo->getOperand(1), "", bo->getIterator());
    op = BinaryOperator::Create(Instruction::Add, op, co, "", bo->getIterator());

    // Check signed wrap
    //op->setHasNoSignedWrap(bo->hasNoSignedWrap());
    //op->setHasNoUnsignedWrap(bo->hasNoUnsignedWrap());

    bo->replaceAllUsesWith(op);
  }
  /* else {
      Type *ty = bo->getType();
      ConstantFP *co =
  (ConstantFP*)ConstantFP::get(ty,(float)llvm::cryptoutils->get_uint64_t());
      op = BinaryOperator::Create(Instruction::FAdd,bo->getOperand(0),co,"",bo);
      op = BinaryOperator::Create(Instruction::FAdd,op,bo->getOperand(1),"",bo);
      op = BinaryOperator::Create(Instruction::FSub,op,co,"",bo);
  } */
}

// Implementation of a = b + (-c)
void Substitution::subNeg(BinaryOperator *bo) {
  Value *op = NULL;

  if (bo->getOpcode() == Instruction::Sub) {
    op = BinaryOperator::CreateNeg(bo->getOperand(1), "", bo->getIterator());
    op =
        BinaryOperator::Create(Instruction::Add, bo->getOperand(0), op, "", bo->getIterator());

    // Check signed wrap
    //op->setHasNoSignedWrap(bo->hasNoSignedWrap());
    //op->setHasNoUnsignedWrap(bo->hasNoUnsignedWrap());
  } else {
    op = UnaryOperator::CreateFNeg(bo->getOperand(1), "", bo->getIterator());
    op = BinaryOperator::Create(Instruction::FAdd, bo->getOperand(0), op, "",
                                bo->getIterator());
  }

  bo->replaceAllUsesWith(op);
}

// Implementation of  r = rand (); a = b + r; a = a - c; a = a - r
void Substitution::subRand(BinaryOperator *bo) {
  BinaryOperator *op = NULL;

  if (bo->getOpcode() == Instruction::Sub) {
    Type *ty = bo->getType();
    ConstantInt *co =
        (ConstantInt *)ConstantInt::get(ty, llvm::cryptoutils->get_uint64_t());
    op =
        BinaryOperator::Create(Instruction::Add, bo->getOperand(0), co, "", bo->getIterator());
    op =
        BinaryOperator::Create(Instruction::Sub, op, bo->getOperand(1), "", bo->getIterator());
    op = BinaryOperator::Create(Instruction::Sub, op, co, "", bo->getIterator());

    // Check signed wrap
    //op->setHasNoSignedWrap(bo->hasNoSignedWrap());
    //op->setHasNoUnsignedWrap(bo->hasNoUnsignedWrap());

    bo->replaceAllUsesWith(op);
  }
  /* else {
      Type *ty = bo->getType();
      ConstantFP *co =
  (ConstantFP*)ConstantFP::get(ty,(float)llvm::cryptoutils->get_uint64_t());
      op = BinaryOperator::Create(Instruction::FAdd,bo->getOperand(0),co,"",bo);
      op = BinaryOperator::Create(Instruction::FSub,op,bo->getOperand(1),"",bo);
      op = BinaryOperator::Create(Instruction::FSub,op,co,"",bo);
  } */
}

// Implementation of  r = rand (); a = b - r; a = a - c; a = a + r
void Substitution::subRand2(BinaryOperator *bo) {
  BinaryOperator *op = NULL;

  if (bo->getOpcode() == Instruction::Sub) {
    Type *ty = bo->getType();
    ConstantInt *co =
        (ConstantInt *)ConstantInt::get(ty, llvm::cryptoutils->get_uint64_t());
    op =
        BinaryOperator::Create(Instruction::Sub, bo->getOperand(0), co, "", bo->getIterator());
    op =
        BinaryOperator::Create(Instruction::Sub, op, bo->getOperand(1), "", bo->getIterator());
    op = BinaryOperator::Create(Instruction::Add, op, co, "", bo->getIterator());

    // Check signed wrap
    //op->setHasNoSignedWrap(bo->hasNoSignedWrap());
    //op->setHasNoUnsignedWrap(bo->hasNoUnsignedWrap());

    bo->replaceAllUsesWith(op);
  }
  /* else {
      Type *ty = bo->getType();
      ConstantFP *co =
  (ConstantFP*)ConstantFP::get(ty,(float)llvm::cryptoutils->get_uint64_t());
      op = BinaryOperator::Create(Instruction::FSub,bo->getOperand(0),co,"",bo);
      op = BinaryOperator::Create(Instruction::FSub,op,bo->getOperand(1),"",bo);
      op = BinaryOperator::Create(Instruction::FAdd,op,co,"",bo);
  } */
}

// Implementation of a = b & c => a = (b^~c)& b
void Substitution::andSubstitution(BinaryOperator *bo) {
  BinaryOperator *op = NULL;

  // Create NOT on second operand => ~c
  op = BinaryOperator::CreateNot(bo->getOperand(1), "", bo->getIterator());

  // Create XOR => (b^~c)
  BinaryOperator *op1 =
      BinaryOperator::Create(Instruction::Xor, bo->getOperand(0), op, "", bo->getIterator());

  // Create AND => (b^~c) & b
  op = BinaryOperator::Create(Instruction::And, op1, bo->getOperand(0), "", bo->getIterator());
  bo->replaceAllUsesWith(op);
}

// Implementation of a = a && b <=> !(!a | !b) && (r | !r)
void Substitution::andSubstitutionRand(BinaryOperator *bo) {
  // Copy of the BinaryOperator type to create the random number with the
  // same type of the operands
  Type *ty = bo->getType();

  // r (Random number)
  ConstantInt *co =
      (ConstantInt *)ConstantInt::get(ty, llvm::cryptoutils->get_uint64_t());

  // !a
  BinaryOperator *op = BinaryOperator::CreateNot(bo->getOperand(0), "", bo->getIterator());

  // !b
  BinaryOperator *op1 = BinaryOperator::CreateNot(bo->getOperand(1), "", bo->getIterator());

  // !r
  BinaryOperator *opr = BinaryOperator::CreateNot(co, "", bo->getIterator());

  // (!a | !b)
  BinaryOperator *opa =
      BinaryOperator::Create(Instruction::Or, op, op1, "", bo->getIterator());

  // (r | !r)
  opr = BinaryOperator::Create(Instruction::Or, co, opr, "", bo->getIterator());

  // !(!a | !b)
  op = BinaryOperator::CreateNot(opa, "", bo->getIterator());

  // !(!a | !b) && (r | !r)
  op = BinaryOperator::Create(Instruction::And, op, opr, "", bo->getIterator());

  // We replace all the old AND operators with the new one transformed
  bo->replaceAllUsesWith(op);
}

// Implementation of a = b | c => a = (b & c) | (b ^ c)
void Substitution::orSubstitutionRand(BinaryOperator *bo) {

  Type *ty = bo->getType();
  ConstantInt *co =
      (ConstantInt *)ConstantInt::get(ty, llvm::cryptoutils->get_uint64_t());

  // !a
  BinaryOperator *op = BinaryOperator::CreateNot(bo->getOperand(0), "", bo->getIterator());

  // !b
  BinaryOperator *op1 = BinaryOperator::CreateNot(bo->getOperand(1), "", bo->getIterator());

  // !r
  BinaryOperator *op2 = BinaryOperator::CreateNot(co, "", bo->getIterator());

  // !a && r
  BinaryOperator *op3 =
      BinaryOperator::Create(Instruction::And, op, co, "", bo->getIterator());

  // a && !r
  BinaryOperator *op4 =
      BinaryOperator::Create(Instruction::And, bo->getOperand(0), op2, "", bo->getIterator());

  // !b && r
  BinaryOperator *op5 =
      BinaryOperator::Create(Instruction::And, op1, co, "", bo->getIterator());

  // b && !r
  BinaryOperator *op6 =
      BinaryOperator::Create(Instruction::And, bo->getOperand(1), op2, "", bo->getIterator());

  // (!a && r) || (a && !r)
  op3 = BinaryOperator::Create(Instruction::Or, op3, op4, "", bo->getIterator());

  // (!b && r) ||(b && !r)
  op4 = BinaryOperator::Create(Instruction::Or, op5, op6, "", bo->getIterator());

  // (!a && r) || (a && !r) ^ (!b && r) ||(b && !r)
  op5 = BinaryOperator::Create(Instruction::Xor, op3, op4, "", bo->getIterator());

  // !a || !b
  op3 = BinaryOperator::Create(Instruction::Or, op, op1, "", bo->getIterator());

  // !(!a || !b)
  op3 = BinaryOperator::CreateNot(op3, "", bo->getIterator());

  // r || !r
  op4 = BinaryOperator::Create(Instruction::Or, co, op2, "", bo->getIterator());

  // !(!a || !b) && (r || !r)
  op4 = BinaryOperator::Create(Instruction::And, op3, op4, "", bo->getIterator());

  // [(!a && r) || (a && !r) ^ (!b && r) ||(b && !r) ] || [!(!a || !b) && (r ||
  // !r)]
  op = BinaryOperator::Create(Instruction::Or, op5, op4, "", bo->getIterator());
  bo->replaceAllUsesWith(op);
}

void Substitution::orSubstitution(BinaryOperator *bo) {
  BinaryOperator *op = NULL;

  // Creating first operand (b & c)
  op = BinaryOperator::Create(Instruction::And, bo->getOperand(0),
                              bo->getOperand(1), "", bo->getIterator());

  // Creating second operand (b ^ c)
  BinaryOperator *op1 = BinaryOperator::Create(
      Instruction::Xor, bo->getOperand(0), bo->getOperand(1), "", bo->getIterator());

  // final op
  op = BinaryOperator::Create(Instruction::Or, op, op1, "", bo->getIterator());
  bo->replaceAllUsesWith(op);
}

// Implementation of a = a ~ b => a = (!a && b) || (a && !b)
void Substitution::xorSubstitution(BinaryOperator *bo) {
  BinaryOperator *op = NULL;

  // Create NOT on first operand
  op = BinaryOperator::CreateNot(bo->getOperand(0), "", bo->getIterator()); // !a

  // Create AND
  op = BinaryOperator::Create(Instruction::And, bo->getOperand(1), op, "",
                              bo->getIterator()); // !a && b

  // Create NOT on second operand
  BinaryOperator *op1 =
      BinaryOperator::CreateNot(bo->getOperand(1), "", bo->getIterator()); // !b

  // Create AND
  op1 = BinaryOperator::Create(Instruction::And, bo->getOperand(0), op1, "",
                               bo->getIterator()); // a && !b

  // Create OR
  op = BinaryOperator::Create(Instruction::Or, op, op1, "",
                              bo->getIterator()); // (!a && b) || (a && !b)
  bo->replaceAllUsesWith(op);
}

// implementation of a = a ^ b <=> (a ^ r) ^ (b ^ r) <=> (!a && r || a && !r) ^
// (!b && r || b && !r)
// note : r is a random number
void Substitution::xorSubstitutionRand(BinaryOperator *bo) {
  BinaryOperator *op = NULL;

  Type *ty = bo->getType();
  ConstantInt *co =
      (ConstantInt *)ConstantInt::get(ty, llvm::cryptoutils->get_uint64_t());

  // !a
  op = BinaryOperator::CreateNot(bo->getOperand(0), "", bo->getIterator());

  // !a && r
  op = BinaryOperator::Create(Instruction::And, co, op, "", bo->getIterator());

  // !r
  BinaryOperator *opr = BinaryOperator::CreateNot(co, "", bo->getIterator());

  // a && !r
  BinaryOperator *op1 =
      BinaryOperator::Create(Instruction::And, bo->getOperand(0), opr, "", bo->getIterator());

  // !b
  BinaryOperator *op2 = BinaryOperator::CreateNot(bo->getOperand(1), "", bo->getIterator());

  // !b && r
  op2 = BinaryOperator::Create(Instruction::And, op2, co, "", bo->getIterator());

  // b && !r
  BinaryOperator *op3 =
      BinaryOperator::Create(Instruction::And, bo->getOperand(1), opr, "", bo->getIterator());

  // (!a && r) || (a && !r)
  op = BinaryOperator::Create(Instruction::Or, op, op1, "", bo->getIterator());

  // (!b && r) || (b && !r)
  op1 = BinaryOperator::Create(Instruction::Or, op2, op3, "", bo->getIterator());

  // (!a && r) || (a && !r) ^ (!b && r) || (b && !r)
  op = BinaryOperator::Create(Instruction::Xor, op, op1, "", bo->getIterator());
  bo->replaceAllUsesWith(op);
}

