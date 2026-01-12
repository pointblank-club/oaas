//===- ConstantObfuscation.h - Constant Obfuscation Pass -----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains includes and defines for the constant obfuscation pass.
// This pass encodes literal constants into computed forms to defeat static
// analysis.
//
// SAFE FOR BINARY OBFUSCATION MODE (McSema lifted IR):
// - Does NOT modify control flow
// - Only transforms constant values into computed sequences
// - CFG-neutral, works purely on SSA values
//
//===----------------------------------------------------------------------===//

#ifndef _CONSTANT_OBFUSCATION_H_
#define _CONSTANT_OBFUSCATION_H_

#include "llvm/Pass.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/CryptoUtils.h"

using namespace llvm;
using namespace std;

namespace llvm {
Pass *createConstantObfuscation();
Pass *createConstantObfuscation(bool flag);
}

#endif
