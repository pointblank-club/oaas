//===- BlockReordering.h - Basic Block Reordering Pass -------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains includes and defines for the basic block reordering pass.
// This pass rearranges the order of basic blocks without changing CFG edges.
//
// SAFE FOR BINARY OBFUSCATION MODE (McSema lifted IR):
// - Does NOT modify control flow edges
// - Only changes block layout order
// - CFG-preserving, confuses static disassemblers
//
//===----------------------------------------------------------------------===//

#ifndef _BLOCK_REORDERING_H_
#define _BLOCK_REORDERING_H_

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
Pass *createBlockReordering();
Pass *createBlockReordering(bool flag);
}

#endif
