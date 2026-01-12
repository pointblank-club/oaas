//===- IndirectBranch.h - Indirect Branch Obfuscation Pass ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains includes and defines for the indirect branch pass.
// This pass converts direct branches to indirect branches via blockaddress
// to defeat static control flow analysis.
//
// MEDIUM RISK FOR BINARY OBFUSCATION MODE (McSema lifted IR):
// - IR remains structurally correct
// - Must not apply to exception handling blocks
// - Backend support varies
//
//===----------------------------------------------------------------------===//

#ifndef _INDIRECT_BRANCH_H_
#define _INDIRECT_BRANCH_H_

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
Pass *createIndirectBranch();
Pass *createIndirectBranch(bool flag);
}

#endif
