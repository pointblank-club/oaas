//===- IndirectCall.h - Indirect Call Obfuscation Pass -------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains includes and defines for the indirect call pass.
// This pass converts direct function calls to indirect calls via function
// pointers to defeat static call graph analysis.
//
// MEDIUM RISK FOR BINARY OBFUSCATION MODE (McSema lifted IR):
// - Does NOT alter CFG topology
// - Must filter out McSema sub_* functions
// - Safe for user-level functions only
//
//===----------------------------------------------------------------------===//

#ifndef _INDIRECT_CALL_H_
#define _INDIRECT_CALL_H_

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
Pass *createIndirectCall();
Pass *createIndirectCall(bool flag);
}

#endif
