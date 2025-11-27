#pragma once

#include "mlir/Pass/Pass.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"

namespace mlir {
namespace obs {

// ======================= STRING ENCRYPTION PASS ============================
struct StringEncryptPass 
    : public PassWrapper<StringEncryptPass, OperationPass<ModuleOp>> {

  StringEncryptPass() = default;
  StringEncryptPass(const std::string &key) : key(key) {}

  // CLI Support
  StringRef getArgument() const override { return "string-encrypt"; }
  StringRef getDescription() const override {
    return "Encrypt string attributes using XOR";
  }

  void runOnOperation() override;

  std::string key = "default_key";
};

std::unique_ptr<Pass> createStringEncryptPass(llvm::StringRef key);


// ======================== SYMBOL OBFUSCATION PASS ==========================
struct SymbolObfuscatePass
    : public PassWrapper<SymbolObfuscatePass, OperationPass<ModuleOp>> {

  SymbolObfuscatePass() = default;
  SymbolObfuscatePass(const std::string &key) : key(key) {}

  StringRef getArgument() const override { return "symbol-obfuscate"; }
  StringRef getDescription() const override {
    return "Obfuscate symbol names randomly";
  }

  void runOnOperation() override;

  std::string key = "seed";
};

std::unique_ptr<Pass> createSymbolObfuscatePass(llvm::StringRef key);


} // namespace obs
} // namespace mlir
