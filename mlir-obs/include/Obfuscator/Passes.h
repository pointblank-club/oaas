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


// =================== CONSTANT OBFUSCATION PASS =============================
struct ConstantObfuscationPass
    : public PassWrapper<ConstantObfuscationPass, OperationPass<ModuleOp>> {

  ConstantObfuscationPass() = default;
  ConstantObfuscationPass(const std::string &key) : key(key) {}

  StringRef getArgument() const override { return "constant-obfuscate"; }
  StringRef getDescription() const override {
    return "Obfuscate all constants: strings, integers, floats (Func Dialect compatible)";
  }

  void runOnOperation() override;

  std::string key = "default_key";
};

std::unique_ptr<Pass> createConstantObfuscationPass(llvm::StringRef key);


// ======================== SYMBOL OBFUSCATION PASS ==========================
// Supports BOTH func::FuncOp (ClangIR/high-level MLIR)
// AND LLVM::LLVMFuncOp (post-lowering from mlir-translate)
struct SymbolObfuscatePass
    : public PassWrapper<SymbolObfuscatePass, OperationPass<ModuleOp>> {

  SymbolObfuscatePass() = default;
  SymbolObfuscatePass(const std::string &key) : key(key) {}

  StringRef getArgument() const override { return "symbol-obfuscate"; }
  StringRef getDescription() const override {
    return "Obfuscate symbol names randomly (supports func + LLVM dialects)";
  }

  void runOnOperation() override;

  std::string key = "seed";

private:
  // Helper to process func::FuncOp (ClangIR / high-level MLIR input)
  void processFuncDialect();

  // Helper to process LLVM::LLVMFuncOp (post-lowering from mlir-translate)
  void processLLVMDialect();
};

std::unique_ptr<Pass> createSymbolObfuscatePass(llvm::StringRef key);


// ===================== CRYPTOGRAPHIC HASH PASS =============================
struct CryptoHashPass
    : public PassWrapper<CryptoHashPass, OperationPass<ModuleOp>> {

  enum class HashAlgorithm {
    SHA256,
    BLAKE2B,
    SIPHASH
  };

  CryptoHashPass() = default;
  CryptoHashPass(HashAlgorithm algo, const std::string &salt, unsigned hashLength)
      : algorithm(algo), salt(salt), hashLength(hashLength) {}

  StringRef getArgument() const override { return "crypto-hash"; }
  StringRef getDescription() const override {
    return "Cryptographically hash symbol names using SHA256/BLAKE2B/SipHash";
  }

  void runOnOperation() override;

  HashAlgorithm algorithm = HashAlgorithm::SHA256;
  std::string salt = "";
  unsigned hashLength = 12;  // Truncate hash to N characters
};

std::unique_ptr<Pass> createCryptoHashPass(
    CryptoHashPass::HashAlgorithm algo = CryptoHashPass::HashAlgorithm::SHA256,
    llvm::StringRef salt = "",
    unsigned hashLength = 12
);

// =================== POLYGEIST-AWARE CONTROL FLOW PASS =====================
// New pass: Operates on SCF dialect from Polygeist for better obfuscation
struct SCFObfuscatePass
    : public PassWrapper<SCFObfuscatePass, OperationPass<ModuleOp>> {

  SCFObfuscatePass() = default;

  StringRef getArgument() const override { return "scf-obfuscate"; }
  StringRef getDescription() const override {
    return "Obfuscate structured control flow (Polygeist SCF dialect)";
  }

  void runOnOperation() override;
};

std::unique_ptr<Pass> createSCFObfuscatePass();


// ======================== SCF OBFUSCATION PASS ==============================
// Operates on SCF dialect operations (loops, conditionals)
// Adds opaque predicates and control flow complexity
struct SCFObfuscatePass
    : public PassWrapper<SCFObfuscatePass, OperationPass<ModuleOp>> {

  SCFObfuscatePass() = default;

  StringRef getArgument() const override { return "scf-obfuscate"; }
  StringRef getDescription() const override {
    return "Obfuscate SCF control flow with opaque predicates";
  }

  void runOnOperation() override;
};

std::unique_ptr<Pass> createSCFObfuscatePass();


} // namespace obs
} // namespace mlir
