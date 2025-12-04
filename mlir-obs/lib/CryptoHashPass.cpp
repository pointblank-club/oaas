#include "Obfuscator/Passes.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

#include <openssl/evp.h>
#include <iomanip>
#include <sstream>
#include <cstring>

using namespace mlir;
using namespace mlir::obs;

namespace {

/// Compute SHA256 hash of input string using modern EVP API
static std::string computeSHA256(const std::string &input, const std::string &salt) {
  std::string data = salt + input + salt;  // Salt prefix and suffix

  unsigned char hash[EVP_MAX_MD_SIZE];
  unsigned int hashLen = 0;

  EVP_MD_CTX *mdctx = EVP_MD_CTX_new();
  const EVP_MD *md = EVP_sha256();

  EVP_DigestInit_ex(mdctx, md, NULL);
  EVP_DigestUpdate(mdctx, data.c_str(), data.size());
  EVP_DigestFinal_ex(mdctx, hash, &hashLen);
  EVP_MD_CTX_free(mdctx);

  std::stringstream ss;
  for (unsigned int i = 0; i < hashLen; i++) {
    ss << std::hex << std::setw(2) << std::setfill('0') << (int)hash[i];
  }
  return ss.str();
}

/// Compute BLAKE2B hash of input string
static std::string computeBLAKE2B(const std::string &input, const std::string &salt) {
  std::string data = salt + input + salt;

  unsigned char hash[EVP_MAX_MD_SIZE];
  unsigned int hashLen = 0;

  EVP_MD_CTX *mdctx = EVP_MD_CTX_new();
  const EVP_MD *md = EVP_blake2b512();

  EVP_DigestInit_ex(mdctx, md, NULL);
  EVP_DigestUpdate(mdctx, data.c_str(), data.size());
  EVP_DigestFinal_ex(mdctx, hash, &hashLen);
  EVP_MD_CTX_free(mdctx);

  std::stringstream ss;
  for (unsigned int i = 0; i < hashLen; i++) {
    ss << std::hex << std::setw(2) << std::setfill('0') << (int)hash[i];
  }
  return ss.str();
}

/// Compute SipHash (using SHA256 as fallback since OpenSSL doesn't have SipHash built-in)
/// For true SipHash, you'd need to link a separate library or implement it
static std::string computeSipHash(const std::string &input, const std::string &salt) {
  // For now, use SHA256 as a fast cryptographic hash
  // TODO: Add proper SipHash implementation if needed
  return computeSHA256(input, salt);
}

/// Generate obfuscated name from hash
static std::string generateHashedName(const std::string &originalName,
                                       CryptoHashPass::HashAlgorithm algo,
                                       const std::string &salt,
                                       unsigned hashLength) {
  std::string fullHash;

  switch (algo) {
    case CryptoHashPass::HashAlgorithm::SHA256:
      fullHash = computeSHA256(originalName, salt);
      break;
    case CryptoHashPass::HashAlgorithm::BLAKE2B:
      fullHash = computeBLAKE2B(originalName, salt);
      break;
    case CryptoHashPass::HashAlgorithm::SIPHASH:
      fullHash = computeSipHash(originalName, salt);
      break;
  }

  // Truncate to specified length
  std::string truncated = fullHash.substr(0, hashLength);

  // Add prefix to indicate it's a function
  return "f_" + truncated;
}

} // namespace

void CryptoHashPass::runOnOperation() {
  ModuleOp module = getOperation();
  MLIRContext *ctx = module.getContext();
  SymbolTable symbolTable(module);

  // Detect which dialect we're working with
  bool hasFuncDialect = false;
  bool hasLLVMDialect = false;

  module.walk([&](Operation *op) {
    if (llvm::isa<func::FuncOp>(op)) {
      hasFuncDialect = true;
    } else if (llvm::isa<LLVM::LLVMFuncOp>(op)) {
      hasLLVMDialect = true;
    }
  });

  // Process Func dialect (high-level MLIR from ClangIR/Polygeist)
  if (hasFuncDialect) {
    llvm::StringMap<std::string> renameMap;

    // Step 1: Collect rename map for func definitions
    module.walk([&](func::FuncOp func) {
      StringRef oldName = func.getSymName();

      // Skip main function to preserve entry point
      if (oldName == "main") {
        return;
      }

      // Skip functions that start with llvm. or mlir. (intrinsics)
      if (oldName.starts_with("llvm.") || oldName.starts_with("mlir.")) {
        return;
      }

      // Only assign a new name once per function
      if (renameMap.find(oldName) == renameMap.end()) {
        std::string newName = generateHashedName(
            oldName.str(),
            algorithm,
            salt,
            hashLength
        );
        renameMap[oldName] = newName;
      }
    });

    // Step 2: Update symbol references *before* renaming definitions
    module.walk([&](Operation *op) {
      SmallVector<NamedAttribute> updatedAttrs;
      bool changed = false;

      for (auto &attr : op->getAttrs()) {
        if (auto symAttr = llvm::dyn_cast<SymbolRefAttr>(attr.getValue())) {
          StringRef old = symAttr.getRootReference();
          auto it = renameMap.find(old);
          if (it != renameMap.end()) {
            auto newRef = SymbolRefAttr::get(ctx, it->second);
            updatedAttrs.emplace_back(attr.getName(), newRef);
            changed = true;
            continue;
          }
        }
        // No change -> keep original
        updatedAttrs.push_back(attr);
      }

      if (changed) {
        op->setAttrs(DictionaryAttr::get(ctx, updatedAttrs));
      }
    });

    // Step 3: Rename function definitions *after* updating uses
    module.walk([&](func::FuncOp func) {
      StringRef oldName = func.getSymName();
      auto it = renameMap.find(oldName);
      if (it != renameMap.end()) {
        symbolTable.setSymbolName(func, it->second);
      }
    });
  }

  // Process LLVM dialect (post-lowering from mlir-translate or ClangIR lowering)
  if (hasLLVMDialect) {
    llvm::StringMap<std::string> renameMap;

    // Step 1: Collect rename map for LLVM func definitions
    module.walk([&](LLVM::LLVMFuncOp func) {
      StringRef oldName = func.getSymName();

      // Skip main function to preserve entry point
      if (oldName == "main") {
        return;
      }

      // Skip LLVM intrinsics
      if (oldName.starts_with("llvm.")) {
        return;
      }

      // Only assign a new name once per function
      if (renameMap.find(oldName) == renameMap.end()) {
        std::string newName = generateHashedName(
            oldName.str(),
            algorithm,
            salt,
            hashLength
        );
        renameMap[oldName] = newName;
      }
    });

    // Step 2: Update symbol references
    module.walk([&](Operation *op) {
      SmallVector<NamedAttribute> updatedAttrs;
      bool changed = false;

      for (auto &attr : op->getAttrs()) {
        if (auto symAttr = llvm::dyn_cast<SymbolRefAttr>(attr.getValue())) {
          StringRef old = symAttr.getRootReference();
          auto it = renameMap.find(old);
          if (it != renameMap.end()) {
            auto newRef = SymbolRefAttr::get(ctx, it->second);
            updatedAttrs.emplace_back(attr.getName(), newRef);
            changed = true;
            continue;
          }
        }
        updatedAttrs.push_back(attr);
      }

      if (changed) {
        op->setAttrs(DictionaryAttr::get(ctx, updatedAttrs));
      }
    });

    // Step 3: Rename LLVM function definitions
    module.walk([&](LLVM::LLVMFuncOp func) {
      StringRef oldName = func.getSymName();
      auto it = renameMap.find(oldName);
      if (it != renameMap.end()) {
        symbolTable.setSymbolName(func, it->second);
      }
    });
  }
}

std::unique_ptr<Pass> mlir::obs::createCryptoHashPass(
    CryptoHashPass::HashAlgorithm algo,
    llvm::StringRef salt,
    unsigned hashLength) {
  return std::make_unique<CryptoHashPass>(algo, salt.str(), hashLength);
}
