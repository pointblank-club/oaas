//===- Passes.h - CIR Dialect Pass Registration --------------------------===//
//
// Part of the LLVM Project - CIR Dialect Passes
//
// This header declares all passes for the CIR dialect, including Layer 1.5
// address obfuscation and CIR-to-Func lowering.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CIR_PASSES_H
#define MLIR_CIR_PASSES_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace cir {

//===----------------------------------------------------------------------===//
// Pass Registration
//===----------------------------------------------------------------------===//

/// Create a pass to perform address-level obfuscation on CIR operations
///
/// This pass implements Layer 1.5 obfuscation by XORing pointer indices
/// with a compile-time generated secret key.
///
/// @param enabled Whether obfuscation is enabled (frontend toggle)
/// @return Unique pointer to the created pass
std::unique_ptr<Pass> createCIRAddressObfuscationPass(bool enabled = true);

/// Create a pass to convert CIR dialect to Func dialect
///
/// This pass lowers CIR operations (cir.load, cir.store, etc.) to
/// standard func dialect operations (memref.load, memref.store, etc.)
/// using MLIR's dialect conversion framework.
///
/// @return Unique pointer to the created pass
std::unique_ptr<Pass> createConvertCIRToFuncPass();

//===----------------------------------------------------------------------===//
// Pass Registration Functions
//===----------------------------------------------------------------------===//

/// Register all CIR passes with the MLIR pass registry
///
/// Call this function to make CIR passes available in mlir-opt and
/// other MLIR tools.
void registerCIRPasses();

/// Generate pass registration code (for mlir-opt integration)
#define GEN_PASS_REGISTRATION
#include "CIR/Passes.h.inc"

} // namespace cir
} // namespace mlir

#endif // MLIR_CIR_PASSES_H
