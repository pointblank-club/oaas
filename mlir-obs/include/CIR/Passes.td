//===- Passes.td - CIR Dialect Pass Definitions ---------------------------===//
//
// Part of the LLVM Project - CIR Dialect Pass TableGen Definitions
//
// This file defines all CIR dialect passes using MLIR's TableGen framework.
// This generates the pass registration code automatically.
//
//===----------------------------------------------------------------------===//

#ifndef CIR_PASSES
#define CIR_PASSES

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// CIRAddressObfuscationPass
//===----------------------------------------------------------------------===//

def CIRAddressObfuscationPass : Pass<"cir-address-obf", "ModuleOp"> {
  let summary = "Apply address-level obfuscation to CIR pointer operations";

  let description = [{
    This pass implements Layer 1.5 obfuscation by applying XOR-based masking
    to pointer accesses in CIR dialect operations. It transforms:

    ```mlir
    %val = cir.load %ptr[%idx]
    ```

    Into:

    ```mlir
    %key = arith.constant <obfuscation_key>
    %masked_idx = arith.xori %idx, %key
    %val = cir.load %ptr[%masked_idx]
    ```

    The obfuscation key is generated at compile-time using a hash of
    __TIME__ and __DATE__ macros combined with magic constants, ensuring
    each compilation produces a unique key.

    When disabled (via frontend toggle), this pass acts as a no-op.

    Affected operations:
    - cir.load: Obfuscates load indices
    - cir.store: Obfuscates store indices
    - cir.ptr_add: Obfuscates pointer arithmetic offsets
    - cir.gep: Obfuscates GetElementPtr indices
  }];

  let constructor = "mlir::cir::createCIRAddressObfuscationPass()";

  let dependentDialects = ["mlir::arith::ArithDialect"];

  let options = [
    Option<"obfuscationEnabled", "enable", "bool", "true",
           "Enable address obfuscation (controlled by frontend toggle)">
  ];
}

//===----------------------------------------------------------------------===//
// ConvertCIRToFuncPass
//===----------------------------------------------------------------------===//

def ConvertCIRToFuncPass : Pass<"convert-cir-to-func", "ModuleOp"> {
  let summary = "Convert CIR dialect to Func dialect";

  let description = [{
    This pass lowers CIR (ClangIR) dialect operations to the standard func
    dialect, enabling further lowering to LLVM IR through the existing
    compilation pipeline.

    Uses MLIR's Dialect Conversion framework with:
    - **TypeConverter**: Maps CIR types (e.g., cir.ptr<T>) to standard types
      (e.g., memref<?xT>)
    - **ConversionPatterns**: Defines operation-by-operation rewrite rules
    - **ConversionTarget**: Marks CIR as illegal and func/arith/memref as legal

    Example transformations:

    ```mlir
    // Before (CIR):
    cir.func @foo(%ptr: !cir.ptr<i32>) -> i32 {
      %idx = arith.constant 5 : index
      %val = cir.load %ptr[%idx]
      cir.return %val
    }

    // After (Func):
    func.func @foo(%ptr: memref<?xi32>) -> i32 {
      %idx = arith.constant 5 : index
      %val = memref.load %ptr[%idx] : memref<?xi32>
      return %val : i32
    }
    ```

    Converted operations:
    - cir.func → func.func
    - cir.load → memref.load
    - cir.store → memref.store
    - cir.ptr_add → memref.subview
    - cir.gep → memref.subview or arithmetic computation
    - cir.return → func.return
  }];

  let constructor = "mlir::cir::createConvertCIRToFuncPass()";

  let dependentDialects = [
    "mlir::func::FuncDialect",
    "mlir::arith::ArithDialect",
    "mlir::cf::ControlFlowDialect",
    "mlir::memref::MemRefDialect"
  ];
}

#endif // CIR_PASSES
