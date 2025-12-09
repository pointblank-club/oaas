//===- Layer1.5_Examples.mlir - Address Obfuscation Examples --------------===//
//
// This file demonstrates the BEFORE and AFTER transformations applied by
// Layer 1.5 (Address Obfuscation Pass) and CIR-to-Func lowering.
//
//===----------------------------------------------------------------------===//

// =============================================================================
// Example 1: Simple Array Load
// =============================================================================

// BEFORE Layer 1.5 (Original CIR):
// ---------------------------------
cir.func @array_load(%arr: !cir.ptr<i32>, %idx: index) -> i32 {
  // Direct load without obfuscation
  %val = cir.load %arr[%idx] : !cir.ptr<i32>
  cir.return %val : i32
}

// AFTER Layer 1.5 (CIR with Address Obfuscation):
// ------------------------------------------------
cir.func @array_load(%arr: !cir.ptr<i32>, %idx: index) -> i32 {
  // Key constant (generated at compile-time, example value shown)
  %key = arith.constant 0x9E3779B97F4A7C15 : index

  // XOR obfuscation applied to index
  %masked_idx = arith.xori %idx, %key : index

  // Load using obfuscated index
  %val = cir.load %arr[%masked_idx] : !cir.ptr<i32>
  cir.return %val : i32
}

// AFTER CIR-to-Func Lowering (Final func dialect):
// -------------------------------------------------
func.func @array_load(%arr: memref<?xi32>, %idx: index) -> i32 {
  // Key constant remains
  %key = arith.constant 0x9E3779B97F4A7C15 : index

  // XOR obfuscation
  %masked_idx = arith.xori %idx, %key : index

  // CIR load converted to memref load
  %val = memref.load %arr[%masked_idx] : memref<?xi32>
  return %val : i32
}

// =============================================================================
// Example 2: Array Store
// =============================================================================

// BEFORE Layer 1.5:
// ---------------------------------
cir.func @array_store(%arr: !cir.ptr<i32>, %idx: index, %val: i32) {
  cir.store %val, %arr[%idx] : !cir.ptr<i32>
  cir.return
}

// AFTER Layer 1.5:
// ------------------------------------------------
cir.func @array_store(%arr: !cir.ptr<i32>, %idx: index, %val: i32) {
  %key = arith.constant 0x9E3779B97F4A7C15 : index
  %masked_idx = arith.xori %idx, %key : index
  cir.store %val, %arr[%masked_idx] : !cir.ptr<i32>
  cir.return
}

// AFTER CIR-to-Func Lowering:
// -------------------------------------------------
func.func @array_store(%arr: memref<?xi32>, %idx: index, %val: i32) {
  %key = arith.constant 0x9E3779B97F4A7C15 : index
  %masked_idx = arith.xori %idx, %key : index
  memref.store %val, %arr[%masked_idx] : memref<?xi32>
  return
}

// =============================================================================
// Example 3: Pointer Arithmetic
// =============================================================================

// BEFORE Layer 1.5:
// ---------------------------------
cir.func @ptr_arithmetic(%base: !cir.ptr<i32>, %offset: index) -> !cir.ptr<i32> {
  %new_ptr = cir.ptr_add %base, %offset : !cir.ptr<i32>
  cir.return %new_ptr : !cir.ptr<i32>
}

// AFTER Layer 1.5:
// ------------------------------------------------
cir.func @ptr_arithmetic(%base: !cir.ptr<i32>, %offset: index) -> !cir.ptr<i32> {
  %key = arith.constant 0x9E3779B97F4A7C15 : index
  %masked_offset = arith.xori %offset, %key : index
  %new_ptr = cir.ptr_add %base, %masked_offset : !cir.ptr<i32>
  cir.return %new_ptr : !cir.ptr<i32>
}

// AFTER CIR-to-Func Lowering:
// -------------------------------------------------
func.func @ptr_arithmetic(%base: memref<?xi32>, %offset: index) -> memref<?xi32> {
  %key = arith.constant 0x9E3779B97F4A7C15 : index
  %masked_offset = arith.xori %offset, %key : index

  // ptr_add converted to subview
  %c0 = arith.constant 0 : index
  %c_1 = arith.constant -1 : index
  %c1 = arith.constant 1 : index
  %new_memref = memref.subview %base[%masked_offset][%c_1][%c1]
    : memref<?xi32> to memref<?xi32, strided<[1], offset: ?>>

  return %new_memref : memref<?xi32, strided<[1], offset: ?>>
}

// =============================================================================
// Example 4: GetElementPtr (GEP) - Struct/Array Access
// =============================================================================

// BEFORE Layer 1.5:
// ---------------------------------
cir.func @gep_access(%base: !cir.ptr<i32>, %idx1: index, %idx2: index) -> i32 {
  %ptr = cir.gep %base, %idx1, %idx2 : !cir.ptr<i32>
  %val = cir.load %ptr[0] : !cir.ptr<i32>
  cir.return %val : i32
}

// AFTER Layer 1.5:
// ------------------------------------------------
cir.func @gep_access(%base: !cir.ptr<i32>, %idx1: index, %idx2: index) -> i32 {
  %key = arith.constant 0x9E3779B97F4A7C15 : index

  // Both GEP indices are obfuscated
  %masked_idx1 = arith.xori %idx1, %key : index
  %masked_idx2 = arith.xori %idx2, %key : index
  %ptr = cir.gep %base, %masked_idx1, %masked_idx2 : !cir.ptr<i32>

  // Load index also obfuscated
  %zero = arith.constant 0 : index
  %masked_zero = arith.xori %zero, %key : index
  %val = cir.load %ptr[%masked_zero] : !cir.ptr<i32>

  cir.return %val : i32
}

// AFTER CIR-to-Func Lowering:
// -------------------------------------------------
func.func @gep_access(%base: memref<?xi32>, %idx1: index, %idx2: index) -> i32 {
  %key = arith.constant 0x9E3779B97F4A7C15 : index

  // Obfuscated indices
  %masked_idx1 = arith.xori %idx1, %key : index
  %masked_idx2 = arith.xori %idx2, %key : index

  // GEP converted to linearized offset computation
  %linear_offset = arith.addi %masked_idx1, %masked_idx2 : index

  // Create subview with computed offset
  %c_1 = arith.constant -1 : index
  %c1 = arith.constant 1 : index
  %ptr_memref = memref.subview %base[%linear_offset][%c_1][%c1]
    : memref<?xi32> to memref<?xi32, strided<[1], offset: ?>>

  // Load with obfuscated zero index
  %zero = arith.constant 0 : index
  %masked_zero = arith.xori %zero, %key : index
  %val = memref.load %ptr_memref[%masked_zero] : memref<?xi32, strided<[1], offset: ?>>

  return %val : i32
}

// =============================================================================
// Example 5: Complete C Function Transformation
// =============================================================================

// Original C Code:
// ----------------
// int sum_array(int* arr, int size) {
//     int sum = 0;
//     for (int i = 0; i < size; i++) {
//         sum += arr[i];
//     }
//     return sum;
// }

// BEFORE Layer 1.5 (CIR from ClangIR):
// ------------------------------------
cir.func @sum_array(%arr: !cir.ptr<i32>, %size: i32) -> i32 {
  %sum = cir.alloca i32
  %i = cir.alloca i32

  %zero = arith.constant 0 : i32
  cir.store %zero, %sum : !cir.ptr<i32>
  cir.store %zero, %i : !cir.ptr<i32>

  cir.br ^loop_cond

^loop_cond:
  %i_val = cir.load %i : !cir.ptr<i32>
  %cmp = arith.cmpi slt, %i_val, %size : i32
  cir.cond_br %cmp, ^loop_body, ^loop_exit

^loop_body:
  %i_idx = cir.load %i : !cir.ptr<i32>
  %idx = arith.index_cast %i_idx : i32 to index
  %arr_val = cir.load %arr[%idx] : !cir.ptr<i32>

  %sum_val = cir.load %sum : !cir.ptr<i32>
  %new_sum = arith.addi %sum_val, %arr_val : i32
  cir.store %new_sum, %sum : !cir.ptr<i32>

  %one = arith.constant 1 : i32
  %i_next = arith.addi %i_val, %one : i32
  cir.store %i_next, %i : !cir.ptr<i32>
  cir.br ^loop_cond

^loop_exit:
  %result = cir.load %sum : !cir.ptr<i32>
  cir.return %result : i32
}

// AFTER Layer 1.5 (Address Obfuscation Applied):
// -----------------------------------------------
cir.func @sum_array(%arr: !cir.ptr<i32>, %size: i32) -> i32 {
  %key = arith.constant 0x9E3779B97F4A7C15 : index  // Key generated once

  %sum = cir.alloca i32
  %i = cir.alloca i32

  %zero = arith.constant 0 : i32

  // Local variable stores remain unobfuscated (optional enhancement)
  cir.store %zero, %sum : !cir.ptr<i32>
  cir.store %zero, %i : !cir.ptr<i32>

  cir.br ^loop_cond

^loop_cond:
  %i_val = cir.load %i : !cir.ptr<i32>
  %cmp = arith.cmpi slt, %i_val, %size : i32
  cir.cond_br %cmp, ^loop_body, ^loop_exit

^loop_body:
  %i_idx = cir.load %i : !cir.ptr<i32>
  %idx = arith.index_cast %i_idx : i32 to index

  // CRITICAL: Array access is obfuscated
  %masked_idx = arith.xori %idx, %key : index
  %arr_val = cir.load %arr[%masked_idx] : !cir.ptr<i32>

  %sum_val = cir.load %sum : !cir.ptr<i32>
  %new_sum = arith.addi %sum_val, %arr_val : i32
  cir.store %new_sum, %sum : !cir.ptr<i32>

  %one = arith.constant 1 : i32
  %i_next = arith.addi %i_val, %one : i32
  cir.store %i_next, %i : !cir.ptr<i32>
  cir.br ^loop_cond

^loop_exit:
  %result = cir.load %sum : !cir.ptr<i32>
  cir.return %result : i32
}

// AFTER CIR-to-Func Lowering (Final):
// ------------------------------------
func.func @sum_array(%arr: memref<?xi32>, %size: i32) -> i32 {
  %key = arith.constant 0x9E3779B97F4A7C15 : index

  %sum = memref.alloca() : memref<i32>
  %i = memref.alloca() : memref<i32>

  %zero = arith.constant 0 : i32
  memref.store %zero, %sum[] : memref<i32>
  memref.store %zero, %i[] : memref<i32>

  cf.br ^loop_cond

^loop_cond:
  %i_val = memref.load %i[] : memref<i32>
  %cmp = arith.cmpi slt, %i_val, %size : i32
  cf.cond_br %cmp, ^loop_body, ^loop_exit

^loop_body:
  %i_idx = memref.load %i[] : memref<i32>
  %idx = arith.index_cast %i_idx : i32 to index

  // Obfuscated array access
  %masked_idx = arith.xori %idx, %key : index
  %arr_val = memref.load %arr[%masked_idx] : memref<?xi32>

  %sum_val = memref.load %sum[] : memref<i32>
  %new_sum = arith.addi %sum_val, %arr_val : i32
  memref.store %new_sum, %sum[] : memref<i32>

  %one = arith.constant 1 : i32
  %i_next = arith.addi %i_val, %one : i32
  memref.store %i_next, %i[] : memref<i32>
  cf.br ^loop_cond

^loop_exit:
  %result = memref.load %sum[] : memref<i32>
  return %result : i32
}

// =============================================================================
// Pipeline Command Examples
// =============================================================================

// Full Layer 1.5 pipeline with all transformations:
// mlir-opt input.cir \
//   --cir-address-obf \
//   --convert-cir-to-func \
//   --canonicalize \
//   --convert-func-to-llvm \
//   -o output.ll

// Layer 1.5 disabled (no-op pass):
// mlir-opt input.cir \
//   --cir-address-obf=enable=false \
//   --convert-cir-to-func \
//   --canonicalize \
//   --convert-func-to-llvm \
//   -o output.ll

// Verify transformations at each stage:
// mlir-opt input.cir --cir-address-obf -o stage1.mlir
// mlir-opt stage1.mlir --convert-cir-to-func -o stage2.mlir
// mlir-opt stage2.mlir --canonicalize --convert-func-to-llvm -o output.ll
