# CMakeLists.txt - CIR Dialect Library Build Configuration

# Add subdirectories
add_subdirectory(Transforms)
add_subdirectory(Conversion)

# CIR Dialect Library
add_mlir_dialect_library(MLIRCIR

  DEPENDS
  MLIRCIRPassIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRFuncDialect
  MLIRMemRefDialect
  MLIRControlFlowDialect
  MLIRIR
  MLIRPass
  MLIRSupport
  MLIRTransforms
)

# Generate pass header from TableGen
set(LLVM_TARGET_DEFINITIONS ../../include/CIR/Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name CIR)
add_public_tablegen_target(MLIRCIRPassIncGen)
