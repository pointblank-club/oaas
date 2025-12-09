# CMakeLists.txt - CIR Transforms Library

add_mlir_library(MLIRCIRTransforms
  CIRAddressObfuscationPass.cpp
  Passes.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/CIR

  DEPENDS
  MLIRCIRPassIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRFuncDialect
  MLIRIR
  MLIRPass
  MLIRSupport
  MLIRTransforms
  MLIRTransformUtils
)
