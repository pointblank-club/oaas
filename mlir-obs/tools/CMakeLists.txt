# Standalone mlir-obfuscate tool
# Works with both Polygeist MLIR and traditional LLVM-dialect MLIR

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

add_llvm_executable(mlir-obfuscate
  mlir-obfuscate.cpp

  DEPENDS
  MLIRObfuscation
)

target_link_libraries(mlir-obfuscate
  PRIVATE
    ${dialect_libs}
    ${conversion_libs}
    MLIRIR
    MLIRPass
    MLIRFuncDialect
    MLIRLLVMDialect
    MLIRSCFDialect
    MLIRArithDialect
    MLIRMemRefDialect
    MLIRAffineDialect
    MLIRTransforms
    MLIRSupport
    MLIROptLib
    MLIRParser
    MLIRTranslateLib
)

target_include_directories(mlir-obfuscate
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_CURRENT_BINARY_DIR}/../include
)

# Install the tool
install(TARGETS mlir-obfuscate DESTINATION bin)
