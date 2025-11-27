add_llvm_library(MLIRObfuscation MODULE
  Passes.cpp
  PassRegistrations.cpp
  SymbolPass.cpp

  LLVM_EXPORTED_SYMBOL_FILE ${CMAKE_CURRENT_SOURCE_DIR}/ObfuscationPlugin.exports
)

target_include_directories(MLIRObfuscation
  PRIVATE
    ${MLIR_INCLUDE_DIRS}
    ${LLVM_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
)

target_link_libraries(MLIRObfuscation
  PRIVATE
    MLIRIR
    MLIRPass
    MLIRTransforms
    MLIRSupport
    MLIRParser
    MLIRDialectUtils
    MLIRFuncDialect
)

mlir_check_all_link_libraries(MLIRObfuscation)
