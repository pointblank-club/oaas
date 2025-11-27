add_library(MLIRObfuscation SHARED
  Passes.cpp
  PassRegistrations.cpp
  SymbolPass.cpp
)

target_include_directories(MLIRObfuscation
  PRIVATE
    ${MLIR_INCLUDE_DIRS}
    ${LLVM_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(MLIRObfuscation
  PRIVATE
    MLIRIR
    MLIRFuncDialect
    MLIRPass
    MLIRTransforms
    MLIRSupport
    MLIRParser
    MLIRDialectUtils
)

mlir_check_all_link_libraries(MLIRObfuscation)
