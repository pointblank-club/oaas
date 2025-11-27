set(SOURCES
  Passes.cpp
  PassRegistrations.cpp
  SymbolPass.cpp
)

add_library(MLIRObfuscation SHARED
  ${SOURCES}
)

# Export all symbols for plugin loading
set_target_properties(MLIRObfuscation PROPERTIES
  CXX_VISIBILITY_PRESET default
  VISIBILITY_INLINES_HIDDEN OFF
)

target_include_directories(MLIRObfuscation
  PRIVATE
    ${MLIR_INCLUDE_DIRS}
    ${LLVM_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
)

target_link_libraries(MLIRObfuscation
  PRIVATE
    MLIRIR
    MLIRPass
    MLIRFuncDialect
    MLIRTransforms
    MLIRParser
    MLIRSupport
    MLIRDialectUtils
)

mlir_check_all_link_libraries(MLIRObfuscation)
