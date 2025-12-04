# Build MLIR Obfuscation plugin as a shared module
# This plugin is loaded by mlir-opt via --load-pass-plugin

add_library(MLIRObfuscation SHARED
  Passes.cpp
  PassRegistrations.cpp
  SymbolPass.cpp
  CryptoHashPass.cpp
  ConstantObfuscationPass.cpp
  SCFPass.cpp
)

# Set output name and properties
set_target_properties(MLIRObfuscation PROPERTIES
  PREFIX ""
  SUFFIX ".so"
)

target_include_directories(MLIRObfuscation
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_CURRENT_BINARY_DIR}/../include
    ${MLIR_INCLUDE_DIRS}
    ${LLVM_INCLUDE_DIRS}
)

# Link only against OpenSSL for crypto functions
# DO NOT link against MLIR/LLVM static libs to avoid duplicate symbol registration
target_link_libraries(MLIRObfuscation
  PRIVATE
    OpenSSL::Crypto
)

# Add LLVM/MLIR compile definitions
target_compile_definitions(MLIRObfuscation PRIVATE ${LLVM_DEFINITIONS})
