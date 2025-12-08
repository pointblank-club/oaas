# Build MLIR Obfuscation plugin as a shared module
# This plugin is loaded by mlir-opt via --load-pass-plugin

add_library(MLIRObfuscation SHARED
  Passes.cpp
  PassRegistrations.cpp
  SymbolPass.cpp
  CryptoHashPass.cpp
  ConstantObfuscationPass.cpp
  SCFPass.cpp
  ImportObfuscationPass.cpp
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

# Link against MLIR/LLVM shared libraries and OpenSSL
target_link_libraries(MLIRObfuscation
  PRIVATE
    MLIR
    LLVM
    OpenSSL::Crypto
)

# Add LLVM/MLIR compile definitions
target_compile_definitions(MLIRObfuscation PRIVATE ${LLVM_DEFINITIONS})

# CRITICAL: Match LLVM build settings - no RTTI, no exceptions
# LLVM was built with LLVM_ENABLE_RTTI=OFF and LLVM_ENABLE_EH=OFF
target_compile_options(MLIRObfuscation PRIVATE -fno-rtti -fno-exceptions)
