# Use the same pattern as LLVM's standalone plugin example
add_llvm_library(MLIRObfuscation
  MODULE
  Passes.cpp
  PassRegistrations.cpp
  SymbolPass.cpp
  CryptoHashPass.cpp
  ConstantObfuscationPass.cpp

  PLUGIN_TOOL
  mlir-opt

  LINK_LIBS
  MLIRIR
  MLIRPass
  MLIRFuncDialect
  MLIRLLVMDialect
  MLIRTransforms
  MLIRSupport
  MLIROptLib
  OpenSSL::Crypto
)

target_include_directories(MLIRObfuscation
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
)
