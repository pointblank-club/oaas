add_library(MLIRObfuscation SHARED
  Passes.cpp
  PassRegistrations.cpp
  SymbolPass.cpp
  CryptoHashPass.cpp
  ConstantObfuscationPass.cpp
  SCFPass.cpp
  ImportObfuscationPass.cpp
)

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

target_link_libraries(MLIRObfuscation
  PRIVATE
    MLIR
    LLVM
    OpenSSL::Crypto
)

target_compile_definitions(MLIRObfuscation PRIVATE ${LLVM_DEFINITIONS})

target_compile_options(MLIRObfuscation PRIVATE -fno-rtti -fno-exceptions)
