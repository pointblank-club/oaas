cmake_minimum_required(VERSION 3.13)
project(MLIROBF LANGUAGES CXX C)

# Use C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Add cmake module path for custom Find*.cmake files
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Locate MLIR & LLVM packages (REQUIRED - core functionality)
find_package(MLIR REQUIRED CONFIG)
find_package(LLVM REQUIRED CONFIG)

# Locate OpenSSL for cryptographic hashing (SHA256, BLAKE2B)
find_package(OpenSSL REQUIRED)

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(AddMLIR)
include(AddLLVM)
include(LLVMConfig)

# ClangIR is natively supported in LLVM 22.0.0
# No additional package finding required - it's part of LLVM/Clang
message(STATUS "✓ Using LLVM ${LLVM_VERSION} with native ClangIR support")
message(STATUS "  ClangIR frontend: clang -emit-cir or clangir command")
message(STATUS "  Default frontend: clang → LLVM IR → mlir-translate")

# Configure header with feature flags
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/include/Obfuscator/Config.h.in"
    "${CMAKE_CURRENT_BINARY_DIR}/include/Obfuscator/Config.h"
)

# Add include path for headers
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)  # For Config.h
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})

# Build library from sources in lib/
add_subdirectory(lib)

# NOTE: mlir-obfuscate tool disabled - use mlir-opt --load-pass-plugin instead
# add_subdirectory(tools)

