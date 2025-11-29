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

# Try to find Polygeist (OPTIONAL - enhanced frontend)
# This enables C/C++ -> high-level MLIR directly
find_package(Polygeist)

if(POLYGEIST_FOUND)
    message(STATUS "✓ Polygeist support ENABLED")
    message(STATUS "  You can now use: cgeist source.c -o source.mlir")
    set(HAVE_POLYGEIST 1)
else()
    message(STATUS "⚠ Polygeist support DISABLED (not found)")
    message(STATUS "  Falling back to: clang + mlir-translate workflow")
    set(HAVE_POLYGEIST 0)
endif()

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

# Add Polygeist includes if available
if(POLYGEIST_FOUND AND POLYGEIST_INCLUDE_DIRS)
    include_directories(${POLYGEIST_INCLUDE_DIRS})
endif()

# Build library from sources in lib/
add_subdirectory(lib)

# Build standalone tool (mlir-obfuscate)
add_subdirectory(tools)

