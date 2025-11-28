cmake_minimum_required(VERSION 3.13)
project(MLIROBF LANGUAGES CXX C)

# Use C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Locate MLIR & LLVM packages
find_package(MLIR REQUIRED CONFIG)
find_package(LLVM REQUIRED CONFIG)

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(AddMLIR)
include(AddLLVM)
include(LLVMConfig)


# Add include path for headers
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})

# Build library from sources in lib/
add_subdirectory(lib)

# Don't build standalone tool for now - focus on plugin
# add_subdirectory(tools)

