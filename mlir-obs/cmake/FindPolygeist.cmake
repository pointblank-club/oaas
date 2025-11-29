# FindPolygeist.cmake
# Locates the Polygeist installation and sets up necessary variables
#
# Variables set:
#   POLYGEIST_FOUND          - True if Polygeist is found
#   POLYGEIST_INCLUDE_DIRS   - Include directories for Polygeist
#   POLYGEIST_LIBRARY_DIRS   - Library directories for Polygeist
#   POLYGEIST_LIBRARIES      - Libraries to link against
#   CGEIST_EXECUTABLE        - Path to the cgeist binary
#   MLIR_CLANG_EXECUTABLE    - Path to mlir-clang binary (alternative name)

# Search paths for Polygeist
set(POLYGEIST_SEARCH_PATHS
    ${POLYGEIST_DIR}
    $ENV{POLYGEIST_DIR}
    ${CMAKE_PREFIX_PATH}
    /usr
    /usr/local
    ${HOME}/polygeist
    ${HOME}/polygeist/build
    ${HOME}/llvm-project/build
)

# Find cgeist executable (main Polygeist C/C++ frontend)
find_program(CGEIST_EXECUTABLE
    NAMES cgeist mlir-clang
    PATHS ${POLYGEIST_SEARCH_PATHS}
    PATH_SUFFIXES bin polygeist/bin
    DOC "Path to cgeist (Polygeist C/C++ to MLIR compiler)"
)

# Find Polygeist include directories
find_path(POLYGEIST_INCLUDE_DIR
    NAMES polygeist/Passes/Passes.h
    PATHS ${POLYGEIST_SEARCH_PATHS}
    PATH_SUFFIXES include polygeist/include
    DOC "Polygeist include directory"
)

# Find Polygeist library directories
find_path(POLYGEIST_LIBRARY_DIR
    NAMES libPolygeistPasses.a libPolygeistPasses.so PolygeistPasses.lib
    PATHS ${POLYGEIST_SEARCH_PATHS}
    PATH_SUFFIXES lib polygeist/lib lib64
    DOC "Polygeist library directory"
)

# Find Polygeist libraries
find_library(POLYGEIST_PASSES_LIB
    NAMES PolygeistPasses
    PATHS ${POLYGEIST_LIBRARY_DIR}
    NO_DEFAULT_PATH
)

find_library(POLYGEIST_MLIRSCF_LIB
    NAMES PolygeistMLIRSCF
    PATHS ${POLYGEIST_LIBRARY_DIR}
    NO_DEFAULT_PATH
)

# Handle the QUIETLY and REQUIRED arguments
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Polygeist
    REQUIRED_VARS CGEIST_EXECUTABLE
    FOUND_VAR POLYGEIST_FOUND
    VERSION_VAR POLYGEIST_VERSION
)

if(POLYGEIST_FOUND)
    set(POLYGEIST_INCLUDE_DIRS ${POLYGEIST_INCLUDE_DIR})
    set(POLYGEIST_LIBRARY_DIRS ${POLYGEIST_LIBRARY_DIR})

    # Set libraries if found
    set(POLYGEIST_LIBRARIES "")
    if(POLYGEIST_PASSES_LIB)
        list(APPEND POLYGEIST_LIBRARIES ${POLYGEIST_PASSES_LIB})
    endif()
    if(POLYGEIST_MLIRSCF_LIB)
        list(APPEND POLYGEIST_LIBRARIES ${POLYGEIST_MLIRSCF_LIB})
    endif()

    message(STATUS "Found Polygeist: ${CGEIST_EXECUTABLE}")
    if(POLYGEIST_INCLUDE_DIRS)
        message(STATUS "  Include dirs: ${POLYGEIST_INCLUDE_DIRS}")
    endif()
    if(POLYGEIST_LIBRARIES)
        message(STATUS "  Libraries: ${POLYGEIST_LIBRARIES}")
    endif()
else()
    message(STATUS "Polygeist not found - will use fallback C/C++ compilation")
    message(STATUS "To enable Polygeist support, install it and set POLYGEIST_DIR")
endif()

mark_as_advanced(
    POLYGEIST_INCLUDE_DIR
    POLYGEIST_LIBRARY_DIR
    POLYGEIST_PASSES_LIB
    POLYGEIST_MLIRSCF_LIB
    CGEIST_EXECUTABLE
)
