add_library(MLIRObfuscation SHARED
  Passes.cpp
  PassRegistrations.cpp
  SymbolPass.cpp
)

target_link_libraries(MLIRObfuscation
  PUBLIC
    MLIRIR
    MLIRPass
    MLIRTransforms
    MLIRSupport
)
