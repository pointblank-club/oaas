add_mlir_library(MLIRObfuscation
  Passes.cpp
  PassRegistrations.cpp
  SymbolPass.cpp

  LINK_LIBS PUBLIC
    MLIRIR
    MLIRPass
    MLIRTransforms
    MLIRSupport
)
