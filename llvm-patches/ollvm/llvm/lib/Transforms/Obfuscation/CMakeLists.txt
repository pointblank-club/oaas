add_llvm_component_library(LLVMObfuscation
  BlockReordering.cpp
  BogusControlFlow.cpp
  ConstantObfuscation.cpp
  CryptoUtils.cpp
  Flattening.cpp
  IndirectBranch.cpp
  IndirectCall.cpp
  LinearMBA.cpp
  NopInsertion.cpp
  SplitBasicBlocks.cpp
  Substitution.cpp
  Utils.cpp

  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/Transforms
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/Transforms/Obfuscation

  DEPENDS
  intrinsics_gen

  LINK_COMPONENTS
  Core
  Support
  TransformUtils
  )

add_subdirectory(Plugin)
