add_llvm_library(LLVMObfuscationPlugin MODULE
  PluginRegistration.cpp
  ../BlockReordering.cpp
  ../BogusControlFlow.cpp
  ../ConstantObfuscation.cpp
  ../CryptoUtils.cpp
  ../Flattening.cpp
  ../IndirectBranch.cpp
  ../IndirectCall.cpp
  ../LinearMBA.cpp
  ../NopInsertion.cpp
  ../SplitBasicBlocks.cpp
  ../Substitution.cpp
  ../Utils.cpp
  ../../Utils/LowerSwitch.cpp

  PLUGIN_TOOL opt

  DEPENDS
  intrinsics_gen
  )

target_include_directories(LLVMObfuscationPlugin PRIVATE
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/Transforms
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/Transforms/Obfuscation
  )
