add_executable(mlir-obfuscate
  mlir-obfuscate.cpp
)

target_link_libraries(mlir-obfuscate
  PRIVATE
    MLIRObfuscation
    MLIRIR
    MLIRPass
    MLIRSupport
    MLIROptLib
)

target_include_directories(mlir-obfuscate
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
)
