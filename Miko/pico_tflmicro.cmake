# Wrapper to include pico-tflmicro without re-initializing the Pico SDK
# This extracts just the library definition from pico-tflmicro

set(TFLM_DIR ${CMAKE_CURRENT_LIST_DIR}/../pico-tflmicro)

add_library(pico-tflmicro STATIC)

target_include_directories(pico-tflmicro
  PUBLIC
  ${TFLM_DIR}/src/
  ${TFLM_DIR}/src/third_party/ruy
  ${TFLM_DIR}/src/third_party/gemmlowp
  ${TFLM_DIR}/src/third_party/kissfft
  ${TFLM_DIR}/src/third_party/flatbuffers
  ${TFLM_DIR}/src/third_party/cmsis/CMSIS/Core/Include
  ${TFLM_DIR}/src/third_party/flatbuffers/include
  ${TFLM_DIR}/src/third_party/cmsis_nn/Include
)

target_compile_definitions(
  pico-tflmicro
  PUBLIC
  TF_LITE_DISABLE_X86_NEON=1
  TF_LITE_STATIC_MEMORY=1
  TF_LITE_USE_CTIME=1
  CMSIS_NN=1
  ARDUINO=1
  TFLITE_USE_CTIME=1
)

set_target_properties(
  pico-tflmicro
  PROPERTIES
  COMPILE_FLAGS "-Os -fno-rtti -fno-exceptions -fno-threadsafe-statics"
)

target_link_libraries(
  pico-tflmicro
  pico_stdlib
  pico_multicore
)

# Include all source files from pico-tflmicro
file(GLOB_RECURSE TFLM_SOURCES
  "${TFLM_DIR}/src/signal/*.cpp"
  "${TFLM_DIR}/src/tensorflow/*.cpp"
  "${TFLM_DIR}/src/tensorflow/*.cc"
)

# Include CMSIS-NN source files
file(GLOB_RECURSE CMSIS_NN_SOURCES
  "${TFLM_DIR}/src/third_party/cmsis_nn/Source/*.c"
)

target_sources(pico-tflmicro PRIVATE ${TFLM_SOURCES} ${CMSIS_NN_SOURCES})
