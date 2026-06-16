cmake_minimum_required(VERSION 3.16)
project(unet_infer CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Locate ONNX Runtime. Defaults to the Homebrew prefix on macOS; override with
#   cmake -DONNXRUNTIME_ROOT=/path/to/onnxruntime ..
if(NOT ONNXRUNTIME_ROOT)
  execute_process(COMMAND brew --prefix onnxruntime
                  OUTPUT_VARIABLE ONNXRUNTIME_ROOT
                  OUTPUT_STRIP_TRAILING_WHITESPACE
                  ERROR_QUIET)
endif()

find_path(ORT_INCLUDE_DIR onnxruntime_cxx_api.h
          HINTS ${ONNXRUNTIME_ROOT}/include
          PATH_SUFFIXES onnxruntime)
find_library(ORT_LIB onnxruntime HINTS ${ONNXRUNTIME_ROOT}/lib)

if(NOT ORT_INCLUDE_DIR OR NOT ORT_LIB)
  message(FATAL_ERROR
    "ONNX Runtime not found. Install it (brew install onnxruntime) or pass "
    "-DONNXRUNTIME_ROOT=/path/to/onnxruntime")
endif()

add_executable(unet_infer unet_infer.cpp)
target_include_directories(unet_infer PRIVATE ${ORT_INCLUDE_DIR})
target_link_libraries(unet_infer PRIVATE ${ORT_LIB})

message(STATUS "ONNX Runtime headers: ${ORT_INCLUDE_DIR}")
message(STATUS "ONNX Runtime library: ${ORT_LIB}")
