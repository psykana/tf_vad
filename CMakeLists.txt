cmake_minimum_required(VERSION 3.16)
project(inference C CXX)

set(CMAKE_CXX_STANDARD 11)

set(TENSORFLOW_SOURCE_DIR "" CACHE PATH
  "Directory that contains the TensorFlow project"
)

file(GLOB model_dirs "${CMAKE_CURRENT_LIST_DIR}/models/*")
list(LENGTH model_dirs list_length)
if(list_length LESS_EQUAL 1)
  message( FATAL_ERROR "Models directory is empty" )
endif()
list(SORT model_dirs COMPARE NATURAL ORDER DESCENDING)
list(GET model_dirs 0 latest_model_dir)
if(NOT EXISTS "${latest_model_dir}/model.tflite")
  message( FATAL_ERROR "Could not find a .tflite model in ${latest_model_dir}" )
endif()

execute_process(
  WORKING_DIRECTORY ${latest_model_dir}
  COMMAND xxd -i model.tflite ${CMAKE_BINARY_DIR}/model.cc
)

if(NOT TENSORFLOW_SOURCE_DIR)
  get_filename_component(TENSORFLOW_SOURCE_DIR
    "${CMAKE_CURRENT_LIST_DIR}/inference_src/tensorflow"
    ABSOLUTE
  )
endif()

add_subdirectory(
  "${TENSORFLOW_SOURCE_DIR}/tensorflow/lite"
  "${CMAKE_CURRENT_BINARY_DIR}/tensorflow-lite"
  EXCLUDE_FROM_ALL
)

find_package(PkgConfig REQUIRED)
pkg_search_module(FFTW REQUIRED fftw3 IMPORTED_TARGET)
include_directories(PkgConfig::FFTW)

add_executable(inference
  inference_src/inference.cpp
  ${CMAKE_BINARY_DIR}/model.cc
)

target_link_libraries(inference
  tensorflow-lite
  PkgConfig::FFTW
  m
)