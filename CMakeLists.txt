cmake_minimum_required(VERSION 3.16)
project(inference LANGUAGES C CXX VERSION 1.0)

set(CMAKE_CXX_STANDARD 14)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

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
if(NOT EXISTS "${latest_model_dir}/model.h")
  message( FATAL_ERROR "Could not find a C header model in ${latest_model_dir}" )
endif()

execute_process(
  WORKING_DIRECTORY ${latest_model_dir}
  COMMAND xxd -i model.tflite ${CMAKE_CURRENT_LIST_DIR}/inference_src/model.cc
  COMMAND cp model.tflite ${CMAKE_BINARY_DIR}
  COMMAND cp model.h ${CMAKE_CURRENT_LIST_DIR}/inference_src/
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

add_library(inferenceLIB SHARED
  inference_src/inference_lib.cpp
  inference_src/model.h
  inference_src/simd_math.h
)

set_target_properties(inferenceLIB PROPERTIES VERSION ${PROJECT_VERSION})

set_target_properties(inferenceLIB PROPERTIES PUBLIC_HEADER api.h)

add_executable(testapp
  inference_src/testApp.cpp
  inference_src/api.h
  inference_src/AudioFile/AudioFile.h
)

target_link_libraries(testapp
  inferenceLIB
  fftw3f
  m
)

add_executable(inference_tflite
  inference_src/inference_tflite.cpp
  inference_src/model.cc
)

target_link_libraries(inference_tflite
  tensorflow-lite
  fftw3f
  m
)

add_executable(inference_naive
  inference_src/inference_naive.cpp
  inference_src/model.h
  inference_src/simd_math.h
)

target_link_libraries(inference_naive
  fftw3f
  m
)
