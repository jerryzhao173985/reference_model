cmake_minimum_required (VERSION 3.16)

# Copyright (c) 2024-2025, ARM Limited.
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

project(tosa_verify_compliance LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_CXX_COMPILER_ID STREQUAL GNU)
  set(CMAKE_CXX_FLAGS "-Wall -Wno-ignored-attributes -Wno-format-truncation")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL Clang)
  set(CMAKE_CXX_FLAGS "-Wall -Wno-ignored-attributes")
endif()

include(CMakePrintHelpers)



set(BASE_REF_DIR ${PROJECT_SOURCE_DIR}/../../..)

# If Serialization Library path is specified, look for library so it doesn't have to be built again.
# Otherwise, set the Serialization Library related paths to thirdparty directory.
if(SERIALIZATION_DIR)
  find_library(SERIALIZATION_LIB
               NAMES libtosa_serialization_lib.a tosa_serialization_lib
               NO_DEFAULT_PATH
               HINTS ${SERIALIZATION_DIR}
               PATH_SUFFIXES lib)

  if(NOT SERIALIZATION_LIB)
    message(FATAL_ERROR "TOSA Serialization Library location was specified but not found at: ${SERIALIZATION_LIB_DIR}")
  endif()
else()
  # Build from third party directory if not found.
  set(SERIALIZATION_LIB tosa_serialization_lib)
  set(SERIALIZATION_DIR ${BASE_REF_DIR}/thirdparty/serialization_lib)
endif()

# If Flatbuffers, Eigen, Half path isn't specified, set to thirdparty directory.
if(NOT FLATBUFFERS_DIR)
  set(FLATBUFFERS_DIR ${BASE_REF_DIR}/thirdparty/serialization_lib/third_party/flatbuffers/)
endif()

if(NOT EIGEN_DIR)
  set(EIGEN_DIR ${BASE_REF_DIR}/thirdparty/eigen/)
endif()

if(NOT HALF_DIR)
  set(HALF_DIR ${BASE_REF_DIR}/thirdparty/serialization_lib/third_party/half)
endif()

if(NOT JSON_DIR)
  set(JSON_DIR ${BASE_REF_DIR}/thirdparty/json)
endif()

include_directories(${BASE_REF_DIR}/reference_model/include)

set(PRIVATE_INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    ${FLATBUFFERS_DIR}/include
    ${EIGEN_DIR}
    ${EIGEN_DIR}/unsupported/
    ${SERIALIZATION_DIR}/include
    ${HALF_DIR}/include
    ${JSON_DIR}/single_include
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../generate
    ${CMAKE_CURRENT_SOURCE_DIR}/../../_deps/cxxopts-src/include
)

add_executable(tosa_verify
    ${BASE_REF_DIR}/reference_model/src/func_debug.cpp
    ${BASE_REF_DIR}/reference_model/src/tensor.cpp
    ${BASE_REF_DIR}/reference_model/src/verify/verify_abs_error.cpp
    ${BASE_REF_DIR}/reference_model/src/verify/verify_dot_product.cpp
    ${BASE_REF_DIR}/reference_model/src/verify/verify_entry.cpp
    ${BASE_REF_DIR}/reference_model/src/verify/verify_exact.cpp
    ${BASE_REF_DIR}/reference_model/src/verify/verify_reduce_product.cpp
    ${BASE_REF_DIR}/reference_model/src/verify/verify_relative.cpp
    ${BASE_REF_DIR}/reference_model/src/verify/verify_rescale_inexact.cpp
    ${BASE_REF_DIR}/reference_model/src/verify/verify_ulp.cpp
    ${BASE_REF_DIR}/reference_model/src/verify/verify_fp_special.cpp
    ${BASE_REF_DIR}/reference_model/src/verify/verify_utils.cpp
    ${BASE_REF_DIR}/reference_model/src/verify/verify_config.cpp
    ${BASE_REF_DIR}/reference_model/src/generate/generate_utils.cpp
    ../../src/verify/verify_main.cpp
)

target_include_directories(tosa_verify
  PRIVATE
    ${PRIVATE_INCLUDE_DIRS}
)

target_link_libraries(tosa_verify
  PRIVATE
    ${SERIALIZATION_LIB}
    ${CMAKE_DL_LIBS}
    cxxopts
)

install(TARGETS tosa_verify DESTINATION bin)
