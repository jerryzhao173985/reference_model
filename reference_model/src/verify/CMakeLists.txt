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

set(PRIVATE_INCLUDE_DIRS
    ${BASE_REF_DIR}/reference_model/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../generate
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
    tosa_serialization_lib
    ${CMAKE_DL_LIBS}
    cxxopts
    eigen
    flatbuffers
    nlohmann_json
)

install(TARGETS tosa_verify DESTINATION bin)
