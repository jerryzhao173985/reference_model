cmake_minimum_required (VERSION 3.4)

# Copyright (c) 2020-2023, ARM Limited.
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

project(tosa_reference_model LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_CXX_COMPILER_ID STREQUAL GNU)
  set(CMAKE_CXX_FLAGS "-Wall -Wno-ignored-attributes -Wno-format-truncation")
else()
  set(CMAKE_CXX_FLAGS "-Wall -Wno-ignored-attributes")
endif()

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
  set(SERIALIZATION_DIR "../thirdparty/serialization_lib/")
endif()

# If Flatbuffers, Eigen, Half path isn't specified, set to thirdparty directory.
if(NOT FLATBUFFERS_DIR)
  set(FLATBUFFERS_DIR "../thirdparty/serialization_lib/third_party/flatbuffers/")
endif()

if(NOT EIGEN_DIR)
  set(EIGEN_DIR "../thirdparty/eigen/")
endif()

if(NOT HALF_DIR)
  set(HALF_DIR "../thirdparty/serialization_lib/third_party/half")
endif()

if(NOT JSON_DIR)
  set(JSON_DIR "../thirdparty/json")
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Common sources required for TOSA Reference Model library, executable and unit tests
set(CXX_SOURCE
    src/func_debug.cc
    src/graph_node.cc
    src/model_runner_impl.cc
    src/model_runner.cc
    src/operators.cc
    src/subgraph_traverser.cc
    src/tensor.cc
    src/generate/generate_dot_product_states.cc
    src/generate/generate_dot_product.cc
    src/generate/generate_entry.cc
    src/generate/generate_utils.cc
    src/verify/verify_dot_product.cc
    src/verify/verify_entry.cc
    src/verify/verify_exact.cc
    src/verify/verify_reduce_product.cc
    src/verify/verify_ulp.cc
    src/verify/verify_utils.cc
    src/ops/op_factory.cc
    src/ops/tensor_ops.cc
    src/ops/activation_funcs.cc
    src/ops/ewise_binary.cc
    src/ops/ewise_unary.cc
    src/ops/ewise_ternary.cc
    src/ops/comparison.cc
    src/ops/reduction.cc
    src/ops/data_layout.cc
    src/ops/scatter_gather.cc
    src/ops/image.cc
    src/ops/type_conversion.cc
    src/ops/data_nodes.cc
    src/ops/custom.cc
    src/ops/control_flow.cc
)

set(PUBLIC_INCLUDE_DIRS
    $<INSTALL_INTERFACE:include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SRC_DIR}/include>
)
set(PRIVATE_INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${FLATBUFFERS_DIR}/include
    ${EIGEN_DIR}
    ${EIGEN_DIR}/unsupported/
    ${SERIALIZATION_DIR}/include
    ${HALF_DIR}/include
    ${JSON_DIR}/single_include
)

# Build TOSA Reference Model library
add_library(tosa_reference_model_lib STATIC ${CXX_SOURCE})

target_include_directories(tosa_reference_model_lib
  PUBLIC
    ${PUBLIC_INCLUDE_DIRS}
  PRIVATE
    ${PRIVATE_INCLUDE_DIRS}
)

target_link_libraries(tosa_reference_model_lib
  PRIVATE
    ${SERIALIZATION_LIB}
)

set(PUBLIC_HEADERS)
list(APPEND PUBLIC_HEADERS
    include/debug_modes.def
    include/debug_types.h
    include/dtype.h
    include/func_config.h
    include/func_debug.h
    include/generate.h
    include/graph_status.h
    include/model_common.h
    include/model_runner.h
    include/types.h
    include/verify.h
    include/version.h
)

set_target_properties(tosa_reference_model_lib PROPERTIES PUBLIC_HEADER "${PUBLIC_HEADERS}")

# Build TOSA verification library
add_library(tosa_reference_verify_lib SHARED
  src/verify/verify_dot_product.cc
  src/verify/verify_entry.cc
  src/verify/verify_exact.cc
  src/verify/verify_reduce_product.cc
  src/verify/verify_ulp.cc
  src/verify/verify_utils.cc
  src/verify/verify_config.cc
  src/func_debug.cc
)
target_include_directories(tosa_reference_verify_lib
  PUBLIC
    ${PUBLIC_INCLUDE_DIRS}
  PRIVATE
    ${PRIVATE_INCLUDE_DIRS}
)

# Build TOSA generator library
add_library(tosa_reference_generate_lib SHARED
  src/generate/generate_dot_product_states.cc
  src/generate/generate_dot_product.cc
  src/generate/generate_entry.cc
  src/generate/generate_utils.cc
  src/func_debug.cc
)
target_include_directories(tosa_reference_generate_lib
  PUBLIC
    ${PUBLIC_INCLUDE_DIRS}
  PRIVATE
    ${PRIVATE_INCLUDE_DIRS}
)

# Build TOSA Refererence Model executable
if(BUILD_TOSA_REFERENCE_MODEL_EXECUTABLE)
  set(CXX_SOURCE_EX src/main.cpp)
  list(APPEND CXX_SOURCE_EX ${CXX_SOURCE})

  add_executable(tosa_reference_model ${CXX_SOURCE_EX})

  target_include_directories(tosa_reference_model
    PUBLIC
      ${PUBLIC_INCLUDE_DIRS}
    PRIVATE
      ${PRIVATE_INCLUDE_DIRS}
  )

  target_link_libraries(tosa_reference_model
    PRIVATE
      ${SERIALIZATION_LIB}
      nlohmann_json::nlohmann_json
      cxxopts
  )

  install(TARGETS tosa_reference_model DESTINATION bin)
endif()

if(BUILD_TOSA_REFERENCE_MODEL_TESTS)
  # Set definition so unit tests can find examples directory.
  add_definitions(-DPROJECT_ROOT=\"${CMAKE_CURRENT_SOURCE_DIR}/\")

  # Set doctest location if not specified.
  if(NOT DOCTEST_DIR)
    set(DOCTEST_DIR "../thirdparty/doctest/doctest")
  endif()

  # Sources only required for unit tests.
  set(CXX_SOURCE_TESTS
      test/generate_tests.cpp
      test/model_runner_tests.cpp
      test/verify_tests.cpp
      ${DOCTEST_DIR}/doctest.h
  )

  list(APPEND CXX_SOURCE_TESTS ${CXX_SOURCE})

  add_executable(unit_tests ${CXX_SOURCE_TESTS})

  target_include_directories(unit_tests
    PUBLIC
      ${PUBLIC_INCLUDE_DIRS}
    PRIVATE
      ${PRIVATE_INCLUDE_DIRS}
      ${DOCTEST_DIR}
    )

  target_link_libraries(unit_tests
    PRIVATE
      ${SERIALIZATION_LIB}
  )
endif()

if(BUILD_MODEL_RUNNER_SAMPLE)
  # Set definition so sample executable can find examples directory.
  add_definitions(-DPROJECT_ROOT=\"${CMAKE_CURRENT_SOURCE_DIR}/\")

  # Sources only required for example executable.
  set(CXX_SOURCE_SAMPLE
      samples/model_runner_simple_sample.cpp
  )

  list(APPEND CXX_SOURCE_SAMPLE ${CXX_SOURCE})

  add_executable(model_runner_sample ${CXX_SOURCE_SAMPLE})

  target_include_directories(model_runner_sample
    PUBLIC
      ${PUBLIC_INCLUDE_DIRS}
    PRIVATE
      ${PRIVATE_INCLUDE_DIRS}
  )

  target_link_libraries(model_runner_sample
    PRIVATE
      ${SERIALIZATION_LIB}
  )
endif()

# Follow GNU packaging norms for installation directory structure.
include(GNUInstallDirs)
install(
  TARGETS tosa_reference_model_lib EXPORT TosaReferenceModelLibTargets
  PUBLIC_HEADER
  ARCHIVE
)

install(EXPORT TosaReferenceModelLibTargets
  FILE TosaReferenceModelLibTargets.cmake
  NAMESPACE TosaReference::
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/tosa_reference_model_lib"
)
