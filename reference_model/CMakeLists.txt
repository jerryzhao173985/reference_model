cmake_minimum_required (VERSION 3.18)

# Copyright (c) 2020-2025, ARM Limited.
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

include(CheckLinkerFlag)

project(tosa_reference_model LANGUAGES CXX)

# The verify directy includes a cmake list that builds the verifier main
# utility tool only.
if(BUILD_TOSA_REFERENCE_MODEL_EXECUTABLE)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/src/verify verify)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_CXX_COMPILER_ID STREQUAL GNU)
  set(CMAKE_CXX_FLAGS "-Wall -Wno-ignored-attributes -Wno-format-truncation")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL Clang)
  set(CMAKE_CXX_FLAGS "-Wall -Wno-ignored-attributes")
endif()

get_target_property(EIGEN_INCLUDES eigen INTERFACE_INCLUDE_DIRECTORIES)
get_target_property(FLATBUFFERS_INCLUDES flatbuffers INTERFACE_INCLUDE_DIRECTORIES)
get_target_property(NLOHMANN_JSON_INCLUDES nlohmann_json INTERFACE_INCLUDE_DIRECTORIES)
get_target_property(PYBIND11_INCLUDES pybind11_headers INTERFACE_INCLUDE_DIRECTORIES)
get_target_property(SERIALIZATION_INCLUDES tosa_serialization_lib INTERFACE_INCLUDE_DIRECTORIES)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  set(CXX_FLAGS
      -Wformat=2 -Wconversion -Wimplicit-fallthrough
      -Wformat-security
      -U_FORTIFY_SOURCE -D_FORTIFY_SOURCE=2
      -D_GLIBCXX_ASSERTIONS
      -fstack-clash-protection -fstack-protector-strong
      -fno-delete-null-pointer-checks -fno-strict-overflow -fno-strict-aliasing)
else()
  # TODO: Add compiler flags for non-linux systems
  set(CXX_FLAGS "")
endif()

function(filter_unsupported_linker_flags in_flags out_flags)
  set(_out_list "")
  foreach(_flag IN LISTS ${in_flags})
    check_linker_flag(CXX "${_flag}" _supported_flag)
    if(_supported_flag)
      list(APPEND _out_list "${_flag}")
    else()
      message(STATUS "Skipping unsupported linker flag: ${_flag}")
    endif()
  endforeach()
  set(${out_flags} "${_out_list}" PARENT_SCOPE)
endfunction()

# Set linker flags, if the selected linker supports them
message(STATUS "Using linker: ${CMAKE_LINKER}")
set(_linker_flags
  "-Wl,-z,nodlopen"
  "-Wl,-z,noexecstack"
  "-Wl,-z,relro"
  "-Wl,-z,now"
  "-Wl,--as-needed"
  "-Wl,--no-copy-dt-needed-entries"
  "-fpie"
)
filter_unsupported_linker_flags(_linker_flags LINKER_FLAGS)

# Apply flags only target
function(apply_flags target)
    if (TARGET ${target})
        target_compile_options(${target} PRIVATE "${CXX_FLAGS}")
        target_link_options(${target} PRIVATE ${LINKER_FLAGS})
    endif()
endfunction()

# Common sources required for TOSA Reference Model library, executable and unit tests
set(CXX_SOURCE
    src/func_debug.cpp
    src/graph_node.cpp
    src/model_runner_impl.cpp
    src/model_runner.cpp
    src/subgraph_traverser.cpp
    src/tensor.cpp
    src/generate/generate_dot_product_states.cpp
    src/generate/generate_dot_product.cpp
    src/generate/generate_pseudo_random.cpp
    src/generate/generate_fixed_data.cpp
    src/generate/generate_full_range.cpp
    src/generate/generate_special.cpp
    src/generate/generate_fp_special.cpp
    src/generate/generate_int_special.cpp
    src/generate/generate_entry.cpp
    src/generate/generate_utils.cpp
    src/verify/verify_abs_error.cpp
    src/verify/verify_dot_product.cpp
    src/verify/verify_entry.cpp
    src/verify/verify_exact.cpp
    src/verify/verify_reduce_product.cpp
    src/verify/verify_relative.cpp
    src/verify/verify_rescale_inexact.cpp
    src/verify/verify_ulp.cpp
    src/verify/verify_fp_special.cpp
    src/verify/verify_utils.cpp
    src/ops/op_factory.cpp
    src/ops/tensor_ops.cpp
    src/ops/activation_funcs.cpp
    src/ops/ewise_binary.cpp
    src/ops/ewise_unary.cpp
    src/ops/ewise_ternary.cpp
    src/ops/comparison.cpp
    src/ops/reduction.cpp
    src/ops/data_layout.cpp
    src/ops/scatter_gather.cpp
    src/ops/image.cpp
    src/ops/type_conversion.cpp
    src/ops/data_nodes.cpp
    src/ops/custom.cpp
    src/ops/control_flow.cpp
    src/ops/shape.cpp
)

set(PUBLIC_INCLUDE_DIRS
    $<INSTALL_INTERFACE:include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SRC_DIR}/include>
)

set(PRIVATE_INCLUDE_DIRS
    ${FLATBUFFERS_INCLUDES}
    ${EIGEN_INCLUDES}
    ${SERIALIZATION_INCLUDES}
    ${NLOHMANN_JSON_INCLUDES}
    ${PYBIND11_INCLUDES}
)

# Build TOSA Reference Model library
add_library(tosa_reference_model_lib STATIC ${CXX_SOURCE})

if(MSVC)
  target_compile_options(tosa_reference_model_lib PRIVATE /Ob0)
  target_compile_options(tosa_reference_model_lib PRIVATE /bigobj)
endif()

apply_flags(tosa_reference_model_lib)

target_include_directories(tosa_reference_model_lib
  SYSTEM PUBLIC # Suppress warnings for third-party headers
    "$<BUILD_INTERFACE:${PRIVATE_INCLUDE_DIRS}>"
  PUBLIC
    ${PUBLIC_INCLUDE_DIRS}
  PRIVATE
    include
    src/verify
    src
)

target_link_libraries(tosa_reference_model_lib
  PRIVATE
    tosa_serialization_lib
)

set(PUBLIC_HEADERS)
list(APPEND PUBLIC_HEADERS
    include/custom_op_interface.h
    include/custom_registry.h
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
  src/verify/verify_abs_error.cpp
  src/verify/verify_dot_product.cpp
  src/verify/verify_entry.cpp
  src/verify/verify_exact.cpp
  src/verify/verify_reduce_product.cpp
  src/verify/verify_relative.cpp
  src/verify/verify_ulp.cpp
  src/verify/verify_rescale_inexact.cpp
  src/verify/verify_fp_special.cpp
  src/verify/verify_utils.cpp
  src/verify/verify_config.cpp
  src/func_debug.cpp
)

# Linker flags specific to shared lib
set(_shared_lib_linker_flags
  "-Wl,-z,noexecstack"
  "-Wl,-z,relro"
  "-Wl,-z,now"
  "-Wl,--as-needed"
  "-Wl,--no-copy-dt-needed-entries"
  "-fPIC"
)
filter_unsupported_linker_flags(_shared_lib_linker_flags LINKER_FLAGS_SHARED_LIB)

target_compile_options(tosa_reference_verify_lib PRIVATE ${CXX_FLAGS})
target_link_options(tosa_reference_verify_lib PRIVATE ${LINKER_FLAGS_SHARED_LIB})

target_include_directories(tosa_reference_verify_lib
  SYSTEM PUBLIC # Suppress warnings for third-party headers
    "$<BUILD_INTERFACE:${PRIVATE_INCLUDE_DIRS}>"
  PUBLIC
    ${PUBLIC_INCLUDE_DIRS}
  PRIVATE
    include
    src/verify
    src
)

# Build TOSA generator library
add_library(tosa_reference_generate_lib SHARED
  src/generate/generate_dot_product_states.cpp
  src/generate/generate_dot_product.cpp
  src/generate/generate_pseudo_random.cpp
  src/generate/generate_fixed_data.cpp
  src/generate/generate_full_range.cpp
  src/generate/generate_special.cpp
  src/generate/generate_fp_special.cpp
  src/generate/generate_int_special.cpp
  src/generate/generate_entry.cpp
  src/generate/generate_utils.cpp
  src/generate/generate_config.cpp
  src/func_debug.cpp
)

target_compile_options(tosa_reference_generate_lib PRIVATE ${CXX_FLAGS})
target_link_options(tosa_reference_generate_lib PRIVATE ${LINKER_FLAGS_SHARED_LIB})

target_include_directories(tosa_reference_generate_lib
  SYSTEM PUBLIC # Suppress warnings for third-party headers
    "$<BUILD_INTERFACE:${PRIVATE_INCLUDE_DIRS}>"
  PUBLIC
    ${PUBLIC_INCLUDE_DIRS}
  PRIVATE
    include
    src/verify
    src
)

# Build TOSA Refererence Model executable
if(BUILD_TOSA_REFERENCE_MODEL_EXECUTABLE)
  set(CXX_SOURCE_EX src/main.cpp)
  list(APPEND CXX_SOURCE_EX ${CXX_SOURCE})
  if (WIN32)
    set(CXX_SOURCE_WIN src/load_library_win.cpp)
    list(APPEND CXX_SOURCE_EX ${CXX_SOURCE_WIN})
  endif()

  add_executable(tosa_reference_model ${CXX_SOURCE_EX})

if(MSVC)
  target_compile_options(tosa_reference_model PRIVATE /Ob0)
  target_compile_options(tosa_reference_model PRIVATE /bigobj)
endif()
  apply_flags(tosa_reference_model)

  target_include_directories(tosa_reference_model
    SYSTEM PUBLIC # Suppress warnings for third-party headers
      "$<BUILD_INTERFACE:${PRIVATE_INCLUDE_DIRS}>"
    PUBLIC
      ${PUBLIC_INCLUDE_DIRS}
    PRIVATE
      include
      src/verify
      src
  )

  target_link_libraries(tosa_reference_model
    PRIVATE
      tosa_serialization_lib
      nlohmann_json::nlohmann_json
      cxxopts
      ${CMAKE_DL_LIBS}
  )

  install(TARGETS tosa_reference_model DESTINATION bin)
endif()

if(BUILD_TOSA_REFERENCE_MODEL_TESTS)
  # Set definition so unit tests can find examples directory.
  add_definitions(-DPROJECT_ROOT=\"${CMAKE_CURRENT_SOURCE_DIR}/\")

  set(TEST_INCLUDE_DIRS
    test/include
  )

  # Sources only required for unit tests.
  set(CXX_SOURCE_TESTS
      test/generate_tests.cpp
      test/model_runner_tests.cpp
      test/verify_tests.cpp
      test/verify_exe_testing.cpp
      test/ref_model_tests/tensor_ops_tests.cpp
      test/ref_model_tests/reduction_tests.cpp
      test/ref_model_tests/ewise_binary_tests.cpp
      test/ref_model_tests/ewise_ternary_tests.cpp
      test/ref_model_tests/type_conversion_tests.cpp
      test/ref_model_tests/activation_funcs_tests.cpp
      test/ref_model_tests/data_layout_tests.cpp
      test/ref_model_tests/dump_tensor_tests.cpp
  )

  list(APPEND CXX_SOURCE_TESTS ${CXX_SOURCE})

  add_executable(unit_tests ${CXX_SOURCE_TESTS})

  if(MSVC)
    target_compile_options(unit_tests PRIVATE /Ob0)
    target_compile_options(unit_tests PRIVATE /bigobj)
  endif()
  apply_flags(unit_tests)

  target_include_directories(unit_tests
    SYSTEM PUBLIC # Suppress warnings for third-party headers
      "$<BUILD_INTERFACE:${DOCTEST_DIR}>"
      "$<BUILD_INTERFACE:${PRIVATE_INCLUDE_DIRS}>"
    PUBLIC
      ${PUBLIC_INCLUDE_DIRS}
    PRIVATE
      ${TEST_INCLUDE_DIRS}
      include
      src/verify
      src
    )

  # GCC-8 and below require explicit linking against stdc++fs
  set(is_gnu "$<CXX_COMPILER_ID:GNU>")
  set(is_8_or_below "$<VERSION_LESS:$<CXX_COMPILER_VERSION>,9.0.0>")
  set(is_gnu_8_or_below "$<AND:${is_gnu},${is_8_or_below}>")
  target_link_libraries(unit_tests
    PRIVATE
      tosa_serialization_lib
      doctest
      "$<${is_gnu_8_or_below}:stdc++fs>"
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

  apply_flags(model_runner_sample)

  target_include_directories(model_runner_sample
    SYSTEM PUBLIC # Suppress warnings for third-party headers
      "$<BUILD_INTERFACE:${PRIVATE_INCLUDE_DIRS}>"
    PUBLIC
      ${PUBLIC_INCLUDE_DIRS}
    PRIVATE
      include
      src/verify
      src
  )

  target_link_libraries(model_runner_sample
    PRIVATE
      tosa_serialization_lib
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

if (BUILD_TOSA_REFERENCE_MODEL_PYBIND)
  set(Python_FIND_VIRTUALENV ONLY)
  find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)

  python_add_library(py_tosa_reference_model MODULE python_bindings/python_bindings.cpp WITH_SOABI)

  set_target_properties(py_tosa_reference_model PROPERTIES OUTPUT_NAME tosa_reference_model)

  target_compile_options(py_tosa_reference_model PRIVATE "${CXX_FLAGS}")
  target_link_options(py_tosa_reference_model PRIVATE ${LINKER_FLAGS_SHARED_LIB})

  target_include_directories(py_tosa_reference_model
    SYSTEM PUBLIC # Suppress warnings for third-party headers
      "$<BUILD_INTERFACE:${PRIVATE_INCLUDE_DIRS}>"
    PUBLIC
      ${PUBLIC_INCLUDE_DIRS}
    PRIVATE
      include
      src/verify
      src
  )

  find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
  target_include_directories(py_tosa_reference_model PRIVATE ${Python3_INCLUDE_DIRS})
  target_link_libraries(py_tosa_reference_model
    PRIVATE
      tosa_reference_model_lib
      ${Python3_LIBRARIES}
  )

  target_compile_definitions(py_tosa_reference_model PRIVATE VERSION_INFO=${PROJECT_VERSION})

  install(TARGETS py_tosa_reference_model DESTINATION src)
endif()
