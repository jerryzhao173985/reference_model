cmake_minimum_required (VERSION 3.16)

# Copyright (c) 2023-2025, ARM Limited.
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

set(CMAKE_INSTALL_PREFIX ".")
project(tosa_tools LANGUAGES CXX)

option(TOSA_TOOLS_BUILD_REFERENCE_MODEL "Enable building of TOSA Reference Model" ON)
option(BUILD_TOSA_REFERENCE_MODEL_EXECUTABLE "Enable building of TOSA Reference Model executable" ON)
option(BUILD_TOSA_REFERENCE_MODEL_TESTS "Enable building of TOSA Reference Model unit tests" ON)
option(BUILD_MODEL_RUNNER_SAMPLE "Enable building of ModelRunner sample executable" OFF)
option(BUILD_TOSA_REFERENCE_MODEL_PYBIND "Enable building of TOSA Reference Model Python bindings" OFF)

# Custom path options for third party dependencies
option(SERIALIZATION_DIR "Location where the TOSA Serialization Library 'include' folder is found" Off)
option(FLATBUFFERS_DIR "Location where the FlatBuffers 'include' and 'lib' folders is found" Off)
option(EIGEN_DIR "Location where the Eigen folder is found" Off)
option(HALF_DIR "Location where the Half folder is found" Off)
option(DOCTEST_DIR "Location where the doctest folder is found (If building unit tests)" Off)

if(MSVC)
  add_compile_options("/Zc:preprocessor")
endif()

if (BUILD_TOSA_REFERENCE_MODEL_PYBIND)
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
endif()

add_subdirectory(thirdparty)

if(TOSA_TOOLS_BUILD_REFERENCE_MODEL)
    add_subdirectory(reference_model)
endif()

add_subdirectory(reference_model/custom_op_example)
