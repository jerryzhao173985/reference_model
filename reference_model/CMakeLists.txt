cmake_minimum_required (VERSION 3.4)

# Copyright (c) 2020, ARM Limited.
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

set(FLATBUFFERS_DIR "../thirdparty/serialization_lib/third_party/flatbuffers/")
set(SERIALIZATION_DIR "../thirdparty/serialization_lib/")

set (CXX_SOURCE
     src/main.cpp
     src/tensor.cc
     src/graph_node.cc
     src/subgraph_traverser.cc
     src/func_debug.cc
     src/func_config.cc
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

add_executable(tosa_reference_model ${CXX_SOURCE})

target_include_directories(tosa_reference_model
  PUBLIC
    $<INSTALL_INTERFACE:include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SRC_DIR}/include>
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${FLATBUFFERS_DIR}/include
    ../thirdparty/eigen/
    ../thirdparty/eigen/unsupported/
    ${SERIALIZATION_DIR}/include
)

target_link_libraries(tosa_reference_model
  PRIVATE
    tosa_serialization_lib
    nlohmann_json::nlohmann_json
)

install (TARGETS tosa_reference_model DESTINATION bin)
