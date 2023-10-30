cmake_minimum_required (VERSION 3.4)

# Copyright (c) 2023, ARM Limited.
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

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set (CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

add_library(tosa_example_plugin SHARED
    custom_op_example.cpp
)

target_include_directories(tosa_example_plugin
    PUBLIC
        ${PUBLIC_INCLUDE_DIRS}
        "../../thirdparty/serialization_lib/include"
        "../../thirdparty/serialization_lib/third_party/flatbuffers/include"
        "../../thirdparty/serialization_lib/third_party/half/include"
        "../../thirdparty/eigen/"
        "../../thirdparty/eigen/unsupported"
        "../include"
        "../src"
        "../src/ops"
    PRIVATE
        ${PRIVATE_INCLUDE_DIRS}
)