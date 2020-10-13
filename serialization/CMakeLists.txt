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


project (tosa)

set (CMAKE_CXX_STANDARD 11)
set (CMAKE_CXX_FLAGS "-g -Wall")
set (FLATBUFFERS_SRC_DIR "../thirdparty/flatbuffers")

set (SOURCE
  tosa_serialization_handler.cpp
)

add_library(tosa_serialization STATIC ${SOURCE})

include_directories("./")

target_link_libraries(tosa_serialization PRIVATE flatbuffers)
