# Copyright (c) 2024, ARM Limited.
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

cmake_minimum_required (VERSION 3.16)

if (CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
    cmake_policy(SET CMP0135 NEW)
endif()

include(FetchContent)

# Fetch cxxopts
FetchContent_Declare(
  cxxopts
  URL               https://github.com/jarro2783/cxxopts/archive/v2.2.1.tar.gz
  URL_MD5           6e70da4fc17a09f32612443f1866042e
  EXCLUDE_FROM_ALL
)
set(CXXOPTS_BUILD_EXAMPLES OFF CACHE INTERNAL "")
set(CXXOPTS_BUILD_TESTS OFF CACHE INTERNAL "")
FetchContent_GetProperties(cxxopts)
if(NOT cxxopts_POPULATED)
    FetchContent_Populate(cxxopts)
    add_subdirectory(${cxxopts_SOURCE_DIR} ${cxxopts_BINARY_DIR} EXCLUDE_FROM_ALL)

endif()

# Fetch doctest
FetchContent_Declare(
  doctest
  GIT_REPOSITORY    https://github.com/doctest/doctest.git
  GIT_TAG           86892fc480f80fb57d9a3926cb506c0e974489d8
  EXCLUDE_FROM_ALL
)
FetchContent_GetProperties(doctest)
if(NOT doctest_POPULATED)
    FetchContent_Populate(doctest)
    add_subdirectory(${doctest_SOURCE_DIR} ${doctest_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()

# Fetch eigen
FetchContent_Declare(
  eigen
  GIT_REPOSITORY    https://gitlab.com/libeigen/eigen.git
  GIT_TAG           3147391d946bb4b6c68edd901f2add6ac1f31f8c
  EXCLUDE_FROM_ALL
)
FetchContent_GetProperties(eigen)
if(NOT eigen_POPULATED)
    FetchContent_Populate(eigen)
    add_subdirectory(${eigen_SOURCE_DIR} ${eigen_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()

# Fetch json
FetchContent_Declare(
  json
  GIT_REPOSITORY    https://github.com/nlohmann/json.git
  GIT_TAG           e7452d87783fbf6e9d320d515675e26dfd1271c5
  EXCLUDE_FROM_ALL
)
FetchContent_GetProperties(json)
if(NOT json_POPULATED)
    FetchContent_Populate(json)
    add_subdirectory(${json_SOURCE_DIR} ${json_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()
