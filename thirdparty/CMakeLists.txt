# SPDX-FileCopyrightText: Copyright 2020-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required (VERSION 3.16)

if (CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
    cmake_policy(SET CMP0135 NEW)
endif()

project(thirdparty LANGUAGES CXX)

include(FetchContent)

add_subdirectory(serialization_lib EXCLUDE_FROM_ALL)

# To override the location of the one of these sources, use the
# FETCHCONTENT_SOURCE_DIR_<uppercaseName> CMake variable. For example:
# cmake -DFETCHCONTENT_SOURCE_DIR_CXXOPTS=/path/to/cxxopts ...

if (NOT TARGET cxxopts)
  FetchContent_Declare(
  cxxopts
  URL https://github.com/jarro2783/cxxopts/archive/v2.2.1.tar.gz
  URL_HASH SHA256=984aa3c8917d649b14d7f6277104ce38dd142ce378a9198ec926f03302399681
  )
  FetchContent_MakeAvailable(cxxopts)
endif()

if (NOT TARGET doctest)
  FetchContent_Declare(
    doctest
    URL https://github.com/doctest/doctest/archive/refs/tags/v2.4.11.tar.gz
    URL_HASH SHA256=632ed2c05a7f53fa961381497bf8069093f0d6628c5f26286161fbd32a560186
  )
  FetchContent_MakeAvailable(doctest)
endif()

if (NOT TARGET eigen)
  FetchContent_Declare(
    eigen
    URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
    URL_HASH SHA256=8586084f71f9bde545ee7fa6d00288b264a2b7ac3607b974e54d13e7162c1c72
  )
  FetchContent_MakeAvailable(eigen)
endif()

if (NOT TARGET nlohmann_json)
  FetchContent_Declare(
    nlohmann_json
    URL https://github.com/nlohmann/json/archive/refs/tags/v3.11.3.tar.gz
    URL_HASH SHA256=0d8ef5af7f9794e3263480193c491549b2ba6cc74bb018906202ada498a79406
  )
  FetchContent_MakeAvailable(nlohmann_json)
endif()

if (NOT TARGET pybind11_headers)
  FetchContent_Declare(
    pybind11
    URL https://github.com/pybind/pybind11/archive/refs/tags/v2.13.6.tar.gz
    URL_HASH SHA256=e08cb87f4773da97fa7b5f035de8763abc656d87d5773e62f6da0587d1f0ec20
  )
  FetchContent_MakeAvailable(pybind11)
endif()
