cmake_minimum_required (VERSION 3.16)

if (CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
    cmake_policy(SET CMP0135 NEW)
endif()

# fetch 3rd party libraries
include(FetchContent)
FetchContent_Declare(
  cxxopts
  URL               https://github.com/jarro2783/cxxopts/archive/v2.2.1.tar.gz
  URL_MD5           6e70da4fc17a09f32612443f1866042e
)

set(CMAKE_INSTALL_PREFIX "./thirdparty" CACHE PATH "..." FORCE)

project(thirdparty LANGUAGES CXX)

add_subdirectory(serialization_lib EXCLUDE_FROM_ALL)

if(BUILD_TOSA_REFERENCE_MODEL_EXECUTABLE)
    add_subdirectory(cxxopts)
    add_subdirectory(json EXCLUDE_FROM_ALL)
endif()

if(BUILD_TOSA_REFERENCE_MODEL_TESTS)
    add_subdirectory(doctest EXCLUDE_FROM_ALL)
endif()
