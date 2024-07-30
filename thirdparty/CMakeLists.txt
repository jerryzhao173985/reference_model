cmake_minimum_required (VERSION 3.16)

if (CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
    cmake_policy(SET CMP0135 NEW)
endif()

set(CMAKE_INSTALL_PREFIX "./thirdparty" CACHE PATH "..." FORCE)

project(thirdparty LANGUAGES CXX)

add_subdirectory(serialization_lib EXCLUDE_FROM_ALL)
