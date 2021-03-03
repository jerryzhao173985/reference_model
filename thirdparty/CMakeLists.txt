cmake_minimum_required (VERSION 3.4)

set(CMAKE_INSTALL_PREFIX "./thirdparty" CACHE PATH "..." FORCE)

project(thirdparty LANGUAGES CXX)

add_subdirectory(serialization_lib)
