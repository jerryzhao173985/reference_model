cmake_minimum_required (VERSION 3.4)

set(CMAKE_INSTALL_PREFIX "./thirdparty" CACHE PATH "..." FORCE)

project(thirdparty LANGUAGES CXX)

# Flatbuffers tests are not needed
set(FLATBUFFERS_BUILD_TESTS OFF)

add_subdirectory(flatbuffers)
