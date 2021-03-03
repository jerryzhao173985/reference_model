cmake_minimum_required (VERSION 3.4)

set(CMAKE_INSTALL_PREFIX ".")
project(tosa_tools LANGUAGES CXX)

option(TOSA_TOOLS_BUILD_REFERENCE_MODEL "Enable building of Tosa Reference Model" ON)

add_subdirectory(thirdparty)

if(TOSA_TOOLS_BUILD_REFERENCE_MODEL)
    add_subdirectory(reference_model)
endif()
