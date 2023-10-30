cmake_minimum_required (VERSION 3.4)

set(CMAKE_INSTALL_PREFIX ".")
project(tosa_tools LANGUAGES CXX)

option(TOSA_TOOLS_BUILD_REFERENCE_MODEL "Enable building of TOSA Reference Model" ON)
option(BUILD_TOSA_REFERENCE_MODEL_EXECUTABLE "Enable building of TOSA Reference Model executable" ON)
option(BUILD_TOSA_REFERENCE_MODEL_TESTS "Enable building of TOSA Reference Model unit tests" ON)
option(BUILD_MODEL_RUNNER_SAMPLE "Enable building of ModelRunner sample executable" OFF)

# Custom path options for third party dependencies
option(SERIALIZATION_DIR "Location where the TOSA Serialization Library 'include' folder is found" Off)
option(FLATBUFFERS_DIR "Location where the FlatBuffers 'include' and 'lib' folders is found" Off)
option(EIGEN_DIR "Location where the Eigen folder is found" Off)
option(HALF_DIR "Location where the Half folder is found" Off)
option(DOCTEST_DIR "Location where the doctest folder is found (If building unit tests)" Off)

add_subdirectory(thirdparty)

if(TOSA_TOOLS_BUILD_REFERENCE_MODEL)
    add_subdirectory(reference_model)
endif()

add_subdirectory(reference_model/custom_op_example)
