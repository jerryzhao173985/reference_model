# Make doctest 2.4.11 compatible with CMake 4
file(READ "${DOCTEST_SOURCE_DIR}/CMakeLists.txt" CONTENTS)
string(REPLACE "cmake_minimum_required(VERSION 3.0)"
               "cmake_minimum_required(VERSION 3.10)"
               CONTENTS "${CONTENTS}")
file(WRITE "${DOCTEST_SOURCE_DIR}/CMakeLists.txt" "${CONTENTS}")
