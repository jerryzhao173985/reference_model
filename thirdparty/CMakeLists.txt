cmake_minimum_required (VERSION 3.4)

# fetch 3rd party libraries
include(FetchContent)
FetchContent_Declare(
  cxxopts
  URL               https://github.com/jarro2783/cxxopts/archive/v2.2.1.tar.gz
  URL_MD5           6e70da4fc17a09f32612443f1866042e
)

set(CMAKE_INSTALL_PREFIX "./thirdparty" CACHE PATH "..." FORCE)

project(thirdparty LANGUAGES CXX)

add_subdirectory(cxxopts)
add_subdirectory(serialization_lib EXCLUDE_FROM_ALL)
add_subdirectory(json EXCLUDE_FROM_ALL)
