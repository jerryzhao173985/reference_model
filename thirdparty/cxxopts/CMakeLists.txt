set(CXXOPTS_BUILD_EXAMPLES OFF CACHE INTERNAL "")
set(CXXOPTS_BUILD_TESTS OFF CACHE INTERNAL "")
FetchContent_GetProperties(cxxopts)
if(NOT cxxopts_POPULATED)
    FetchContent_Populate(cxxopts)
    add_subdirectory(${cxxopts_SOURCE_DIR} ${cxxopts_BINARY_DIR} EXCLUDE_FROM_ALL)

endif()
