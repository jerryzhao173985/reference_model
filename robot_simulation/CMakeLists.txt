cmake_minimum_required(VERSION 3.16)
project(vsg_ode_robot VERSION 1.0.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(vsg REQUIRED)
find_package(vsgXchange REQUIRED)
find_package(ODE REQUIRED)
find_package(Threads REQUIRED)

# Set up include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${ODE_INCLUDE_DIRS}
)

# Source files
set(SOURCES
    src/main.cpp
    src/Robot.cpp
    src/PhysicsWorld.cpp
    src/RobotController.cpp
    src/Terrain.cpp
    src/Visualizer.cpp
)

# Create executable
add_executable(vsg_ode_robot ${SOURCES})

# Link libraries
target_link_libraries(vsg_ode_robot
    vsg::vsg
    vsgXchange::vsgXchange
    ${ODE_LIBRARIES}
    Threads::Threads
)

# Copy shaders to build directory
file(COPY shaders DESTINATION ${CMAKE_BINARY_DIR})

# Installation
install(TARGETS vsg_ode_robot DESTINATION bin)
install(DIRECTORY shaders DESTINATION share/vsg_ode_robot)