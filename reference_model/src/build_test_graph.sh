#!/bin/bash
# Enhanced build script for graph examples
# Run from the reference_model directory

echo "=== TOSA Reference Model Graph Examples Build Script ==="

# Set paths
SRC_DIR="reference_model/src"

# Compiler settings
CXX="${CXX:-g++}"
CXXFLAGS="-std=c++17 -O2 -g -Wall"

# Function to check if the reference model is built
check_reference_model() {
    local possible_dirs=(
        "build/debug"
        "build"
        "build/release" 
        "../build"
        "../build/debug"
        "../build/release"
    )
    
    for dir in "${possible_dirs[@]}"; do
        if [ -f "${dir}/reference_model/libtosa_reference_model_lib.a" ] || 
           [ -f "${dir}/reference_model/libtosa_reference_model_lib.dylib" ] ||
           [ -f "${dir}/reference_model/libtosa_reference_model_lib.so" ]; then
            echo "Found reference model in: ${dir}"
            export TOSA_BUILD_DIR="${dir}"
            return 0
        fi
    done
    return 1
}

# Function to build the full TOSA example
build_full_example() {
    echo ""
    echo "--- Building Full TOSA Example ---"
    echo "This uses the actual TOSA reference model infrastructure"
    
    if ! check_reference_model; then
        echo "✗ TOSA reference model not found!"
        echo ""
        echo "To build the reference model first:"
        echo "  mkdir build && cd build"
        echo "  cmake .. && make"
        echo "  cd .."
        echo ""
        return 1
    fi
    
    # Setup include paths
    FULL_CXXFLAGS="${CXXFLAGS}"
    FULL_CXXFLAGS="${FULL_CXXFLAGS} -I${SRC_DIR}"
    FULL_CXXFLAGS="${FULL_CXXFLAGS} -I${SRC_DIR}/ops"
    FULL_CXXFLAGS="${FULL_CXXFLAGS} -Iinclude"
    FULL_CXXFLAGS="${FULL_CXXFLAGS} -Ireference_model/include"
    
    # CRITICAL: Add serialization library include for attribute.h
    FULL_CXXFLAGS="${FULL_CXXFLAGS} -Ithirdparty/serialization_lib/include"
    
    # Add third party includes if they exist
    # Check build deps first (where cmake downloads dependencies)
    if [ -d "${TOSA_BUILD_DIR}/_deps/eigen-src" ]; then
        FULL_CXXFLAGS="${FULL_CXXFLAGS} -I${TOSA_BUILD_DIR}/_deps/eigen-src"
        FULL_CXXFLAGS="${FULL_CXXFLAGS} -I${TOSA_BUILD_DIR}/_deps/eigen-src/unsupported"
    elif [ -d "thirdparty/eigen" ]; then
        FULL_CXXFLAGS="${FULL_CXXFLAGS} -Ithirdparty/eigen"
        FULL_CXXFLAGS="${FULL_CXXFLAGS} -Ithirdparty/eigen/unsupported"
    elif [ -d "third_party/eigen" ]; then
        FULL_CXXFLAGS="${FULL_CXXFLAGS} -Ithird_party/eigen"
        FULL_CXXFLAGS="${FULL_CXXFLAGS} -Ithird_party/eigen/unsupported"
    fi
    
    # Flatbuffers - critical for attribute.h
    # Check in build deps first (where cmake downloads it)
    if [ -d "${TOSA_BUILD_DIR}/_deps/flatbuffers-src/include" ]; then
        FULL_CXXFLAGS="${FULL_CXXFLAGS} -I${TOSA_BUILD_DIR}/_deps/flatbuffers-src/include"
    elif [ -d "thirdparty/flatbuffers/include" ]; then
        FULL_CXXFLAGS="${FULL_CXXFLAGS} -Ithirdparty/flatbuffers/include"
    elif [ -d "third_party/flatbuffers/include" ]; then
        FULL_CXXFLAGS="${FULL_CXXFLAGS} -Ithird_party/flatbuffers/include"
    fi
    
    # JSON includes
    if [ -d "${TOSA_BUILD_DIR}/_deps/nlohmann_json-src/single_include" ]; then
        FULL_CXXFLAGS="${FULL_CXXFLAGS} -I${TOSA_BUILD_DIR}/_deps/nlohmann_json-src/single_include"
    elif [ -d "thirdparty/json/single_include" ]; then
        FULL_CXXFLAGS="${FULL_CXXFLAGS} -Ithirdparty/json/single_include"
    elif [ -d "third_party/json/single_include" ]; then
        FULL_CXXFLAGS="${FULL_CXXFLAGS} -Ithird_party/json/single_include"
    fi
    
    # Half precision includes
    if [ -d "${TOSA_BUILD_DIR}/include" ]; then
        # Generated headers might be here
        FULL_CXXFLAGS="${FULL_CXXFLAGS} -I${TOSA_BUILD_DIR}/include"
    fi
    if [ -d "thirdparty/half/include" ]; then
        FULL_CXXFLAGS="${FULL_CXXFLAGS} -Ithirdparty/half/include"
    elif [ -d "third_party/half/include" ]; then
        FULL_CXXFLAGS="${FULL_CXXFLAGS} -Ithird_party/half/include"
    fi
    
    # Eigen configuration
    FULL_CXXFLAGS="${FULL_CXXFLAGS} -DEIGEN_NO_DEBUG"
    FULL_CXXFLAGS="${FULL_CXXFLAGS} -DEIGEN_STACK_ALLOCATION_LIMIT=0"
    
    # Find libraries - we know they're in build/debug
    LIBS=""
    
    # Reference model library
    if [ -f "${TOSA_BUILD_DIR}/reference_model/libtosa_reference_model_lib.a" ]; then
        LIBS="${LIBS} ${TOSA_BUILD_DIR}/reference_model/libtosa_reference_model_lib.a"
        echo "Found static reference model library"
    elif [ -f "${TOSA_BUILD_DIR}/reference_model/libtosa_reference_model_lib.dylib" ]; then
        LIBS="${LIBS} ${TOSA_BUILD_DIR}/reference_model/libtosa_reference_model_lib.dylib"
        echo "Found dynamic reference model library"
    fi
    
    # Serialization library (in thirdparty)
    if [ -f "${TOSA_BUILD_DIR}/thirdparty/serialization_lib/libtosa_serialization_lib.a" ]; then
        LIBS="${LIBS} ${TOSA_BUILD_DIR}/thirdparty/serialization_lib/libtosa_serialization_lib.a"
        echo "Found serialization library"
    fi
    
    # Also check for flatbuffers if needed
    if [ -f "${TOSA_BUILD_DIR}/lib/libflatbuffers.a" ]; then
        LIBS="${LIBS} ${TOSA_BUILD_DIR}/lib/libflatbuffers.a"
        echo "Found flatbuffers library in lib/"
    elif [ -f "${TOSA_BUILD_DIR}/_deps/flatbuffers-build/libflatbuffers.a" ]; then
        LIBS="${LIBS} ${TOSA_BUILD_DIR}/_deps/flatbuffers-build/libflatbuffers.a"
        echo "Found flatbuffers library in _deps/"
    fi
    
    echo ""
    echo "Include flags: ${FULL_CXXFLAGS}"
    echo "Libraries: ${LIBS}"
    echo ""
    
    ${CXX} ${FULL_CXXFLAGS} -o test_graph_example ${SRC_DIR}/test_graph_example.cpp ${LIBS} -lpthread
    
    if [ $? -eq 0 ]; then
        echo "✓ Full TOSA example built successfully!"
        echo "Run with: ./test_graph_example"
        return 0
    else
        echo "✗ Failed to build full TOSA example"
        return 1
    fi
}

# # Function to build the simple standalone demo
# build_simple_demo() {
#     echo ""
#     echo "--- Building Simple Tensor Demo (standalone) ---"
#     echo "This demonstrates core graph execution concepts without dependencies"
    
#     ${CXX} ${CXXFLAGS} -o simple_tensor_demo ${SRC_DIR}/simple_tensor_demo.cpp
    
#     if [ $? -eq 0 ]; then
#         echo "✓ Simple demo built successfully!"
#         echo "Run with: ./simple_tensor_demo"
#         return 0
#     else
#         echo "✗ Failed to build simple demo"
#         return 1
#     fi
# }

# # try to build the simple demo first
# if build_simple_demo; then
#     echo ""
#     echo "🎉 Simple demo is ready!"
#     echo ""
#     echo "The simple_tensor_demo shows the core concepts:"
#     echo "  - Tensor data structures"
#     echo "  - Graph nodes and operations" 
#     echo "  - Ready-list scheduling"
#     echo "  - Step-by-step execution"
#     echo ""
# else
#     echo "❌ Could not build simple demo"
# fi

# Then try the full example
if build_full_example; then
    echo ""
    echo "🎉 Full TOSA example is also ready!"
    echo ""
    echo "The test_graph_example shows the real TOSA infrastructure:"
    echo "  - SubgraphTraverser scheduling"
    echo "  - Actual TOSA operators (ABS, ADD)"
    echo "  - Serialization/deserialization"
    echo "  - Memory management"
    echo ""
else
    echo ""
    echo "ℹ️  Full TOSA example could not be built (requires reference model)"
    echo ""
    # echo "To build the full example:"
    # echo "1. Build the reference model first:"
    # echo "   mkdir build && cd build"
    # echo "   cmake .. && make"
    # echo "   cd .."
    # echo ""
    # echo "2. Then run this script again"
    # echo ""
fi

# echo "=== Build Summary ==="
# if [ -f "simple_tensor_demo" ]; then
#     echo "✓ simple_tensor_demo - Ready to run"
# fi
# if [ -f "test_graph_example" ]; then
#     echo "✓ test_graph_example - Ready to run"
# fi

# echo ""
# echo "Both examples demonstrate the same graph:"
# echo "  input1 -> ABS -> temp -> ADD(temp, input2) -> output"
# echo ""
# echo "Start with: ./simple_tensor_demo" 