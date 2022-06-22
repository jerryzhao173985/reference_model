#!/bin/bash -x
set -e

run_test()
{
   TEST=$1; shift
   FRAMEWORK=$1; shift

   echo "####  RUNNING EXAMPLE ${FRAMEWORK} ${TEST}"

   # Assumes the reference model is already built in ../build

   ../build/reference_model/tosa_reference_model \
      --test_desc=examples/${TEST}/flatbuffer-${FRAMEWORK}/desc.json \
      --ofm_file=out.npy
    python3 -c "import sys; import numpy as np; a = np.load(sys.argv[1]); b = np.load(sys.argv[2]); sys.exit(int((a != b).all()));" \
        examples/${TEST}/${FRAMEWORK}_result.npy \
        examples/${TEST}/flatbuffer-${FRAMEWORK}/out.npy
}

run_test test_add_1x4x4x4_f32 tf
run_test test_add_1x4x4x4_f32 tflite
run_test test_conv2d_1x1_1x32x32x8_f32_st11_padSAME_dilat11 tf
run_test test_conv2d_1x1_1x32x32x8_f32_st11_padSAME_dilat11 tflite
run_test test_conv2d_1x1_1x32x32x8_qi8_st11_padSAME_dilat11 tflite
