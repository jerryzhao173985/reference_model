#!/usr/bin/env python3

# Copyright (c) 2020, ARM Limited.
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


import argparse
import sys
import re
import os
import subprocess
import shlex
import json
import glob
import math
import queue
import threading
import traceback


from enum import IntEnum, Enum, unique
from datetime import datetime

# Include the ../shared directory in PYTHONPATH
parent_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(parent_dir, '..', 'scripts'))
sys.path.append(os.path.join(parent_dir, '..', 'scripts', 'xunit'))
import xunit
from tosa_serializer import *
from tosa_test_gen import TosaTestGen
import tosa

# Used for parsing a comma-separated list of integers in a string
# to an actual list of integers
def str_to_list(in_s):
    '''Converts a comma-separated list of string integers to a python list of ints'''
    lst = in_s.split(',')
    out_list = []
    for i in lst:
        out_list.append(int(i))
    return out_list

def auto_int(x):
    '''Converts hex/dec argument values to an int'''
    return int(x, 0)

def parseArgs():

    parser = argparse.ArgumentParser()
    parser.add_argument('-o', dest='output_dir', type=str, default='vtest',
                        help='Test output directory')

    parser.add_argument('--seed', dest='random_seed', default=42, type=int,
                        help='Random seed for test generation')

    parser.add_argument('--filter', dest='filter', default='', type=str,
                        help='Filter operator test names by this expression')

    parser.add_argument('-v', '--verbose', dest='verbose', action='count',
                        help='Verbose operation')

    # Constraints on tests
    parser.add_argument('--tensor-dim-range', dest='tensor_shape_range', default='1,64',
                        type=lambda x: str_to_list(x),
                        help='Min,Max range of tensor shapes')

    parser.add_argument('--max-batch-size', dest='max_batch_size', default=1, type=int,
                        help='Maximum batch size for NHWC tests')

    parser.add_argument('--max-conv-padding', dest='max_conv_padding', default=1, type=int,
                        help='Maximum padding for Conv tests')

    parser.add_argument('--max-conv-dilation', dest='max_conv_dilation', default=2, type=int,
                        help='Maximum dilation for Conv tests')

    parser.add_argument('--max-conv-stride', dest='max_conv_stride', default=2, type=int,
                        help='Maximum stride for Conv tests')

    parser.add_argument('--max-pooling-padding', dest='max_pooling_padding', default=1, type=int,
                        help='Maximum padding for pooling tests')

    parser.add_argument('--max-pooling-stride', dest='max_pooling_stride', default=2, type=int,
                        help='Maximum stride for pooling tests')

    parser.add_argument('--max-pooling-kernel', dest='max_pooling_kernel', default=2, type=int,
                        help='Maximum padding for pooling tests')

    parser.add_argument('--num-rand-permutations', dest='num_rand_permutations', default=6, type=int,
                        help='Number of random permutations for a given shape/rank for randomly-sampled parameter spaces')

    # Targetting a specific shape/rank/dtype
    parser.add_argument('--target-shape', dest='target_shapes', action='append', default=[], type=lambda x: str_to_list(x),
                        help='Create tests with a particular input tensor shape, e.g., 1,4,4,8 (may be repeated for tests that require multiple input shapes)')

    parser.add_argument('--target-rank', dest='target_ranks', action='append', default=None, type=lambda x: auto_int(x),
                        help='Create tests with a particular input tensor rank')

    parser.add_argument('--target-dtype', dest='target_dtypes', action='append', default=None, type=lambda x: dtype_str_to_val(x),
                        help='Create test with a particular DType (may be repeated)')

    args = parser.parse_args()

    return args

def main():


    args = parseArgs()

    ttg = TosaTestGen(args)

    testList = []
    for op in ttg.TOSA_OP_LIST:
        if re.match(args.filter + '.*', op):
            testList.extend(ttg.genOpTestList(op, shapeFilter=args.target_shapes, rankFilter=args.target_ranks, dtypeFilter=args.target_dtypes))

    print('{} matching tests'.format(len(testList)))
    for opName, testStr, dtype, shapeList, testArgs in testList:
        print(testStr)
        ttg.serializeTest(opName, testStr, dtype, shapeList, testArgs)
    print('Done creating {} tests'.format(len(testList)))


if __name__ == '__main__':
    exit(main())
