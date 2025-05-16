#!/usr/bin/env python3
# Copyright (c) 2020-2025, ARM Limited.
# SPDX-License-Identifier: Apache-2.0
import argparse
import os
import re
import traceback

import numpy as np

# All of the available supported frameworks
AVAILABLE_FRAMEWORKS = []

try:
    #  Level | Level for Humans | Level Description
    # -------|------------------|------------------------------------
    #  0     | DEBUG            | [Default] Print all messages
    #  1     | INFO             | Filter out INFO messages
    #  2     | WARNING          | Filter out INFO & WARNING messages
    #  3     | ERROR            | Filter out all messages
    # Filter TensorFlow debug message except errors
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

    # Flake8 E402 - ignore imports not at top of file to allow os.environ setting
    import tensorflow as tf  # noqa: E402
    from frameworks.tf.tf_op_list import TF_OP_LIST  # noqa: E402
    from frameworks.write_test_json import write_test_json_tf  # noqa: E402
    from frameworks.test_gen_utils import get_tf_dtype  # noqa: E402

    from tensorflow.lite.python.interpreter import OpResolverType  # noqa: E402

    AVAILABLE_FRAMEWORKS.append("tf")
    AVAILABLE_FRAMEWORKS.append("tflite")
except ImportError:
    print(
        "Cannot import TensorFlow in `tosa_verif_framework_generator`. Skipping TF/TFL tests"
    )

try:
    # Flake8 E402 - ignore imports not at top of file to allow os.environ setting
    from torch_mlir import fx  # noqa: E402
    from frameworks.torch.torch_op_list import TORCH_OP_LIST  # noqa: E402
    from frameworks.write_test_json import write_test_json_torch  # noqa: E402
    from frameworks.test_gen_utils import get_torch_dtype  # noqa: E402

    AVAILABLE_FRAMEWORKS.append("torch")
except ImportError:
    print(
        "Cannot import Torch-MLIR in `tosa_verif_framework_generator`. Skipping Torch tests"
    )

from frameworks.shape_list import shape_list  # noqa: E402
from frameworks.test_gen_utils import (  # noqa: E402
    QuantType,
    get_shape_str,
)  # noqa: E402


def gen_rand_shapes(args):
    """Overwrite the global shape list with a new list of random shapes"""
    global shape_list

    rng = np.random.default_rng(args.random_seed)

    # Don't let things get too big... cap the maximum volume, but let
    # an individual dimension be 1..47
    max_total_volume = 32 * 32 * 4

    shape_list = []
    # Only iterate over ranks 2, 3, 4, and 5
    for rank in range(2, 6):
        for n in range(args.random_shapes):
            new_shape = rng.integers(1, 48, size=rank)

            # Set the batch dimension on 4D or 5D objects to 1
            if rank == 4 or rank == 5:
                new_shape[0] = 1

            # Limit the total shape volume and throw out any
            # shapes that wouldn't leave at least size=2 in some non-batch dimension
            volume = 1
            skip_shape = False
            for i in range(rank):
                volume *= new_shape[i]

                # Reduce the shape, while it's larger than the maximum volume
                while volume > max_total_volume:
                    new_shape[i] = new_shape[i] // 2
                    volume = volume // 2

                    # Now an untenable dimension size?  Skip this one.
                    if new_shape[i] < 1:
                        skip_shape = True

            if not skip_shape:
                shape_list.append(tuple(new_shape))


# === TF/TFL test builders & runners === #


# Construct, run and save a whole tensorflow tf.function to a protobuf file
# or convert to .tflite if it's quantized unit test
def run_unit_test_tf(
    op_name,
    args,
    test_dir,
    curr_shape,
    addl_args,
    dtype,
    excluded_framework_list,
    quantized_inference_dtype,
    result_name,
    seed,
):
    try:
        op = TF_OP_LIST[op_name]
        op_fcn, tensor_gen_fcn, arg_gen_fcn = op["build_fcn"]

        # Get and seed a random number generator for this test
        rng = np.random.default_rng(seed)

        # return placeholders=(str: name, np.array: value)
        # consts=(str: name, np.array: value)
        placeholders, consts = (
            tensor_gen_fcn(op, curr_shape, dtype, rng, False)
            if tensor_gen_fcn.__name__ == "tgBFuzz"
            else tensor_gen_fcn(op, curr_shape, dtype, rng)
        )

        # if test doesn't have any placeholders/consts, terminated
        if len(placeholders) == 0 and len(consts) == 0:
            return True

        if not args.quiet:
            print("   {}              ".format(test_dir))

        try:
            os.mkdir(test_dir)
        except FileExistsError:
            pass

        const_nodes = [value for name, value in consts]

        num_placeholders = len(placeholders)
        # if test is quantized, create tensor quantization metadata info for
        # each input tensor, based on different quantized type
        if quantized_inference_dtype:
            is_quantized = True
            # TODO: support INT8 IFM x INT4 weight later
            if quantized_inference_dtype == QuantType.ALL_U8:
                qzero = [128] * num_placeholders
                numpy_dtype = [np.uint8] * num_placeholders
                tflite_inference_dtype = tf.uint8
            elif quantized_inference_dtype == QuantType.ALL_I8:
                qzero = [0] * num_placeholders
                numpy_dtype = [np.int8] * num_placeholders
                tflite_inference_dtype = tf.int8
            elif quantized_inference_dtype == QuantType.ALL_I16:
                qzero = [0] * num_placeholders
                numpy_dtype = [np.int16] * num_placeholders
                tflite_inference_dtype = tf.int16
            elif quantized_inference_dtype == QuantType.CONV_U8_U8:
                assert (
                    num_placeholders == 1
                ), "Unsupported number of placeholders for Convolution: {}".format(
                    num_placeholders
                )
                qzero = [128] * num_placeholders
                if num_placeholders == 2:
                    numpy_dtype = [np.uint8, np.uint8]
                else:
                    numpy_dtype = [np.uint8, np.uint8, np.int32]
                tflite_inference_dtype = tf.uint8
            elif quantized_inference_dtype == QuantType.CONV_I8_I8:
                assert (
                    num_placeholders == 1
                ), "Unsupported number of placeholders for Convolution: {}".format(
                    num_placeholders
                )
                qzero = [0] * num_placeholders
                if num_placeholders == 2:
                    numpy_dtype = [np.int8, np.int8]
                else:
                    numpy_dtype = [np.int8, np.int8, np.int32]
                tflite_inference_dtype = tf.int8
            elif quantized_inference_dtype == QuantType.CONV_I16_I8:
                assert (
                    num_placeholders == 1
                ), "Unsupported number of placeholders for Convolution: {}".format(
                    num_placeholders
                )
                if num_placeholders == 2:
                    qzero = [0, 0]
                    numpy_dtype = [np.int16, np.int8]
                else:
                    qzero = [0, 0, 0]
                    numpy_dtype = [
                        np.int16,
                        np.int8,
                        np.int64,
                    ]  # np.int64 to represent 40 bits accumulator
                tflite_inference_dtype = tf.int16
            else:
                raise Exception(
                    "Unsupported fakequant dtype: {}".format(quantized_inference_dtype)
                )

        else:
            is_quantized = False

        tf_model_filename = None
        tf_result_npy_filename = None
        tf_result_name = None

        tflite_model_filename = None
        tflite_result_npy_filename = None
        tflite_result_name = None

        placeholder_names = []
        placeholder_vals = []
        placeholder_signatures = ()
        placeholder_npy_filenames = []
        placeholder_shapes = []
        placeholder_dynamic = False

        for idx, (name, val) in enumerate(placeholders):
            input_shape = tuple(val.shape)

            try:
                dynamic_shape_dim_tuples = op["dynamic_shape_dim"]
                dim_tuple = dynamic_shape_dim_tuples[idx]
                input_shape = list(input_shape)

                # Set the dimensions of input that are listed in the builder profile to unknown.
                for dim in dim_tuple:
                    input_shape[dim] = None

                # When any dimension size is unknown, mark the placeholder as dynamic type.
                placeholder_dynamic = True

                addl_args.append(tuple(input_shape))
            except KeyError:
                pass

            placeholder_names.append(name)
            placeholder_signatures = placeholder_signatures + (
                tf.TensorSpec(shape=input_shape, dtype=val.dtype, name=name),
            )
            placeholder_npy_filenames.append("{}.npy".format(name.split(":")[0]))
            placeholder_shapes.append(val.shape)

        # Get test builder class
        fcn_node = op_fcn(*const_nodes, *addl_args, result_name)
        concrete_function = tf.function(input_signature=placeholder_signatures)(
            fcn_node.eval
        ).get_concrete_function()

        if is_quantized:
            assert dtype is tf.float32, "quantized test must come from float32 graph"

            # 1. Quantize float placeholder npy to quantized to feed the graph
            for idx, (name, val) in enumerate(placeholders):
                # we use np.amin()/np.amax() to determine dynamic range
                # for quantized test
                zeropoint = 0
                scale = 1.0
                if numpy_dtype[idx] != np.int64:
                    qmin = np.iinfo(numpy_dtype[idx]).min
                    qmax = np.iinfo(numpy_dtype[idx]).max
                    num_bits = np.iinfo(numpy_dtype[idx]).bits
                # 40 bit is represented as np.int64
                else:
                    num_bits = 40
                    qmin = -(1 << num_bits)
                    qmax = (1 << num_bits) - 1

                min_val = np.amin(val)
                max_val = np.amax(val)

                # for single value tensor, we set scale equal to the abs(value),
                # and fix zeropoint to 128
                # if val > 0, it'll be represented as 129,
                #    where val = (129 - 128) * val
                # if val < 0, it'll be represented as 127,
                #    where val = (127 - 128) * (-val)
                # if val == 0, it'll be represted as 128, with range [-128.0, 128.0]
                # and let quantized 1 represent the value
                # also adjust effective min/max consequently
                if max_val == min_val:
                    if max_val != 0:
                        scale = abs(max_val)
                    else:
                        scale = 1.0
                    min_val = float(qmin - qzero[idx]) * scale
                    max_val = float(qmax - qzero[idx]) * scale
                else:
                    scale = (max_val - min_val) / float(qmax - qmin)
                    if op_name == "squared_difference":
                        zeropoint = -int(round((-min_val) / scale)) + qmin
                    else:
                        zeropoint = int(round((-min_val) / scale)) + qmin

                # run through tf.fakequant first to assure quantization error aligned
                fakequant_val = tf.quantization.fake_quant_with_min_max_args(
                    val,
                    min=min_val,
                    max=max_val,
                    num_bits=num_bits,
                    name="gen_quant_npy",
                )

                quant_val = np.round(fakequant_val / scale) + zeropoint

                # very few unit tests after TF hash may/2020, this quantized
                # value for some reason exceed [0, 255] range
                saved_val = np.clip(quant_val, qmin, qmax).astype(numpy_dtype[idx])

                np.save(
                    os.path.join(test_dir, placeholder_npy_filenames[idx]),
                    saved_val,
                    False,
                )

                placeholder_vals.append(tf.convert_to_tensor(saved_val))

            # 2. Convert the model to quantized TFLite flatbuffer
            module = tf.Module()
            converter = tf.lite.TFLiteConverter.from_concrete_functions(
                [concrete_function], module
            )
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.experimental_new_converter = True

            # use MLIR-based post-quantizer
            converter.experimental_new_quantizer = True

            flag = (
                tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8  # noqa: E501
            )
            if tflite_inference_dtype == tf.int16:
                converter.target_spec.supported_ops = [flag]

            # Generator function for integer quantization of TFLiteConverter
            # which generates a few hundred input samples with the same order, type, and shape as the inputs,
            # to calibrate/estimate the range of the floating-point inputs.
            # For broadcast fuzzing tests, fuzzing needs to be disabled, otherwise, it causes a mismatch of
            # tensor shapes of inputs.
            def input_stats():
                for i in range(0, args.num_samples):
                    placeholders, _ = (
                        tensor_gen_fcn(op, placeholder_shapes[0], dtype, rng, True)
                        if tensor_gen_fcn == "tgBFuzz"
                        else tensor_gen_fcn(op, placeholder_shapes[0], dtype, rng)
                    )
                    yield [s[1] for s in placeholders]

            converter.representative_dataset = input_stats
            converter.inference_input_type = tflite_inference_dtype
            converter.inference_output_type = tflite_inference_dtype

            tflite_model = converter.convert()

            tflite_model_filename = "model.tflite"

            # Write out converted model to disk
            with open(os.path.join(test_dir, tflite_model_filename), "wb") as f:
                f.write(tflite_model)

        else:  # is_quantized is False
            # 1. Saved out numpy array directly
            for idx, (name, val) in enumerate(placeholders):
                placeholder_vals.append(tf.convert_to_tensor(val))

                # Complex tensors are expected to be repsesented by a
                # single floating point tensor of shape [?, ..., ?, 2].
                if val.dtype == np.complex64:
                    val_shape = val.shape + (2,)
                    val = val.view(np.float32)
                    val = val.reshape(val_shape)

                np.save(
                    os.path.join(test_dir, placeholder_npy_filenames[idx]), val, False
                )

            # 2.a Saved out .pb if framework includes tensorflow
            if "tf" not in excluded_framework_list:
                # Write out graph as protobuf to disk
                tf_model_filename = "model.pb"
                tf.io.write_graph(
                    concrete_function.graph, test_dir, tf_model_filename, True
                )

            # 2.b Saved out .tflite if framework includes tflite
            if "tflite" not in excluded_framework_list:
                # Convert the model to TFLite flatbuffer
                module = tf.Module()

                if op_name == "callonce" or op_name == "lstm_stateful":
                    converter = tf.lite.TFLiteConverter.from_concrete_functions(
                        [concrete_function], fcn_node
                    )
                else:
                    converter = tf.lite.TFLiteConverter.from_concrete_functions(
                        [concrete_function], module
                    )

                converter.experimental_new_converter = True

                # Even it's non-quantized int32 test, this needs to be set to tf.float32
                converter.inference_input_type = tf.float32
                converter.inference_output_type = tf.float32
                tflite_model = converter.convert()

                # Write out converted model to disk
                tflite_model_filename = "model.tflite"
                with open(os.path.join(test_dir, tflite_model_filename), "wb") as f:
                    f.write(tflite_model)

        # Get TF reference result if .pb is specified
        if tf_model_filename:
            tf_result_npy_filename = "tf_result.npy"
            tf_result = concrete_function(*placeholder_vals)
            np.save(os.path.join(test_dir, tf_result_npy_filename), tf_result, False)

            tf_result_name = result_name

        # Get TFLite inference result if .tflite is specified
        if tflite_model_filename:
            tflite_result_npy_filename = "tflite_result.npy"

            ops_with_optimized_only_kernel = ["elu", "ceil", "gather", "rfft2d"]

            if args.tflite_kernel_mode == "optimized" or (
                op_name in ops_with_optimized_only_kernel
            ):
                interpreter = tf.lite.Interpreter(
                    model_path=os.path.join(test_dir, tflite_model_filename)
                )
            elif args.tflite_kernel_mode == "reference":
                interpreter = tf.lite.Interpreter(
                    model_path=os.path.join(test_dir, tflite_model_filename),
                    experimental_op_resolver_type=OpResolverType.BUILTIN_REF,
                )
            else:
                assert 0, "unknown tflite interpreter mode {}".format(
                    args.tflite_kernel_mode
                )

            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()

            # Prototype dynamic_shape testing
            # Need to resize the input tensors to known shapes when evaluating
            for idx, val in enumerate(placeholder_vals):
                if len(placeholder_shapes[idx]) != 0:
                    interpreter.resize_tensor_input(
                        input_details[idx]["index"], placeholder_shapes[idx]
                    )
            interpreter.allocate_tensors()

            assert len(input_details) == len(
                placeholder_vals
            ), "number of placeholder mismatch"

            for idx, val in enumerate(placeholder_vals):
                interpreter.set_tensor(input_details[idx]["index"], val.numpy())

            interpreter.invoke()
            tflite_result = interpreter.get_tensor(output_details[0]["index"])

            np.save(
                os.path.join(test_dir, tflite_result_npy_filename), tflite_result, False
            )

            # Result tensor name would change after converting to TFLite flatbuffer
            # Overwrite the information from TFLite models directly.
            # Assume single result tensor now
            tflite_result_name = output_details[0]["name"]

        _, test_name = os.path.split(test_dir)

        # For specifying the number of variable tensors if the graph has any
        try:
            num_varaibles = op["num_variables"]
        except KeyError:
            num_varaibles = 0

        # Write out test descriptor
        write_test_json_tf(
            filename=os.path.join(test_dir, "test.json"),
            tf_model_filename=tf_model_filename,
            tf_result_npy_filename=tf_result_npy_filename,
            tf_result_name=tf_result_name,
            tflite_model_filename=tflite_model_filename,
            tflite_result_npy_filename=tflite_result_npy_filename,
            tflite_result_name=tflite_result_name,
            ifm_name=placeholder_names,
            ifm_file=placeholder_npy_filenames,
            ifm_shape=placeholder_shapes,
            ifm_dynamic=placeholder_dynamic,
            framework_exclusions=excluded_framework_list,
            quantized=is_quantized,
            test_name=test_name,
            num_variables=num_varaibles,
        )
    except Exception as e:
        msg = "Error running task: {}".format(e)
        print(msg)
        print("".join(traceback.format_exception(None, value=e, tb=e.__traceback__)))
        return False
    return True


def build_const_net_tf(
    args,
    curr_shape,
    op_name,
    dtype,
    excluded_framework_list,
    quantized_inference_dtype,
    result_name,
    seed,
    rng,
    filter,
    unit_test_args,
):
    if quantized_inference_dtype:
        quant_dtype = get_tf_dtype(quantized_inference_dtype)
        test_dir = "test_tf_{}_{}".format(
            op_name, get_shape_str(curr_shape, quant_dtype)
        )
    else:
        test_dir = "test_tf_{}_{}".format(op_name, get_shape_str(curr_shape, dtype))
    test_dir = os.path.join(args.output_dir, test_dir)

    # If the operator has an additional function to generate arguments, call it
    # here and iterate through the argument list that it generates
    op = TF_OP_LIST[op_name]
    op_fcn, tensor_gen_fcn, arg_gen_fcn = op["build_fcn"]

    try:
        rank_lo, rank_hi = op["rank"]
    except KeyError:
        # Set testing rank to (1, 4) in default.
        rank_lo = 1
        rank_hi = 4

    if len(curr_shape) not in range(rank_lo, rank_hi + 1):
        return

    if op_name == "left_shift" or op_name == "right_shift":
        addl_args_tuple = arg_gen_fcn(op, curr_shape, rng, dtype)
    else:
        addl_args_tuple = arg_gen_fcn(op, curr_shape, rng)

    for desc, addl_args in addl_args_tuple:
        # Only filter on the full test_name, not the output directory
        _, test_name = os.path.split(test_dir + desc)
        if not filter or filter.search(test_name):
            unit_test_args.append(
                [
                    op_name,
                    args,
                    test_dir + desc,
                    curr_shape,
                    addl_args,
                    dtype,
                    excluded_framework_list,
                    quantized_inference_dtype,
                    result_name,
                    seed,
                ]
            )


# python hash is not reproducible, create hash for our purpose
def op_name_hash(op_name):
    result = 0xDEADBEEF
    for ch in op_name:
        if result & 1:
            result = (ord(ch) << 24) ^ (result >> 1) ^ 0x82608EDB
        else:
            result = (ord(ch) << 24) ^ (result >> 1)

    return result


def generate_op_tests_tf(
    args, op_name, shape_list, result_name, filter, unit_test_args
):
    if not args.quiet:
        print(
            "Generating TF/TFL tests for {}                                        ".format(
                op_name
            )
        )

    op = TF_OP_LIST[op_name]

    # Seed the RNG so that we get the same random tests for each test each time
    # If the number of tests for a given generation function changes, the tests
    # for that operator may also change accordingly, but this will at least keep
    # down churn across operators.

    bounded_hash_val = (args.random_seed + op_name_hash(op_name)) % np.iinfo(
        np.int32
    ).max
    rng = np.random.default_rng(bounded_hash_val)

    # this is a dictionary with 'tf' and 'tflite' as key
    # and value being the data types we want to test under these framework

    if isinstance(op["types"], dict):
        try:
            tf_dtypes = op["types"]["tf"]
        except KeyError:
            tf_dtypes = []
        try:
            tflite_dtypes = op["types"]["tflite"]
        except KeyError:
            tflite_dtypes = []
    elif isinstance(op["types"], list):
        tf_dtypes = op["types"]
        tflite_dtypes = op["types"]

    tf_nonquantized_dtypes = tf_dtypes  # tf doesn't support quantized data types
    tflite_quantized_dtypes = []
    tflite_nonquantized_dtypes = []
    for dtype in tflite_dtypes:
        if isinstance(dtype, QuantType):
            tflite_quantized_dtypes.append(dtype)
        else:
            tflite_nonquantized_dtypes.append(dtype)

    nonquantized_dtypes_set = set(tf_nonquantized_dtypes).union(
        set(tflite_nonquantized_dtypes)
    )
    nonquantized_dtypes = list(nonquantized_dtypes_set)
    quantized_dtypes = tflite_quantized_dtypes

    # append custom_shapes or replace shape_list with custom_shapes
    try:
        custom_shapes = op["custom_shapes"]
        if custom_shapes["custom_shape_only"]:
            shape_list = custom_shapes["shape_list"]
        else:
            shape_list = shape_list.copy()
            shape_list.extend(custom_shapes["shape_list"])
    except KeyError:
        pass

    try:
        result_name = op["output_name"]
    except KeyError:
        pass

    # populate non quantized unit test arguments
    for dtype in nonquantized_dtypes:
        excluded_framework_set = set(AVAILABLE_FRAMEWORKS)
        if dtype in tf_nonquantized_dtypes:
            excluded_framework_set.remove("tf")
        if dtype in tflite_nonquantized_dtypes:
            excluded_framework_set.remove("tflite")
        excluded_framework_list = list(excluded_framework_set)

        for curr_shape in shape_list:
            build_const_net_tf(
                args,
                curr_shape,
                op_name,
                dtype,
                excluded_framework_list,
                None,
                result_name,
                bounded_hash_val,
                rng,
                filter,
                unit_test_args,
            )

    # populate quantized unit test arguments
    # must exclude 'tf' and source dtype being tf.float32
    for dtype in quantized_dtypes:
        for curr_shape in shape_list:
            build_const_net_tf(
                args,
                curr_shape,
                op_name,
                tf.float32,
                ["tf"],
                dtype,
                result_name,
                bounded_hash_val,
                rng,
                filter,
                unit_test_args,
            )

    return unit_test_args


def create_dynamic_op_lists_tf():
    """The templated operators are conv2d-style operators with a number of kernel
    sizes.  Since the operator is unchanged, we generate the range of kernel
    sizes here in this loop and remove the original templates from the list.

    This could be expanded to non-conv2d-style operators in the future."""

    # Dynamically create op lists for convolutions with a list of kernel sizes
    KERNELS = [
        [1, 1],
        [3, 3],
        [5, 5],
    ]

    # dim = [D, H, W]
    KERNELS_3D = [
        [1, 1, 1],
        [2, 3, 3],
        [3, 5, 5],
    ]

    TEMPLATE_LIST = [
        "conv2d",
        "conv2d_bias",
        "conv2d_relu",
        "conv2d_relu6",
        "conv2d_relu_n1_to_1",
        "conv2d_tanh",
        "depthwise_conv2d",
        "depthwise_conv2d_bias",
        "transpose_conv2d",
    ]

    TEMPLATE_LIST_CONV3D = [
        "conv3d",
        "conv3d_bias",
    ]

    for t in TEMPLATE_LIST:
        for k in KERNELS:
            testName = "{}_{}x{}".format(t, k[0], k[1])
            TF_OP_LIST[testName] = TF_OP_LIST["{}_TEMPLATE".format(t)].copy()
            TF_OP_LIST[testName]["filter"] = k
            TF_OP_LIST[testName]["template"] = False

    # The existing operators don't support the dimension of kernel that is higher than 2.
    for t in TEMPLATE_LIST_CONV3D:
        for k in KERNELS_3D:
            testName = "{}_{}x{}x{}".format(t, k[0], k[1], k[2])
            TF_OP_LIST[testName] = TF_OP_LIST["{}_TEMPLATE".format(t)].copy()
            TF_OP_LIST[testName]["filter"] = k
            TF_OP_LIST[testName]["template"] = False

    # Delete any templates after having created any dynamic ops
    # This is a two-pass operation because it's bad practice to delete
    # keys from dictionaries while iterating
    keyList = []
    for k in TF_OP_LIST:
        try:
            if TF_OP_LIST[k]["template"]:
                keyList.append(k)
                continue
        except KeyError:
            pass

    for k in keyList:
        del TF_OP_LIST[k]


# === Torch test builders & runners === #


# Construct, run and save a whole Torch module to a Torch MLIR file
def run_unit_test_torch(
    op_name,
    args,
    test_dir,
    curr_shape,
    addl_args,
    dtype,
    quantized_inference_dtype,
    seed,
):
    try:
        op = TORCH_OP_LIST[op_name]
        op_fcn, tensor_gen_fcn, arg_gen_fcn = op["build_fcn"]

        # Get and seed a random number generator for this test
        rng = np.random.default_rng(seed)

        # return placeholders=(str: name, np.array: value)
        # consts=(str: name, np.array: value)
        placeholders, consts = (
            tensor_gen_fcn(op, curr_shape, dtype, rng, False)
            if tensor_gen_fcn.__name__ == "tgBFuzz"
            else tensor_gen_fcn(op, curr_shape, dtype, rng)
        )

        # if test doesn't have any placeholders/consts, terminated
        if len(placeholders) == 0 and len(consts) == 0:
            return True

        if not args.quiet:
            print("   {}              ".format(test_dir))

        try:
            os.mkdir(test_dir)
        except FileExistsError:
            pass

        const_nodes = [value for name, value in consts]

        # TODO: add quantized types support
        if quantized_inference_dtype:
            is_quantized = True
        else:
            is_quantized = False

        torch_mlir_filename = None
        torch_result_npy_filename = None

        placeholder_names = []
        placeholder_vals = []
        placeholder_npy_filenames = []
        placeholder_shapes = []
        placeholder_annotations = [None]

        _, test_name = os.path.split(test_dir)

        for idx, (name, val) in enumerate(placeholders):
            placeholder_annotations.append((val.shape, dtype, True))

            placeholder_names.append(name)
            placeholder_npy_filenames.append("{}.npy".format(name.split(":")[0]))
            placeholder_shapes.append(val.shape)

        # Get test builder class
        fcn_node = op_fcn(*const_nodes, *addl_args)

        # Save output numpy array directly
        for idx, (name, val) in enumerate(placeholders):
            placeholder_vals.append(val)

            # Complex tensors are expected to be repsesented by a
            # single floating point tensor of shape [?, ..., ?, 2].
            if val.dtype == np.complex64:
                val_shape = val.shape + (2,)
                val = val.view(np.float32)
                val = val.reshape(val_shape)

            np.save(os.path.join(test_dir, placeholder_npy_filenames[idx]), val, False)

        # Create Torch MLIR with placeholder values using FX Importer (PyTorch 2.0)
        mlir_module = fx.export_and_import(
            fcn_node, *placeholder_vals, output_type="torch"
        )

        torch_mlir_filename = "torch.mlir"

        with open(os.path.join(test_dir, torch_mlir_filename), "w") as f:
            f.write(str(mlir_module))

        # Generate Torch reference output numpy
        torch_result = fcn_node.forward(*placeholder_vals)
        torch_result_npy_filename = "torch_result.npy"
        np.save(
            os.path.join(test_dir, torch_result_npy_filename),
            torch_result.detach().numpy(),
            False,
        )

        # For specifying the number of variable tensors if the graph has any
        try:
            num_varaibles = op["num_variables"]
        except KeyError:
            num_varaibles = 0

        # Write out test descriptor
        write_test_json_torch(
            filename=os.path.join(test_dir, "test.json"),
            torch_mlir_filename=torch_mlir_filename,
            torch_result_npy_filename=torch_result_npy_filename,
            ifm_name=placeholder_names,
            ifm_file=placeholder_npy_filenames,
            ifm_shape=placeholder_shapes,
            quantized=is_quantized,
            test_name=test_name,
            num_variables=num_varaibles,
        )
    except Exception as e:
        msg = "Error running task: {}".format(e)
        print(msg)
        print("".join(traceback.format_exception(None, value=e, tb=e.__traceback__)))
        return False
    return True


def build_const_net_torch(
    args,
    curr_shape,
    op_name,
    dtype,
    quantized_inference_dtype,
    seed,
    rng,
    filter,
    unit_test_args,
):
    if quantized_inference_dtype:
        quant_dtype = get_torch_dtype(quantized_inference_dtype)
        test_dir = "test_torch_{}_{}".format(
            op_name, get_shape_str(curr_shape, quant_dtype)
        )
    else:
        test_dir = "test_torch_{}_{}".format(op_name, get_shape_str(curr_shape, dtype))
    test_dir = os.path.join(args.output_dir, test_dir)

    # If the operator has an additional function to generate arguments, call it
    # here and iterate through the argument list that it generates
    op = TORCH_OP_LIST[op_name]
    op_fcn, tensor_gen_fcn, arg_gen_fcn = op["build_fcn"]

    try:
        rank_lo, rank_hi = op["rank"]
    except KeyError:
        # Set testing rank to (1, 4) in default.
        rank_lo = 1
        rank_hi = 4

    if len(curr_shape) not in range(rank_lo, rank_hi + 1):
        return

    addl_args_tuple = arg_gen_fcn(op, curr_shape, rng)

    for desc, addl_args in addl_args_tuple:
        # Only filter on the full test_name, not the output directory
        _, test_name = os.path.split(test_dir + desc)
        if not filter or filter.search(test_name):
            unit_test_args.append(
                [
                    op_name,
                    args,
                    test_dir + desc,
                    curr_shape,
                    addl_args,
                    dtype,
                    quantized_inference_dtype,
                    seed,
                ]
            )


def generate_op_tests_torch(args, op_name, shape_list, filter, unit_test_args):
    if not args.quiet:
        print(
            "Generating Torch tests for {}                                        ".format(
                op_name
            )
        )

    op = TORCH_OP_LIST[op_name]

    # Seed the RNG so that we get the same random tests for each test each time
    # If the number of tests for a given generation function changes, the tests
    # for that operator may also change accordingly, but this will at least keep
    # down churn across operators.

    bounded_hash_val = (args.random_seed + op_name_hash(op_name)) % np.iinfo(
        np.int32
    ).max
    rng = np.random.default_rng(bounded_hash_val)

    torch_dtypes = op["types"]

    # TODO: add support for Torch quantized types

    # append custom_shapes or replace shape_list with custom_shapes
    try:
        custom_shapes = op["custom_shapes"]
        if custom_shapes["custom_shape_only"]:
            shape_list = custom_shapes["shape_list"]
        else:
            shape_list = shape_list.copy()
            shape_list.extend(custom_shapes["shape_list"])
    except KeyError:
        pass

    # populate unit test arguments
    for dtype in torch_dtypes:
        for curr_shape in shape_list:
            build_const_net_torch(
                args,
                curr_shape,
                op_name,
                dtype,
                None,
                bounded_hash_val,
                rng,
                filter,
                unit_test_args,
            )

    return unit_test_args


def create_dynamic_op_lists_torch():
    """The templated operators are conv2d-style operators with a number of kernel
    sizes.  Since the operator is unchanged, we generate the range of kernel
    sizes here in this loop and remove the original templates from the list.

    This could be expanded to non-conv2d-style operators in the future."""

    # Dynamically create op lists for convolutions with a list of kernel sizes
    KERNELS = [
        [1, 1],
        [3, 3],
        [5, 5],
    ]

    TEMPLATE_LIST = ["conv2d", "conv2d_bias", "maxpool2d", "avg_pool2d"]

    for t in TEMPLATE_LIST:
        for k in KERNELS:
            testName = "{}_{}x{}".format(t, k[0], k[1])
            TORCH_OP_LIST[testName] = TORCH_OP_LIST["{}_TEMPLATE".format(t)].copy()
            TORCH_OP_LIST[testName]["filter"] = k
            TORCH_OP_LIST[testName]["template"] = False

    # Delete any templates after having created any dynamic ops
    # This is a two-pass operation because it's bad practice to delete
    # keys from dictionaries while iterating
    keyList = []
    for k in TORCH_OP_LIST:
        try:
            if TORCH_OP_LIST[k]["template"]:
                keyList.append(k)
                continue
        except KeyError:
            pass

    for k in keyList:
        del TORCH_OP_LIST[k]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--framework",
        dest="framework",
        choices=["tf", "torch", "all"],
        default="all",
        help="Framework to generate tests for (tf, torch, or all)",
    )
    parser.add_argument(
        "--seed", dest="random_seed", default=42, type=int, help="Random seed"
    )
    parser.add_argument(
        "--random-shapes",
        dest="random_shapes",
        default=0,
        type=int,
        help=(
            "Use N random shapes of each rank for generating tests,"
            "seeded with random seed"
        ),
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        dest="output_dir",
        default=".",
        type=str,
        help="Test output directory path prefix",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        dest="quiet",
        default=False,
        action="store_true",
        help="Do not print test names",
    )
    parser.add_argument(
        "-j", "--jobs", dest="jobs", type=int, default=1, help="Number of parallel jobs"
    )
    parser.add_argument(
        "-m",
        "--tflite-kernel-mode",
        dest="tflite_kernel_mode",
        type=str,
        choices=["reference", "optimized"],
        default="reference",
        help="TFLite interpreter kernel mode",
    )
    parser.add_argument(
        "--num-samples",
        dest="num_samples",
        default=200,
        type=int,
        help="Number of input samples for post-training quantization",
    )
    parser.add_argument(
        "--filter",
        dest="filter",
        default="",
        type=str,
        help="Filter test names by this expression",
    )
    args = parser.parse_args()

    # Turn the filter into a re object if present
    filter = None
    if args.filter != "":
        filter = re.compile(args.filter)

    # Autodetect CPU count
    if args.jobs <= 0:
        args.jobs = os.cpu_count()

    if "tf" in AVAILABLE_FRAMEWORKS:
        # Disable TF info messages
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    try:
        os.makedirs(args.output_dir)
    except FileExistsError:
        pass

    if args.random_shapes:
        gen_rand_shapes(args)

    # Determine the specified frameworks
    if args.framework != "all" and args.framework not in AVAILABLE_FRAMEWORKS:
        raise ValueError(f"Requested framework `{args.framework}` not available")

    if args.framework == "all":
        frameworks = list(AVAILABLE_FRAMEWORKS)
    elif args.framework == "torch":
        frameworks = ["torch"]
    else:
        frameworks = ["tf", "tflite"]

    tf_args = []
    torch_args = []
    errors = 0

    # Create dynamic ops, generate tests, and run tests
    if "tf" in frameworks:
        create_dynamic_op_lists_tf()

        for op in TF_OP_LIST:
            generate_op_tests_tf(args, op, shape_list, "result", filter, tf_args)

        for t in tf_args:
            if not run_unit_test_tf(*t):
                errors = errors + 1
    if "torch" in frameworks:
        create_dynamic_op_lists_torch()

        for op in TORCH_OP_LIST:
            generate_op_tests_torch(args, op, shape_list, filter, torch_args)

        for t in torch_args:
            if not run_unit_test_torch(*t):
                errors = errors + 1

    if not args.quiet:
        print("\nAll tasks done - with {} errors".format(errors))

    return 1 if errors else 0


if __name__ == "__main__":
    exit(main())
