# Copyright (c) 2020-2022, ARM Limited.
# SPDX-License-Identifier: Apache-2.0
import numpy as np


class ArgGen:
    """Argument generator functions.  These functions take a shape and dtype to
    create arguments for an operator. Methods are prefixed with 'ag' to make
    search easy."""

    def __init__(self):
        pass

    @staticmethod
    def agNone(op, shapes, rng):
        """A trivial argument generator for operators that only take tensor
        operands"""
        return [("", [])]

    # Build the axis argument for operators where we want to iterate over N axes
    # as an argument
    @staticmethod
    def agAxes(op, shapes, rng):
        axes = []
        for i in range(-len(shapes), len(shapes), 1):
            if i >= 0:
                axes.append(["_axis_{}".format(i), [i]])
            else:
                axes.append(["_axis_m{}".format(-i), [i]])
        return axes

    # Build the axis LIST argument for operators that take an axis list.
    # This builds a list of each axis individually, plus one element
    # that contains a list of all axes.  Note that we need to pack the list in
    # an additional list so that it isn't exploded when being passed to the
    # build_operator function.
    # tensor_arg_count not used
    def agAxesList(op, shapes, rng):
        axes = ArgGen.agAxes(op, shapes, rng)
        axes_list = []
        for desc, a in axes:
            axes_list.append([desc, [a]])

        axes_list.append(["_axisall", [list(range(len(shapes)))]])
        axes_list.append(["_axisall_none", [None]])
        return axes_list

    def agAxesListKeepdims(op, shapes, rng):
        axes = ArgGen.agAxes(op, shapes, rng)
        axes_list = []
        for desc, a in axes:
            axes_list.append([desc + "_keep0", [a, False]])
            # avoid trying to reduce an axis of shape 1, as the TFL converter
            # will optimize away the entire reduction
            if (a[0] >= 0 and shapes[a[0]] != 1) or (
                a[0] < 0 and shapes[len(shapes) + a[0]] != 1
            ):
                axes_list.append([desc + "_keep1", [a, True]])

        axes_list.append(["_axisall_keep0", [list(range(len(shapes))), False]])
        axes_list.append(["_axisall_keep0_none", [None, False]])
        # another instance where the reduce gets optimized out.
        if len(shapes) != 1:
            axes_list.append(["_axisall_keep1", [list(range(len(shapes))), True]])
            axes_list.append(["_axisall_keep1_none", [None, True]])
        # no longer test axis empty, as TFL converter optimizes the reduce out
        return axes_list

    # conv2d argument generators build the TF constants
    def agConv2d(op, shapes, rng):
        arg_list = []

        # Must be rank 4
        if len(shapes) < 4:
            return arg_list

        filter_h, filter_w = op["filter"]

        # strides, padding, dilations,
        for stride_h in [1, 2]:
            for stride_w in [1, 2]:
                for padding in ["SAME", "VALID"]:
                    for dilation_h in [1, 2]:
                        for dilation_w in [1, 2]:

                            # Disqualify argument combinations that would cause
                            # an illegal convolution

                            if (padding == "VALID") and (
                                (shapes[1] - (filter_h - 1) * 2 - dilation_h) <= 0
                                or (shapes[2] - (filter_w - 1) * 2 - dilation_w) <= 0
                            ):
                                continue

                            if (
                                (shapes[1] - 1 - (filter_h - 1) * dilation_h) % stride_h
                                != 0
                            ) or (
                                (shapes[2] - 1 - (filter_w - 1) * dilation_w) % stride_w
                                != 0
                            ):
                                # Not an exact integer output
                                continue

                            arg_list.append(
                                [
                                    "_st{}{}_pad{}_dilat{}{}".format(
                                        stride_h,
                                        stride_w,
                                        padding,
                                        dilation_h,
                                        dilation_w,
                                    ),
                                    [
                                        [stride_h, stride_w],
                                        padding,
                                        [dilation_h, dilation_w],
                                    ],
                                ]
                            )
        return arg_list

    # conv2d argument generators build the TF constants
    def agDepthwiseConv2d(op, shapes, rng):
        arg_list = []

        # Must be rank 4
        if len(shapes) < 4:
            return arg_list

        filter_h, filter_w = op["filter"]

        # strides, padding, dilations, Depthwise conv2d is the same as conv2d
        # except that strides in h/w must be the same and the argument must be
        # formatted as [1, stride_h, stride_w, 1] in TF.
        for stride in [1, 2]:
            for padding in ["SAME", "VALID"]:
                for dilation_h in [1, 2]:
                    for dilation_w in [1, 2]:

                        # Disqualify argument combinations that would cause an illegal
                        # convolution

                        if (padding == "VALID") and (
                            (shapes[1] - (filter_h - 1) * 2 - dilation_h) <= 0
                            or (shapes[2] - (filter_w - 1) * 2 - dilation_w) <= 0
                        ):
                            continue

                        # When dilation is used, stride must be 1x1 (TF rules)
                        if dilation_h > 1 or dilation_w > 1:
                            if stride > 1:
                                continue

                        # Dilation must evenly divide the tensor.  Some of our inputs
                        # intentionally use odd-sized tensors.
                        if shapes[1] % dilation_h != 0 or shapes[2] % dilation_w != 0:
                            continue

                        if (
                            (shapes[1] - 1 - (filter_h - 1) * dilation_h) % stride != 0
                        ) or (
                            (shapes[2] - 1 - (filter_w - 1) * dilation_w) % stride != 0
                        ):
                            # Not an exact integer output
                            continue

                        arg_list.append(
                            [
                                "_st{}{}_pad{}_dilat{}{}".format(
                                    stride, stride, padding, dilation_h, dilation_w
                                ),
                                [
                                    [1, stride, stride, 1],
                                    padding,
                                    [dilation_h, dilation_w],
                                ],
                            ]
                        )
        return arg_list

    # conv2d argument generators build the TF constants
    def agTransposeConv2d(op, shapes, rng):
        arg_list = []

        # Must be rank 4
        if len(shapes) < 4:
            return arg_list

        filter_h, filter_w = op["filter"]

        # strides, padding, dilations,
        for stride_h in [1, 2]:
            for stride_w in [1, 2]:
                for padding in ["SAME", "VALID"]:
                    if padding == "SAME":
                        out_height = (shapes[1]) * stride_h
                        out_width = (shapes[2]) * stride_w
                    else:  # padding == 'VALID'
                        out_height = (shapes[1] - 1) * stride_h + filter_h
                        out_width = (shapes[2] - 1) * stride_w + filter_w

                    output_shape = [shapes[0], out_height, out_width, shapes[3] * 2]
                    arg_list.append(
                        [
                            "_st{}{}_pad{}".format(stride_h, stride_w, padding),
                            [output_shape, [stride_h, stride_w], padding],
                        ]
                    )
        return arg_list

    def agPooling(op, shapes, rng):
        arg_list = []

        # Must be rank 4
        if len(shapes) < 4:
            return arg_list

        for stride_h in [1, 2]:
            for stride_w in [1, 2]:
                for kernel_h in [1, 2]:
                    for kernel_w in [1, 2]:
                        for padding in ["SAME", "VALID"]:

                            if (padding == "VALID") and (
                                (shapes[1] % (kernel_h * stride_h) > 0)
                                or (shapes[2] % (kernel_w * stride_w) > 0)
                                or (shapes[1] <= kernel_h)
                                or (shapes[2] <= kernel_w)
                            ):
                                continue

                            if (padding == "SAME") and (
                                (shapes[1] < kernel_h) or (shapes[2] < kernel_w)
                            ):
                                continue

                            if ((shapes[1] - kernel_h) % stride_h != 0) or (
                                (shapes[2] - kernel_w) % stride_w != 0
                            ):
                                # Not an exact integer output
                                continue

                            arg_list.append(
                                [
                                    "_st{}{}_pad{}_kern{}{}".format(
                                        stride_h, stride_w, padding, kernel_h, kernel_w
                                    ),
                                    [
                                        [stride_h, stride_w],
                                        [kernel_h, kernel_w],
                                        padding,
                                    ],
                                ]
                            )
        return arg_list

    def getFactors(val, start=1):
        factors = []
        for i in range(start, int(np.sqrt(val))):
            if (val % i) == 0:
                factors.append(i)

        return factors

    def agReshape(op, shapes, rng):
        # This is slow code.  Fortunately, the numbers involved are small
        arg_list = []

        total_elements = 1
        for s in shapes:
            total_elements *= s

        # Find integer factors of this shape
        factors = ArgGen.getFactors(total_elements)

        for rank in range(1, len(shapes) + 1):
            if len(factors) < rank:
                break

            new_shape = []
            remaining_elements = total_elements

            # Randomly shuffle the factors and iteratively pick from the factors
            # of the remaining elements
            shuffled_factors = rng.permutation(factors)
            for i in range(rank):
                # Pick rank - 1 factors
                new_shape.append(shuffled_factors[0])
                remaining_elements = remaining_elements // shuffled_factors[0]
                shuffled_factors = rng.permutation(
                    ArgGen.getFactors(remaining_elements)
                )
            new_shape.append(remaining_elements)

            # Don't do no-op reshapes because TFLite optimizes out the op
            if new_shape == list(shapes):
                continue

            arg_list.append(["_rank{}".format(rank), [new_shape]])

        return arg_list

    def agTranspose(op, shapes, rng):
        arg_list = []

        # Must have at least two dimensions to transpose
        if (len(shapes)) < 2:
            return arg_list

        # Pick a bunch of random permutations
        range_arr = np.arange(len(shapes))
        for i in range(len(shapes)):
            perm = rng.permutation(range_arr).astype(np.int32)
            # print('\n shape {} permute{} perm: {} arr: {}'.format(shapes, i,
            # perm, range_arr))
            if np.allclose(perm, range_arr):
                print("skipped")
                continue
            arg_list.append(["_permute{}".format(i), [perm]])

        return arg_list

    def agSlice(op, shapes, rng):
        arg_list = []

        rank = len(shapes)

        if rank == 1 and shapes[0] == 1:
            return arg_list

        for i in range(4):
            # Pick a few random start points, axes, and strides
            start = np.empty((rank), dtype=int)
            size = np.empty((rank), dtype=int)
            for j in range(rank):
                if shapes[j] > 2:
                    start[j] = rng.integers(0, shapes[j] - 2)
                    # print('j = {}: {} - {} - 1: {}'.format(j, shapes[j],
                    # start[j], shapes[j] - start[j] - 1))
                    size[j] = rng.integers(1, shapes[j] - start[j] - 1)
                else:
                    start[j] = 0
                    size[j] = shapes[j]

                arg_list.append(["_perm{}".format(i), [start, size]])

        return arg_list

    def agStridedSlice(op, shapes, rng):
        arg_list = []

        rank = len(shapes)

        # Reference model is limited to rank=6 internally right now
        if rank > 3:
            return arg_list

        if rank == 1 and shapes[0] == 1:
            return arg_list

        for i in range(4):
            # Pick a few random begin points, axes, and strides
            begin = np.empty((rank), dtype=int)
            end = np.empty((rank), dtype=int)
            strides = np.empty((rank), dtype=int)

            begin_mask = rng.integers(0, (1 << (rank - 1)))
            end_mask = rng.integers(0, (1 << (rank - 1)))

            for j in range(rank):

                if begin_mask & (1 << j) or shapes[j] < 2:
                    begin[j] = 0
                else:
                    begin[j] = rng.integers(0, shapes[j] - 1)

                if end_mask & (1 << j) or shapes[j] < 2 or (begin[j] + 2) >= shapes[j]:
                    end[j] = shapes[j]
                else:
                    end[j] = rng.integers(begin[j] + 1, shapes[j] - 1)

                possible_stride = ArgGen.getFactors(end[j] - begin[j], 2)

                if not possible_stride:
                    strides[j] = 1
                else:
                    strides[j] = rng.choice(possible_stride)

            # Randomly set the masks, except ellipsis_mask and new_axis_mask
            # which must be zero for now For begin/end mask this to work,
            # strides must be adjusted to still be divsible...
            ellipsis_mask = 0
            new_axis_mask = 0

            # if rng.choice([0, 1]) and rank > 1:
            #    new_axis_mask = 1 << rng.integers(0, rank - 1)
            # else:
            #    new_axis_mask = 0

            if rng.choice([0, 1]) and rank > 1:
                shrink_axis_mask = 1 << rng.integers(0, rank - 1)
            else:
                shrink_axis_mask = 0

            # Only one of these bits may be set.  Prefer shrink_axis_mask
            new_axis_mask = new_axis_mask & ~shrink_axis_mask

            arg_list.append(
                [
                    "_perm{}".format(i),
                    [
                        begin,
                        end,
                        strides,
                        begin_mask,
                        end_mask,
                        ellipsis_mask,
                        new_axis_mask,
                        shrink_axis_mask,
                    ],
                ]
            )

            # print('Shape: {} begin={} end={} strides={} begin_mask={:x}
            # end_mask={:x} new_axis_mask={:x} shrink_mask={:x}'.format(shapes,
            # begin, end, strides, begin_mask, end_mask, new_axis_mask,
            # shrink_axis_mask))

        return arg_list

    # tf.stack axis can be [0, rank(input)]
    def agStack(op, shapes, rng):
        axes = []
        for i in range(len(shapes) + 1):
            axes.append(["_axis{}".format(i), [i]])
        return axes

    def agPad(op, shapes, rng):
        arg_list = []

        rank = len(shapes)
        for left in range(3):
            for right in range(3):
                paddings = np.zeros((rank, 2), dtype=np.int32)
                for d in range(rank):
                    paddings[d, 0] = left
                    paddings[d, 1] = right

                    arg_list.append(["_pad{}{}".format(left, right), [paddings]])
        return arg_list

    def agFill(op, shapes, rng):
        values = []
        for i in range(4):
            value = rng.integers(0, 10, dtype=np.int32)
            values.append(["_value{}".format(value), [shapes, value]])
        return values

    def getValuesToSum(total, rng):
        # Get a list of random integers that sum up to 'total'
        vals = []

        # np.random.randint() min and max to be different, so if the remainder
        # is 1, give up
        while total > 1:
            vals.append(rng.integers(1, total))
            total = total - vals[-1]

        if total == 1:
            vals.append(1)

        return vals

    def agSplit(op, shapes, rng):
        arg_list = []

        rank = len(shapes)

        # Shuffle the random number generator a few more times to get
        # a better range of axes across shapes
        for i in range(rank):
            for j in range(shapes[i]):
                rng.integers(shapes[i])

        for i in range(3):
            # Need to generate tests for both the num_splits and size_vector versions.
            axis = rng.choice(np.arange(0, rank))

            # For num_splits, get a few divisors of the given axis
            divs = ArgGen.getFactors(shapes[axis], 2)

            if divs:
                # Get no more than 2 samples
                splits = list(rng.choice(divs, size=2))

                for s in splits:
                    arg_list.append(
                        ["_split{}_axis{}".format(int(s), axis), [int(s), axis]]
                    )

            # For vector splits, get a list of integers that sum up to the axis size
            vals = ArgGen.getValuesToSum(shapes[axis], rng)

            if len(vals) > 1:
                arg_list.append(["_splitv_axis{}".format(axis), [vals, axis]])

        return arg_list

    def agTile(op, shapes, rng):
        arg_list = []

        rank = len(shapes)

        # create 1D multiples list
        multiples = list()
        for i in range(rank):
            multiples.append(rng.integers(1, 4))

        multiples_str = "x".join(list(str(i) for i in multiples))

        arg_list.append(["_tile_{}".format(multiples_str), [multiples]])

        return arg_list

    def agGather(op, shapes, rng):
        args = []
        for batch_dims in range(len(shapes) - 1):
            for axis in range(batch_dims, len(shapes)):
                # indices value must be within [0, shapes[i])

                # Create an arbitrary shape for the indices
                indices_rank = rng.integers(batch_dims + 1, 4)
                indices_shape = rng.integers(1, 8, size=indices_rank)

                # Copy in the batch dimensions because they must match
                for b in range(batch_dims):
                    indices_shape[b] = shapes[b]

                # Calculate total element count
                indices_size = 1
                for j in range(indices_rank):
                    indices_size = indices_shape[j] * indices_size

                indices = rng.integers(0, shapes[axis], indices_size, np.int32).reshape(
                    indices_shape
                )

                args.append(
                    [
                        "_batchdims_{}_axis_{}".format(batch_dims, axis),
                        [indices, batch_dims, axis],
                    ]
                )
        return args

    def agGatherND(op, shapes, rng):
        args = []

        for N in range(1, len(shapes) - 1):
            # Rank includes the N dimension
            indices_rank = rng.integers(2, 4, size=1)[0]
            indices_shape = []

            indices_shape = rng.integers(1, 8, size=indices_rank)
            indices_shape[-1] = N

            indices_count = 1
            for i in range(indices_rank - 1):
                indices_count = indices_count * indices_shape[i]

            indices_list = np.zeros(shape=(indices_count, N), dtype=np.int32)

            for i in range(indices_count):
                for j in range(N):
                    indices_list[i, j] = rng.integers(0, shapes[j], size=1)[0]

            indices = indices_list.reshape(indices_shape)

            args.append(["_n{}".format(N), [indices]])

        return args

    def agScatterND(op, shapes, rng):
        args = []

        # ScatterND has to generate a constant shapes tensor, indices
        # tensor, and a tensor of updates.  Unforunately, the updates
        # need to be a size that's based on the N generated in this
        # function and the dtype known only in the TensorGen function,
        # but not in ArgGen.
        #
        # There are many bad ways to solve this and we'll choose the
        # least of the evils which still gives reasonable coverage of
        # the possible operand shapes.
        for N in range(1, len(shapes)):
            # Rank includes the N dimension
            indices_rank = rng.integers(2, 4, size=1)[0]
            indices_shape = []

            indices_shape = rng.integers(1, 8, size=indices_rank)
            indices_shape[-1] = N

            # Store the Shapes, and the indicies value tensor as arguments.
            args.append(["_n{}".format(N), [shapes, indices_shape, N, rng]])

        return args

    def agSpaceToBatch(op, shapes, rng):
        batch_rank = 1
        channel_rank = 1
        block_rank = len(shapes) - batch_rank - channel_rank

        # must have at least rank 1 (M) block
        if block_rank < 1:
            return []

        args = []
        block_shape = []
        padding_shape = []

        for i in range(block_rank):
            block_size = 2
            padding_size = block_size - (shapes[i + 1] % block_size)
            block_shape.append(block_size)
            padding_shape.append([0, padding_size])

        args.append(["_blockrank_{}".format(block_rank), [block_shape, padding_shape]])
        return args

    def agBatchToSpace(op, shapes, rng):
        batch_rank = 1
        channel_rank = 1
        block_rank = len(shapes) - batch_rank - channel_rank

        # must have at least rank 1 (M) block
        if block_rank < 1:
            return []

        args = []
        block_shape = []
        padding_shape = []
        block_prod = 1

        for i in range(block_rank):
            block_size = 2
            block_prod = block_prod * block_size
            crop_size = 0
            block_shape.append(block_size)
            padding_shape.append([0, crop_size])

        # batch / prod(block_shape[i]) must be integer
        # transpose to swap depth and batch. so shape[-1] would be batch dim
        if shapes[-1] % block_prod == 0:
            args.append(
                ["_blockrank_{}".format(block_rank), [block_shape, padding_shape]]
            )

        return args

    def agSpaceToDepth(op, shapes, rng):
        # must be rank 4 input tensor
        if len(shapes) != 4:
            return []

        block_size = 2

        # spatial dimension must be divisible by block_size
        if shapes[1] % block_size != 0 or shapes[2] % block_size != 0:
            return []

        args = []
        args.append(["_blocksize_{}".format(block_size), [block_size]])

        return args

    def agDepthToSpace(op, shapes, rng):
        # must be rank 4 input tensor
        if len(shapes) != 4:
            return []

        block_size = 2
        # depth dimension must be divisible by block_size * block_size
        if shapes[3] % (block_size * block_size) != 0:
            return []

        args = []
        args.append(["_blocksize_{}".format(block_size), [block_size]])

        return args

    def agFakequant(op, shapes, rng):
        args = []
        for num_bits in [8, 16]:
            for narrow in [False, True]:
                args.append(
                    ["_bits{}_narrow{}".format(num_bits, narrow), [num_bits, narrow]]
                )

        return args

    def agShift(op, shapes, rng):
        args = []

        for shift in rng.integers(0, 32, size=8):
            args.append(["_shift{}".format(shift), [shift]])

        return args

    def agFloat(op, shapes, rng):
        args = []

        i = 0
        for alpha in np.float32(rng.random(size=2)):
            args.append(["_{}".format(i), [alpha]])

        return args

    # Similar to agAxes, but tf.OneHot only allow axis from [-1, rank(input)]
    def agOneHot(op, shapes, rng):
        axes = []
        for i in range(-1, len(shapes) + 1, 1):
            if i >= 0:
                axes.append(["_axis_{}".format(i), [i]])
            else:
                axes.append(["_axis_m{}".format(-i), [i]])
        return axes
