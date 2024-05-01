# Copyright (c) 2021-2024, ARM Limited.
# SPDX-License-Identifier: Apache-2.0
"""Select generated tests."""
import argparse
import itertools
import json
import logging
import re
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List

logging.basicConfig()
logger = logging.getLogger("test_select")


def expand_params(permutes: Dict[str, List[Any]], others: Dict[str, List[Any]]):
    """Generate permuted combinations of a dictionary of values and combine with others.

    permutes: a dictionary with sequences of values to be fully permuted
    others: a dictionary with sequences of values not fully permuted, but all used

    This yields dictionaries with one value from each of the items in permutes,
    combined with one value from each of the items in others.

    Example 1:

        permutes = {"a": [1, 2], "b": [3, 4]}
        others = {"c": [5, 6, 7], "d" [True, False]}

        generates:

            [
                {"a": 1, "b": 3, "c": 5, "d": True},
                {"a": 1, "b": 4, "c": 6, "d": False`},
                {"a": 2, "b": 3, "c": 7, "d": True},
                {"a": 2, "b": 4, "c": 5, "d": False`},
            ]

    Example 2:

        permutes = {"a": [1, 2], "b": [3, 4]}
        others = {"c": [5, 6, 7, 8, 9], "d" [True, False]}

        generates:

            [
                {"a": 1, "b": 3, "c": 5, "d": True},
                {"a": 1, "b": 4, "c": 6, "d": False},
                {"a": 2, "b": 3, "c": 7, "d": True},
                {"a": 2, "b": 4, "c": 8, "d": False},
                {"a": 1, "b": 3, "c": 9, "d": True},
            ]

    Raises:
        ValueError if any item is in both permutes and others
    """
    for k in permutes:
        if k in others:
            raise ValueError(f"item conflict: {k}")

    p_keys = []
    p_vals = []
    # if permutes is empty, p_permute_len should be 0, but we leave it as 1
    # so we return a single, empty dictionary, if others is also empty
    p_product_len = 1
    # extract the keys and values from the permutes dictionary
    # and calulate the product of the number of values in each item as we do so
    for k, v in permutes.items():
        p_keys.append(k)
        p_vals.append(v)
        p_product_len *= len(v)
    # create a cyclic generator for the product of all the permuted values
    p_product = itertools.product(*p_vals)
    p_generator = itertools.cycle(p_product)

    o_keys = []
    o_vals = []
    o_generators = []
    # extract the keys and values from the others dictionary
    # and create a cyclic generator for each list of values
    for k, v in others.items():
        o_keys.append(k)
        o_vals.append(v)
        o_generators.append(itertools.cycle(v))

    # The number of params dictionaries generated will be the maximumum size
    # of the permuted values and the non-permuted values from others
    max_items = max([p_product_len] + [len(x) for x in o_vals])

    # create a dictionary with a single value for each of the permutes and others keys
    for _ in range(max_items):
        params = {}
        # add the values for the permutes parameters
        # the permuted values generator returns a value for each of the permuted keys
        # in the same order as they were originally given
        p_vals = next(p_generator)
        for i in range(len(p_keys)):
            params[p_keys[i]] = p_vals[i]
        # add the values for the others parameters
        # there is a separate generator for each of the others values
        for i in range(len(o_keys)):
            params[o_keys[i]] = next(o_generators[i])
        yield params


class Operator:
    """Base class for operator specific selection properties."""

    # A registry of all Operator subclasses, indexed by the operator name
    registry = {}

    def __init_subclass__(cls, **kwargs):
        """Subclass initialiser to register all Operator classes."""
        super().__init_subclass__(**kwargs)
        cls.registry[cls.name] = cls

    # Derived classes must override the operator name
    name = None
    # Operators with additional parameters must override the param_names
    # NB: the order must match the order the values appear in the test names
    param_names = ["shape", "type"]

    # Working set of param_names - updated for negative tests
    wks_param_names = None

    def __init__(
        self,
        test_dir: Path,
        config: Dict[str, Dict[str, List[Any]]],
        negative=False,
        ignore_missing=False,
    ):
        """Initialise the selection parameters for an operator.

        test_dir: the directory where the tests for all operators can
            be found
        config: a dictionary with:
                "params" - a dictionary with mappings of parameter
                    names to the values to select (a sub-set of
                    expected values for instance)
                "permutes" - a list of parameter names to be permuted
                "preselected" - a list of dictionaries containing
                    parameter names and pre-chosen values
                "sparsity" - a dictionary of parameter names with a
                    sparsity value
                "full_sparsity" - "true"/"false" to use the sparsity
                    value on permutes/params/preselected
                "exclude_patterns" - a list of regex's whereby each
                    match will not be considered for selection.
                    Exclusion happens BEFORE test selection (i.e.
                    before permutes are applied).
                "errorifs" - list of ERRORIF case names to be selected
                    after exclusion (negative tests)
        negative: bool indicating if negative testing is being selected
            which filters for ERRORIF in the test name and only selects
            the first test found (ERRORIF tests)
        ignore_missing: bool indicating if missing tests should be ignored

        EXAMPLE CONFIG (with non-json comments):
            "params": {
                "output_type": [
                    "outi8",
                    "outb"
                ]
            },
            "permutes": [
                "shape",
                "type"
            ],
            "sparsity": {
                "pad": 15
            },
            "preselected": [
                {
                    "shape": "6",
                    "type": "i8",
                    "pad": "pad00"
                }
            ],
            "exclude_patterns": [
                # Exclude positive (not ERRORIF) integer tests
                "^((?!ERRORIF).)*_(i8|i16|i32|b)_out(i8|i16|i32|b)",
                # Exclude negative (ERRORIF) i8 test
                ".*_ERRORIF_.*_i8_outi8"
            ],
            "errorifs": [
                "InputZeroPointNotZero"
            ]
        """
        assert isinstance(
            self.name, str
        ), f"{self.__class__.__name__}: {self.name} is not a valid operator name"

        self.negative = negative
        self.ignore_missing = ignore_missing
        self.wks_param_names = self.param_names.copy()
        if self.negative:
            # need to override positive set up - use "errorifs" config if set
            # add in errorif case before shape to support all ops, including
            # different ops like COND_IF and CONVnD etc
            index = self.wks_param_names.index("shape")
            self.wks_param_names[index:index] = ["ERRORIF", "case"]
            config["params"] = {x: [] for x in self.wks_param_names}
            config["params"]["case"] = (
                config["errorifs"] if "errorifs" in config else []
            )
            config["permutes"] = []
            config["preselected"] = {}

        self.params = config["params"] if "params" in config else {}
        self.permutes = config["permutes"] if "permutes" in config else []
        self.sparsity = config["sparsity"] if "sparsity" in config else {}
        self.full_sparsity = (
            (config["full_sparsity"] == "true") if "full_sparsity" in config else False
        )
        self.preselected = config["preselected"] if "preselected" in config else {}
        self.exclude_patterns = (
            config["exclude_patterns"] if "exclude_patterns" in config else []
        )
        self.non_permutes = [x for x in self.wks_param_names if x not in self.permutes]
        logger.info(f"{self.name}: permutes={self.permutes}")
        logger.info(f"{self.name}: non_permutes={self.non_permutes}")
        logger.info(f"{self.name}: exclude_patterns={self.exclude_patterns}")

        self.test_paths = []
        excluded_paths = []
        for path in self.get_test_paths(test_dir, self.negative):
            pattern_match = False
            for pattern in self.exclude_patterns:
                if re.fullmatch(pattern, path.name):
                    excluded_paths.append(path)
                    pattern_match = True
                    break
            if not pattern_match:
                self.test_paths.append(path)

        logger.debug(f"{self.name}: regex excluded paths={excluded_paths}")

        if not self.test_paths:
            logger.error(f"no tests found for {self.name} in {test_dir}")
        logger.debug(f"{self.name}: paths={self.test_paths}")

        # get default parameter values for any not given in the config
        default_params = self.get_default_params()
        for param in default_params:
            if param not in self.params or not self.params[param]:
                self.params[param] = default_params[param]
        for param in self.wks_param_names:
            logger.info(f"{self.name}: params[{param}]={self.params[param]}")

    @staticmethod
    def _get_test_paths(test_dir: Path, base_dir_glob, path_glob, negative):
        """Generate test paths for operators using operator specifics."""
        for base_dir in sorted(test_dir.glob(base_dir_glob)):
            for path in sorted(base_dir.glob(path_glob)):
                if (not negative and "ERRORIF" not in str(path)) or (
                    negative and "ERRORIF" in str(path)
                ):
                    # Check for test set paths
                    match = re.match(r"(.*)_(s[0-9]+|full|fs)", path.name)
                    if match:
                        if match.group(2) in ["s0", "full", "fs"]:
                            # Only return the truncated test name
                            # of the first test of a set, and for full tests
                            yield path.with_name(match.group(1))
                    else:
                        yield path

    @classmethod
    def get_test_paths(cls, test_dir: Path, negative):
        """Generate test paths for this operator."""
        yield from Operator._get_test_paths(test_dir, f"{cls.name}*", "*", negative)

    def path_params(self, path):
        """Return a dictionary of params from the test path."""
        params = {}
        op_name_parts = self.name.split("_")
        values = path.name.split("_")[len(op_name_parts) :]
        assert len(values) == len(
            self.wks_param_names
        ), f"len({values}) == len({self.wks_param_names})"
        for i, param in enumerate(self.wks_param_names):
            params[param] = values[i]
        return params

    def get_default_params(self):
        """Get the default parameter values from the test names."""
        params = {param: set() for param in self.wks_param_names}
        for path in self.test_paths:
            path_params = self.path_params(path)
            for k in params:
                params[k].add(path_params[k])
        for param in params:
            params[param] = sorted(list(params[param]))
        return params

    @staticmethod
    def _get_test_set_paths(path):
        """Expand a path to find all the test sets."""
        s = 0
        paths = []
        # Have a bound for the maximum test sets
        while s < 100:
            set_path = path.with_name(f"{path.name}_s{s}")
            if set_path.exists():
                paths.append(set_path)
            else:
                if s == 0:
                    logger.warning(f"Could not find test set 0 - {str(set_path)}")
                break
            s += 1
        return paths

    @staticmethod
    def _get_extra_test_paths(path):
        """Expand a path to find extra tests."""
        paths = []
        for suffix in ["full", "fs"]:
            suffix_path = path.with_name(f"{path.name}_{suffix}")
            if suffix_path.exists():
                paths.append(suffix_path)
        return paths

    def select_tests(self):  # noqa: C901 (function too complex)
        """Generate the paths to the selected tests for this operator."""
        if not self.test_paths:
            # Exit early when nothing to select from
            return

        # the test paths that have not been selected yet
        unused_paths = set(self.test_paths)

        # a list of dictionaries of unused preselected parameter combinations
        unused_preselected = [x for x in self.preselected]
        logger.debug(f"preselected: {unused_preselected}")

        # a list of dictionaries of unused permuted parameter combinations
        permutes = {k: self.params[k] for k in self.permutes}
        others = {k: self.params[k] for k in self.non_permutes}
        unused_permuted = [x for x in expand_params(permutes, others)]
        logger.debug(f"permuted: {unused_permuted}")

        # a dictionary of sets of unused parameter values
        if self.negative:
            # We only care about selecting a test for each errorif case
            unused_values = {k: set() for k in self.params}
            unused_values["case"] = set(self.params["case"])
        else:
            unused_values = {k: set(v) for k, v in self.params.items()}

        # select tests matching permuted, or preselected, parameter combinations
        for n, path in enumerate(self.test_paths):
            path_params = self.path_params(path)
            if path_params in unused_permuted or path_params in unused_preselected:
                unused_paths.remove(path)
                if path_params in unused_preselected:
                    unused_preselected.remove(path_params)
                if path_params in unused_permuted:
                    unused_permuted.remove(path_params)
                    if self.negative:
                        # remove any other errorif cases, so we only match one
                        for p in list(unused_permuted):
                            if p["case"] == path_params["case"]:
                                unused_permuted.remove(p)
                if self.full_sparsity:
                    # Test for sparsity
                    skip = False
                    for k in path_params:
                        if k in self.sparsity and n % self.sparsity[k] != 0:
                            logger.debug(f"Skipping due to {k} sparsity - {path.name}")
                            skip = True
                            break
                    if skip:
                        continue
                # remove the param values used by this path
                for k in path_params:
                    unused_values[k].discard(path_params[k])
                logger.debug(f"FOUND wanted: {path.name}")
                if path.exists():
                    yield path
                else:
                    # Must be a test set - expand to all test sets
                    for p in Operator._get_test_set_paths(path):
                        yield p
                # check for extra tests
                for p in Operator._get_extra_test_paths(path):
                    yield p

        # search for tests that match any unused parameter values
        for n, path in enumerate(sorted(list(unused_paths))):
            path_params = self.path_params(path)
            # select paths with unused param values
            # skipping some, if sparsity is set for the param
            for k in path_params:
                if path_params[k] in unused_values[k] and (
                    k not in self.sparsity or n % self.sparsity[k] == 0
                ):
                    # remove the param values used by this path
                    for p in path_params:
                        unused_values[p].discard(path_params[p])
                    sparsity = self.sparsity[k] if k in self.sparsity else 0
                    logger.debug(f"FOUND unused [{k}/{n}/{sparsity}]: {path.name}")
                    if path.exists():
                        yield path
                    else:
                        # Must be a test set - expand to all test sets
                        for p in Operator._get_test_set_paths(path):
                            yield p
                    break

        if not self.ignore_missing:
            # report any preselected combinations that were not found
            for params in unused_preselected:
                logger.warning(f"MISSING preselected: {params}")
            # report any permuted combinations that were not found
            for params in unused_permuted:
                logger.debug(f"MISSING permutation: {params}")
            # report any param values that were not found
            for k, values in unused_values.items():
                if values:
                    if k not in self.sparsity:
                        logger.warning(
                            f"MISSING {len(values)} values for {k}: {values}"
                        )
                    else:
                        logger.info(
                            f"Skipped {len(values)} values for {k} due to sparsity setting"
                        )
                        logger.debug(f"Values skipped: {values}")


class AbsOperator(Operator):
    """Test selector for the ABS operator."""

    name = "abs"


class ArithmeticRightShiftOperator(Operator):
    """Test selector for the Arithmetic Right Shift operator."""

    name = "arithmetic_right_shift"
    param_names = ["shape", "type", "rounding"]


class AddOperator(Operator):
    """Test selector for the ADD operator."""

    name = "add"


class AddShapeOperator(Operator):
    """Test selector for the ADD_SHAPE operator."""

    name = "add_shape"


class ArgmaxOperator(Operator):
    """Test selector for the ARGMAX operator."""

    name = "argmax"
    param_names = ["shape", "type", "axis"]


class AvgPool2dOperator(Operator):
    """Test selector for the AVG_POOL2D operator."""

    name = "avg_pool2d"
    param_names = ["shape", "type", "accum_type", "stride", "kernel", "pad"]


class BitwiseAndOperator(Operator):
    """Test selector for the BITWISE_AND operator."""

    name = "bitwise_and"


class BitwiseNotOperator(Operator):
    """Test selector for the BITWISE_NOT operator."""

    name = "bitwise_not"


class BitwiseOrOperator(Operator):
    """Test selector for the BITWISE_OR operator."""

    name = "bitwise_or"


class BitwiseXorOperator(Operator):
    """Test selector for the BITWISE_XOR operator."""

    name = "bitwise_xor"


class CastOperator(Operator):
    """Test selector for the CAST operator."""

    name = "cast"
    param_names = ["shape", "type", "output_type"]


class CeilOperator(Operator):
    """Test selector for the CEIL operator."""

    name = "ceil"


class ClampOperator(Operator):
    """Test selector for the CLAMP operator."""

    name = "clamp"


class CLZOperator(Operator):
    """Test selector for the CLZ operator."""

    name = "clz"


class ConcatOperator(Operator):
    """Test selector for the CONCAT operator."""

    name = "concat"
    param_names = ["shape", "type", "axis"]


class ConcatShapeOperator(Operator):
    """Test selector for the CONCAT_SHAPE operator."""

    name = "concat_shape"


class CondIfOperator(Operator):
    """Test selector for the COND_IF operator."""

    name = "cond_if"
    param_names = ["variant", "shape", "type", "cond"]


class ConstOperator(Operator):
    """Test selector for the CONST operator."""

    name = "const"


class ConstShapeOperator(Operator):
    """Test selector for the CONST_SHAPE operator."""

    name = "const_shape"


class Conv2dOperator(Operator):
    """Test selector for the CONV2D operator."""

    name = "conv2d"
    param_names = [
        "kernel",
        "shape",
        "type",
        "accum_type",
        "stride",
        "pad",
        "dilation",
        "local_bound",
    ]


class Conv3dOperator(Operator):
    """Test selector for the CONV3D operator."""

    name = "conv3d"
    param_names = [
        "kernel",
        "shape",
        "type",
        "accum_type",
        "stride",
        "pad",
        "dilation",
        "local_bound",
    ]


class DepthwiseConv2dOperator(Operator):
    """Test selector for the DEPTHWISE_CONV2D operator."""

    name = "depthwise_conv2d"
    param_names = [
        "kernel",
        "shape",
        "type",
        "accum_type",
        "stride",
        "pad",
        "dilation",
        "local_bound",
    ]


class DimOeprator(Operator):
    """Test selector for the DIM operator."""

    name = "dim"
    param_names = ["shape", "type", "axis"]


class DivShapeOperator(Operator):
    """Test selector for the DIV_SHAPE operator."""

    name = "div_shape"


class EqualOperator(Operator):
    """Test selector for the EQUAL operator."""

    name = "equal"


class ExpOperator(Operator):
    """Test selector for the EXP operator."""

    name = "exp"


class ErfOperator(Operator):
    """Test selector for the ERF operator."""

    name = "erf"


class FFT2DOperator(Operator):
    """Test selector for the FFT2D operator."""

    name = "fft2d"
    param_names = ["shape", "type", "inverse"]


class FloorOperator(Operator):
    """Test selector for the FLOOR operator."""

    name = "floor"


class FullyConnectedOperator(Operator):
    """Test selector for the FULLY_CONNECTED operator."""

    name = "fully_connected"
    param_names = ["shape", "type", "accum_type"]


class GatherOperator(Operator):
    """Test selector for the GATHER operator."""

    name = "gather"


class GreaterOperator(Operator):
    """Test selector for the GREATER operator."""

    name = "greater"

    @classmethod
    def get_test_paths(cls, test_dir: Path, negative):
        """Generate test paths for this operator."""
        yield from Operator._get_test_paths(test_dir, f"{cls.name}", "*", negative)


class GreaterEqualOperator(Operator):
    """Test selector for the GREATER_EQUAL operator."""

    name = "greater_equal"


class IdentityOperator(Operator):
    """Test selector for the IDENTITY operator."""

    name = "identity"


class IntDivOperator(Operator):
    """Test selector for the INTDIV operator."""

    name = "intdiv"


class LogOperator(Operator):
    """Test selector for the LOG operator."""

    name = "log"


class LogicalAndOperator(Operator):
    """Test selector for the LOGICAL_AND operator."""

    name = "logical_and"


class LogicalLeftShiftOperator(Operator):
    """Test selector for the LOGICAL_LEFT_SHIFT operator."""

    name = "logical_left_shift"


class LogicalNotOperator(Operator):
    """Test selector for the LOGICAL_NOT operator."""

    name = "logical_not"


class LogicalOrOperator(Operator):
    """Test selector for the LOGICAL_OR operator."""

    name = "logical_or"


class LogicalRightShiftOperator(Operator):
    """Test selector for the LOGICAL_RIGHT_SHIFT operator."""

    name = "logical_right_shift"


class LogicalXorOperator(Operator):
    """Test selector for the LOGICAL_XOR operator."""

    name = "logical_xor"


class MatmulOperator(Operator):
    """Test selector for the MATMUL operator."""

    name = "matmul"
    param_names = ["shape", "type", "accum_type"]


class MaximumOperator(Operator):
    """Test selector for the Maximum operator."""

    name = "maximum"


class MaxPool2dOperator(Operator):
    """Test selector for the MAX_POOL2D operator."""

    name = "max_pool2d"
    param_names = ["shape", "type", "stride", "kernel", "pad"]


class MinimumOperator(Operator):
    """Test selector for the Minimum operator."""

    name = "minimum"


class MulOperator(Operator):
    """Test selector for the MUL operator."""

    name = "mul"
    param_names = ["shape", "type", "perm", "shift"]


class MulShapeOperator(Operator):
    """Test selector for the MUL_SHAPE operator."""

    name = "mul_shape"


class NegateOperator(Operator):
    """Test selector for the Negate operator."""

    name = "negate"


class PadOperator(Operator):
    """Test selector for the PAD operator."""

    name = "pad"
    param_names = ["shape", "type", "pad"]


class PowOperator(Operator):
    """Test selector for the POW operator."""

    name = "pow"


class ReciprocalOperator(Operator):
    """Test selector for the RECIPROCAL operator."""

    name = "reciprocal"


class ReduceAllOperator(Operator):
    """Test selector for the REDUCE_ALL operator."""

    name = "reduce_all"
    param_names = ["shape", "type", "axis"]


class ReduceAnyOperator(Operator):
    """Test selector for the REDUCE_ANY operator."""

    name = "reduce_any"
    param_names = ["shape", "type", "axis"]


class ReduceMaxOperator(Operator):
    """Test selector for the REDUCE_MAX operator."""

    name = "reduce_max"
    param_names = ["shape", "type", "axis"]


class ReduceMinOperator(Operator):
    """Test selector for the REDUCE_MIN operator."""

    name = "reduce_min"
    param_names = ["shape", "type", "axis"]


class ReduceProductOperator(Operator):
    """Test selector for the REDUCE_PRODUCT operator."""

    name = "reduce_product"
    param_names = ["shape", "type", "axis"]


class ReduceSumOperator(Operator):
    """Test selector for the REDUCE_SUM operator."""

    name = "reduce_sum"
    param_names = ["shape", "type", "axis"]


class RescaleOperator(Operator):
    """Test selector for the RESCALE operator."""

    name = "rescale"
    param_names = [
        "shape",
        "type",
        "output_type",
        "scale",
        "double_round",
        "per_channel",
    ]


class ReshapeOperator(Operator):
    """Test selector for the RESHAPE operator."""

    name = "reshape"
    param_names = ["shape", "type", "perm", "rank", "out"]


class ResizeOperator(Operator):
    """Test selector for the RESIZE operator."""

    name = "resize"
    param_names = [
        "shape",
        "type",
        "mode",
        "output_type",
        "scale",
        "offset",
        "border",
    ]


class ReverseOperator(Operator):
    """Test selector for the REVERSE operator."""

    name = "reverse"
    param_names = ["shape", "type", "axis"]


class RFFT2DOperator(Operator):
    """Test selector for the RFFT2D operator."""

    name = "rfft2d"


class RsqrtOperator(Operator):
    """Test selector for the RSQRT operator."""

    name = "rsqrt"


class CosOperator(Operator):
    """Test selector for the COS operator."""

    name = "cos"


class SinOperator(Operator):
    """Test selector for the SIN operator."""

    name = "sin"


class ScatterOperator(Operator):
    """Test selector for the SCATTER operator."""

    name = "scatter"


class SelectOperator(Operator):
    """Test selector for the SELECT operator."""

    name = "select"


class SigmoidOperator(Operator):
    """Test selector for the SIGMOID operator."""

    name = "sigmoid"


class SliceOperator(Operator):
    """Test selector for the SLICE operator."""

    name = "slice"
    param_names = ["shape", "type", "perm"]


class SubOperator(Operator):
    """Test selector for the SUB operator."""

    name = "sub"


class SubShapeOperator(Operator):
    """Test selector for the SUB_SHAPE operator."""

    name = "sub_shape"


class TableOperator(Operator):
    """Test selector for the TABLE operator."""

    name = "table"


class TanhOperator(Operator):
    """Test selector for the TANH operator."""

    name = "tanh"


class TileOperator(Operator):
    """Test selector for the TILE operator."""

    name = "tile"
    param_names = ["shape", "type", "perm"]


class TransposeOperator(Operator):
    """Test selector for the TRANSPOSE operator."""

    name = "transpose"
    param_names = ["shape", "type", "perm"]

    @classmethod
    def get_test_paths(cls, test_dir: Path, negative):
        """Generate test paths for this operator."""
        yield from Operator._get_test_paths(test_dir, f"{cls.name}", "*", negative)


class TransposeConv2dOperator(Operator):
    """Test selector for the TRANSPOSE_CONV2D operator."""

    name = "transpose_conv2d"
    param_names = [
        "kernel",
        "shape",
        "type",
        "accum_type",
        "stride",
        "pad",
        "out_shape",
        "local_bound",
    ]

    def path_params(self, path):
        """Return a dictionary of params from the test path."""
        params = super().path_params(path)
        # out_shape is different for every test case, so ignore it for selection
        params["out_shape"] = ""
        return params


class WhileLoopOperator(Operator):
    """Test selector for the WHILE_LOOP operator."""

    name = "while_loop"
    param_names = ["shape", "type", "cond"]


def parse_args():
    """Parse the arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test-dir",
        default=Path.cwd(),
        type=Path,
        help=(
            "The directory where test subdirectories for all operators can be found"
            " (default: current working directory)"
        ),
    )
    parser.add_argument(
        "--config",
        default=Path(__file__).with_suffix(".json"),
        type=Path,
        help="A JSON file defining the parameters to use for each operator",
    )
    parser.add_argument(
        "--selector",
        default="default",
        type=str,
        help="The selector in the selection dictionary to use for each operator",
    )
    parser.add_argument(
        "--full-path", action="store_true", help="output the full path for each test"
    )
    parser.add_argument(
        "-v",
        dest="verbosity",
        action="count",
        default=0,
        help="Verbosity (can be used multiple times for more details)",
    )
    parser.add_argument(
        "operators",
        type=str,
        nargs="*",
        help=(
            f"Select tests for the specified operator(s)"
            f" - all operators are assumed if none are specified)"
            f" - choose from: {[n for n in Operator.registry]}"
        ),
    )
    parser.add_argument(
        "--test-type",
        dest="test_type",
        choices=["positive", "negative"],
        default="positive",
        type=str,
        help="type of tests selected, positive or negative",
    )
    return parser.parse_args()


def main():
    """Example test selection."""
    args = parse_args()

    loglevels = (logging.ERROR, logging.WARNING, logging.INFO, logging.DEBUG)
    logger.setLevel(loglevels[min(args.verbosity, len(loglevels) - 1)])
    logger.info(f"{__file__}: args: {args}")

    try:
        with open(args.config, "r") as fd:
            config = json.load(fd)
    except Exception as e:
        logger.error(f"Config file error: {e}")
        return 2

    negative = args.test_type == "negative"
    for op_name in Operator.registry:
        if not args.operators or op_name in args.operators:
            op_params = config[op_name] if op_name in config else {}
            if "selection" in op_params and args.selector in op_params["selection"]:
                selection_config = op_params["selection"][args.selector]
            else:
                logger.warning(
                    f"Could not find selection config {args.selector} for {op_name}"
                )
                selection_config = {}
            op = Operator.registry[op_name](args.test_dir, selection_config, negative)
            for test_path in op.select_tests():
                print(test_path.resolve() if args.full_path else test_path.name)

    return 0


if __name__ == "__main__":
    exit(main())
