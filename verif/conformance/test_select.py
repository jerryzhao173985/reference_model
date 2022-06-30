# Copyright (c) 2021-2022, ARM Limited.
# SPDX-License-Identifier: Apache-2.0
"""Select generated tests."""
import argparse
import itertools
import json
import logging
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
        exclude_types=None,
    ):
        """Initialise the selection parameters for an operator.

        test_dir: the directory where the tests for all operators can be found
        config: a dictionary with:
                "params" - mappings of parameter names to the values to select
                "permutes" - a list of parameter names to be permuted
                "errorifs" - list of ERRORIF case names to be selected (negative test)
        negative: bool indicating if negative testing is being selected (ERRORIF tests)
        """
        assert isinstance(
            self.name, str
        ), f"{self.__class__.__name__}: {self.name} is not a valid operator name"

        self.negative = negative
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
        self.preselected = config["preselected"] if "preselected" in config else {}
        self.non_permutes = [x for x in self.wks_param_names if x not in self.permutes]
        logger.info(f"{self.name}: permutes={self.permutes}")
        logger.info(f"{self.name}: non_permutes={self.non_permutes}")

        if exclude_types is None:
            exclude_types = []
        self.test_paths = [
            p
            for p in self.get_test_paths(test_dir, self.negative)
            # exclusion of types if requested
            if self.path_params(p)["type"] not in exclude_types
        ]
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
        for path in self.test_paths:
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
                # remove the param values used by this path
                for k in path_params:
                    unused_values[k].discard(path_params[k])
                logger.debug(f"FOUND: {path.name}")
                yield path

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
                    logger.debug(f"FOUND: {path.name}")
                    yield path
                    break

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
                    logger.warning(f"MISSING {len(values)} values for {k}: {values}")
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


class ArgmaxOperator(Operator):
    """Test selector for the ARGMAX operator."""

    name = "argmax"
    param_names = ["shape", "type", "axis"]


class AvgPool2dOperator(Operator):
    """Test selector for the AVG_POOL2D operator."""

    name = "avg_pool2d"
    param_names = ["shape", "type", "stride", "kernel", "pad"]


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


class ClampOperator(Operator):
    """Test selector for the CLAMP operator."""

    name = "clamp"


class CLZOperator(Operator):
    """Test selector for the CLZ operator."""

    name = "clz"
    param_names = ["shape", "type"]


class ConcatOperator(Operator):
    """Test selector for the CONCAT operator."""

    name = "concat"
    param_names = ["shape", "type", "axis"]


class CondIfOperator(Operator):
    """Test selector for the COND_IF operator."""

    name = "cond_if"
    param_names = ["variant", "shape", "type", "cond"]


class ConstOperator(Operator):
    """Test selector for the CONST operator."""

    name = "const"


class Conv2dOperator(Operator):
    """Test selector for the CONV2D operator."""

    name = "conv2d"
    param_names = ["kernel", "shape", "type", "stride", "pad", "dilation"]


class Conv3dOperator(Operator):
    """Test selector for the CONV3D operator."""

    name = "conv3d"
    param_names = ["kernel", "shape", "type", "stride", "pad", "dilation"]


class DepthwiseConv2dOperator(Operator):
    """Test selector for the DEPTHWISE_CONV2D operator."""

    name = "depthwise_conv2d"
    param_names = ["kernel", "shape", "type", "stride", "pad", "dilation"]


class EqualOperator(Operator):
    """Test selector for the EQUAL operator."""

    name = "equal"


class FullyConnectedOperator(Operator):
    """Test selector for the FULLY_CONNECTED operator."""

    name = "fully_connected"


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
    """Test selector for the INTDIV."""

    name = "intdiv"


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


class NegateOperator(Operator):
    """Test selector for the Negate operator."""

    name = "negate"


class PadOperator(Operator):
    """Test selector for the PAD operator."""

    name = "pad"
    param_names = ["shape", "type", "pad"]


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
    param_names = ["shape", "type", "perm", "rank"]


class ResizeOperator(Operator):
    """Test selector for the RESIZE operator."""

    name = "resize"
    param_names = [
        "shape",
        "type",
        "mode",
        "shift",
        "output_dims",
        "output_type",
        "stride",
        "offset",
    ]


class ReverseOperator(Operator):
    """Test selector for the REVERSE operator."""

    name = "reverse"
    param_names = ["shape", "type", "axis"]


class ScatterOperator(Operator):
    """Test selector for the SCATTER operator."""

    name = "scatter"


class SelectOperator(Operator):
    """Test selector for the SELECT operator."""

    name = "select"


class SliceOperator(Operator):
    """Test selector for the SLICE operator."""

    name = "slice"
    param_names = ["shape", "type", "perm"]


class SubOperator(Operator):
    """Test selector for the SUB operator."""

    name = "sub"


class TableOperator(Operator):
    """Test selector for the TABLE operator."""

    name = "table"


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
    param_names = ["kernel", "shape", "type", "stride", "pad", "out_shape"]

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
    logger.basicConfig(level=loglevels[min(args.verbosity, len(loglevels) - 1)])
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
            op = Operator.registry[op_name](
                args.test_dir, op_params, negative, exclude_types=["float"]
            )
            for test_path in op.select_tests():
                print(test_path.resolve() if args.full_path else test_path.name)

    return 0


if __name__ == "__main__":
    exit(main())
