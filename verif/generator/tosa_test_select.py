# Copyright (c) 2024, ARM Limited.
# SPDX-License-Identifier: Apache-2.0
import copy
import logging

import generator.tosa_utils as gtu

logging.basicConfig()
logger = logging.getLogger("tosa_verif_build_tests")


class Test:
    """Test container to allow group and permute selection."""

    def __init__(
        self, opName, testStr, dtype, error, shapeList, argsDict, testOpName=None
    ):
        self.opName = opName
        self.testStr = testStr
        self.dtype = dtype
        self.error = error
        self.shapeList = shapeList
        self.argsDict = argsDict
        # Given test op name used for look up in TOSA_OP_LIST for "conv2d_1x1" for example
        self.testOpName = testOpName if testOpName is not None else opName

        self.key = None
        self.groupKey = None
        self.mark = False

    def __str__(self):
        return self.testStr

    def __lt__(self, other):
        return self.testStr < str(other)

    def getArg(self, param):
        # Get parameter values (arguments) for this test
        if param == "rank":
            return len(self.shapeList[0])
        elif param == "type":
            if isinstance(self.dtype, list):
                return tuple(self.dtype)
            return self.dtype
        elif param == "shape" and "shape" not in self.argsDict:
            return str(self.shapeList[0])

        if param in self.argsDict:
            # Turn other args into hashable string without newlines
            val = str(self.argsDict[param])
            return ",".join(str(val).splitlines())
        else:
            assert False, f"Argument {param} not found in argsDict for {self.testStr}"
            return None

    def setKey(self, keyParams):
        if self.error is None:
            # Create the main key based on primary parameters
            key = [self.getArg(param) for param in keyParams]
            self.key = tuple(key)
        else:
            # Use the error as the key
            self.key = self.error
        return self.key

    def getKey(self):
        return self.key

    def setGroupKey(self, groupParams):
        # Create the group key based on arguments that do not define the group
        # Therefore this test will match other tests that have the same arguments
        # that are NOT the group arguments (group arguments like test set number)
        paramsList = sorted(["shape", "type"] + list(self.argsDict.keys()))
        key = []
        for param in paramsList:
            if param in groupParams:
                continue
            key.append(self.getArg(param))
        self.groupKey = tuple(key)
        return self.groupKey

    def getGroupKey(self):
        return self.groupKey

    def inGroup(self, groupKey):
        return self.groupKey == groupKey

    def setMark(self):
        # Marks the test as important
        self.mark = True

    def getMark(self):
        return self.mark

    def isError(self):
        return self.error is not None


def _get_selection_info_from_op(op, selectionCriteria, item, default):
    # Get selection info from the op
    if (
        "selection" in op
        and selectionCriteria in op["selection"]
        and item in op["selection"][selectionCriteria]
    ):
        return op["selection"][selectionCriteria][item]
    else:
        return default


def _get_tests_by_group(tests):
    # Create simple structures to record the tests in groups
    groups = []
    group_tests = {}

    for test in tests:
        key = test.getGroupKey()
        if key in group_tests:
            group_tests[key].append(test)
        else:
            group_tests[key] = [test]
            groups.append(key)

    # Return list of test groups (group keys) and a dictionary with a list of tests
    # associated with each group key
    return groups, group_tests


def _get_specific_op_info(opName, opSelectionInfo, testOpName):
    # Get the op specific section from the selection config
    name = opName if opName in opSelectionInfo else testOpName
    if name not in opSelectionInfo:
        logger.info(f"No op entry found for {opName} in test selection config")
        return {}
    return opSelectionInfo[name]


class TestOpList:
    """All the tests for one op grouped by permutations."""

    def __init__(self, opName, opSelectionInfo, selectionCriteria, testOpName):
        self.opName = opName
        self.testOpName = testOpName
        op = _get_specific_op_info(opName, opSelectionInfo, testOpName)

        # See verif/conformance/README.md for more information on
        # these selection arguments
        self.permuteArgs = _get_selection_info_from_op(
            op, selectionCriteria, "permutes", ["rank", "type"]
        )
        self.paramArgs = _get_selection_info_from_op(
            op, selectionCriteria, "full_params", []
        )
        self.specificArgs = _get_selection_info_from_op(
            op, selectionCriteria, "specifics", {}
        )
        self.groupArgs = _get_selection_info_from_op(
            op, selectionCriteria, "groups", ["s"]
        )
        self.maximumPerPermute = _get_selection_info_from_op(
            op, selectionCriteria, "maximum", None
        )
        self.numErrorIfs = _get_selection_info_from_op(
            op, selectionCriteria, "num_errorifs", 1
        )
        self.selectAll = _get_selection_info_from_op(
            op, selectionCriteria, "all", False
        )

        self.tests = []
        self.testStrings = set()
        self.shapes = set()

        self.permutes = set()
        self.testsPerPermute = {}
        self.paramsPerPermute = {}
        self.specificsPerPermute = {}

        self.selectionDone = False

    def __len__(self):
        return len(self.tests)

    def add(self, test):
        # Add a test to this op group and set up the permutations/group for it
        assert test.opName.startswith(self.opName)
        if str(test) in self.testStrings:
            logger.debug(f"Skipping duplicate test: {str(test)}")
            return

        self.tests.append(test)
        self.testStrings.add(str(test))

        self.shapes.add(test.getArg("shape"))

        # Work out the permutation key for this test
        permute = test.setKey(self.permuteArgs)
        # Set up the group key for the test (for pulling out groups during selection)
        test.setGroupKey(self.groupArgs)

        if permute not in self.permutes:
            # New permutation
            self.permutes.add(permute)
            # Set up area to record the selected tests
            self.testsPerPermute[permute] = []
            if self.paramArgs:
                # Set up area to record the unique test params found
                self.paramsPerPermute[permute] = {}
                for param in self.paramArgs:
                    self.paramsPerPermute[permute][param] = set()
            # Set up copy of the specific test args for selecting these
            self.specificsPerPermute[permute] = copy.deepcopy(self.specificArgs)

    def _init_select(self):
        # Can only perform the selection process once as it alters the permute
        # information set at init
        assert not self.selectionDone

        # Count of non-specific tests added to each permute (not error)
        if not self.selectAll:
            countPerPermute = {permute: 0 for permute in self.permutes}

        # Go through each test looking for permutes, unique params & specifics
        for test in self.tests:
            permute = test.getKey()
            append = False
            possible_append = False

            if test.isError():
                # Error test, choose up to number of tests
                if len(self.testsPerPermute[permute]) < self.numErrorIfs:
                    append = True
            else:
                if self.selectAll:
                    append = True
                else:
                    # See if this is a specific test to add
                    for param, values in self.specificsPerPermute[permute].items():
                        arg = test.getArg(param)
                        # Iterate over a copy of the values, so we can remove them from the original
                        if arg in values.copy():
                            # Found a match, remove it, so we don't look for it later
                            values.remove(arg)
                            # Mark the test as special (and so shouldn't be removed)
                            test.setMark()
                            append = True

                    if self.paramArgs:
                        # See if this test contains any new params we should keep
                        # Perform this check even if we have already selected the test
                        # so we can record the params found
                        for param in self.paramArgs:
                            arg = test.getArg(param)
                            if arg not in self.paramsPerPermute[permute][param]:
                                # We have found a new value for this arg, record it
                                self.paramsPerPermute[permute][param].add(arg)
                                possible_append = True
                    else:
                        # No params set, so possible test to add up to maximum
                        possible_append = True

                    if (not append and possible_append) and (
                        self.maximumPerPermute is None
                        or countPerPermute[permute] < self.maximumPerPermute
                    ):
                        # Not selected but could be added and we have space left if
                        # a maximum is set.
                        append = True
                        countPerPermute[permute] += 1

            # Check for grouping with chosen tests
            if not append:
                # We will keep any tests together than form a group
                key = test.getGroupKey()
                for t in self.testsPerPermute[permute]:
                    if t.getGroupKey() == key:
                        if t.getMark():
                            test.setMark()
                        append = True

            # Check for FP special tests
            if not append:
                if test.argsDict["dg_type"] == gtu.DataGenType.FP_SPECIAL:
                    append = True

            if append:
                self.testsPerPermute[permute].append(test)

        self.selectionDone = True

    def select(self, rng=None):
        # Create selection of tests with optional shuffle
        if not self.selectionDone:
            if rng:
                rng.shuffle(self.tests)

            self._init_select()

        # Now create the full list of selected tests per permute
        selection = []

        for permute, tests in self.testsPerPermute.items():
            selection.extend(tests)

        return selection

    def all(self):
        # Un-selected list of tests - i.e. all of them
        return self.tests


class TestList:
    """List of all tests grouped by operator."""

    def __init__(self, opSelectionInfo, selectionCriteria="default"):
        self.opLists = {}
        self.opSelectionInfo = opSelectionInfo
        self.selectionCriteria = selectionCriteria

    def __len__(self):
        length = 0
        for opName in self.opLists.keys():
            length += len(self.opLists[opName])
        return length

    def add(self, test):
        if test.opName not in self.opLists:
            self.opLists[test.opName] = TestOpList(
                test.opName,
                self.opSelectionInfo,
                self.selectionCriteria,
                test.testOpName,
            )
        self.opLists[test.opName].add(test)

    def _get_tests(self, selectMode, rng):
        selection = []

        for opList in self.opLists.values():
            if selectMode:
                tests = opList.select(rng=rng)
            else:
                tests = opList.all()
            selection.extend(tests)

        if selectMode:
            selection = sorted(selection)
        return selection

    def select(self, rng=None):
        return self._get_tests(True, rng)

    def all(self):
        return self._get_tests(False, None)
