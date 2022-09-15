"""TOSA test runner module for a dummy System Under Test (SUT)."""
# Copyright (c) 2021, ARM Limited.
# SPDX-License-Identifier: Apache-2.0
from runner.tosa_test_runner import TosaTestRunner


class TosaSUTRunner(TosaTestRunner):
    """TOSA dummy SUT runner."""

    def __init__(self, args, runnerArgs, testDirPath):
        """Initialize using the given test details."""
        super().__init__(args, runnerArgs, testDirPath)

    def runTestGraph(self):
        """Nothing run as this is a dummy SUT that does nothing."""
        graphResult = TosaTestRunner.TosaGraphResult.TOSA_VALID
        graphMessage = "Dummy system under test - nothing run"

        # Return graph result and message
        return graphResult, graphMessage
