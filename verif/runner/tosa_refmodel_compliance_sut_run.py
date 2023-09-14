"""TOSA ref model compliance runner module."""
# Copyright (c) 2023, ARM Limited.
# SPDX-License-Identifier: Apache-2.0
from runner.tosa_refmodel_sut_run import TosaSUTRunner as TosaRefRunner


class TosaSUTRunner(TosaRefRunner):
    """Compliance mode enabled ref model runner."""

    def __init__(self, args, runnerArgs, testDirPath):
        """Initialize the TosaTestRunner base class"""
        super().__init__(args, runnerArgs, testDirPath)

        # Override - Set compliance mode precise FP64 calculations
        self.compliance = True

    # All other functions inherited from refmodel_sut_run
