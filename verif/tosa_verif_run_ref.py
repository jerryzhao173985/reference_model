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
import importlib


from enum import IntEnum, Enum, unique
from datetime import datetime

# Include the ../scripts and ../scripts/xunit directory in PYTHONPATH
parent_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(parent_dir, "..", "scripts"))
sys.path.append(os.path.join(parent_dir, "..", "scripts", "xunit"))

import xunit

# Include the ../thirdparty/serialization_lib/python directory in PYTHONPATH
parent_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(
    os.path.join(parent_dir, "..", "thirdparty", "serialization_lib", "python")
)
import tosa
from tosa_test_gen import TosaTestGen
from tosa_test_runner import TosaTestRunner

no_color_printing = False
# from run_tf_unit_test import LogColors, print_color, run_sh_command


def parseArgs():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t",
        "--test",
        dest="test",
        type=str,
        nargs="+",
        help="Test(s) to run",
        required=True,
    )
    parser.add_argument(
        "--seed",
        dest="random_seed",
        default=42,
        type=int,
        help="Random seed for test generation",
    )
    parser.add_argument(
        "--ref-model-path",
        dest="ref_model_path",
        default="build/reference_model/tosa_reference_model",
        type=str,
        help="Path to reference model executable",
    )
    parser.add_argument(
        "--ref-debug",
        dest="ref_debug",
        default="",
        type=str,
        help="Reference debug flag (low, med, high)",
    )
    parser.add_argument(
        "--ref-intermediates",
        dest="ref_intermediates",
        default=0,
        type=int,
        help="Reference model dumps intermediate tensors",
    )
    parser.add_argument(
        "-v", "--verbose", dest="verbose", action="count", help="Verbose operation"
    )
    parser.add_argument(
        "-j", "--jobs", dest="jobs", type=int, default=1, help="Number of parallel jobs"
    )
    parser.add_argument(
        "--sut-module",
        "-s",
        dest="sut_module",
        type=str,
        nargs="+",
        default=["tosa_ref_run"],
        help="System under test module to load (derives from TosaTestRunner).  May be repeated",
    )
    parser.add_argument(
        "--sut-module-args",
        dest="sut_module_args",
        type=str,
        nargs="+",
        default=[],
        help="System under test module arguments.  Use sutmodulename:argvalue to pass an argument.  May be repeated.",
    )
    parser.add_argument(
        "--xunit-file",
        dest="xunit_file",
        type=str,
        default="result.xml",
        help="XUnit output file",
    )

    args = parser.parse_args()

    # Autodetect CPU count
    if args.jobs <= 0:
        args.jobs = os.cpu_count()

    return args


def workerThread(task_queue, runnerList, args, result_queue):
    while True:
        try:
            test = task_queue.get(block=False)
        except queue.Empty:
            break

        if test is None:
            break

        msg = ""
        start_time = datetime.now()
        try:

            for runnerModule, runnerArgs in runnerList:
                if args.verbose:
                    print(
                        "Running runner {} with test {}".format(
                            runnerModule.__name__, test
                        )
                    )
                runner = runnerModule.TosaRefRunner(args, runnerArgs, test)
                try:
                    rc = runner.runModel()
                except Exception as e:
                    rc = TosaTestRunner.Result.INTERNAL_ERROR
                    print(f"runner.runModel Exception: {e}")
                    print(
                        "".join(
                            traceback.format_exception(
                                etype=type(e), value=e, tb=e.__traceback__
                            )
                        )
                    )
        except Exception as e:
            print("Internal regression error: {}".format(e))
            print(
                "".join(
                    traceback.format_exception(
                        etype=type(e), value=e, tb=e.__traceback__
                    )
                )
            )
            rc = TosaTestRunner.Result.INTERNAL_ERROR

        end_time = datetime.now()

        result_queue.put((test, rc, msg, end_time - start_time))
        task_queue.task_done()

    return True


def loadRefModules(args):
    # Returns a tuple of (runner_module, [argument list])
    runnerList = []
    for r in args.sut_module:
        if args.verbose:
            print("Loading module {}".format(r))

        runner = importlib.import_module(r)

        # Look for arguments associated with this runner
        runnerArgPrefix = "{}:".format(r)
        runnerArgList = []
        for a in args.sut_module_args:
            if a.startswith(runnerArgPrefix):
                runnerArgList.append(a[len(runnerArgPrefix) :])
        runnerList.append((runner, runnerArgList))

    return runnerList


def main():
    args = parseArgs()

    runnerList = loadRefModules(args)

    threads = []
    taskQueue = queue.Queue()
    resultQueue = queue.Queue()

    for t in args.test:
        taskQueue.put((t))

    print("Running {} tests ".format(taskQueue.qsize()))

    for i in range(args.jobs):
        t = threading.Thread(
            target=workerThread, args=(taskQueue, runnerList, args, resultQueue)
        )
        t.setDaemon(True)
        t.start()
        threads.append(t)

    taskQueue.join()

    resultList = []
    results = [0] * len(TosaTestRunner.Result)

    while True:
        try:
            test, rc, msg, time_delta = resultQueue.get(block=False)
        except queue.Empty:
            break

        resultList.append((test, rc, msg, time_delta))
        results[rc] = results[rc] + 1

    xunit_result = xunit.xunit_results("Regressions")
    xunit_suite = xunit_result.create_suite("Unit tests")

    # Sort by test name
    for test, rc, msg, time_delta in sorted(resultList, key=lambda tup: tup[0]):
        test_name = test
        xt = xunit.xunit_test(test_name, "reference")

        xt.time = str(
            float(time_delta.seconds) + (float(time_delta.microseconds) * 1e-6)
        )

        if (
            rc == TosaTestRunner.Result.EXPECTED_PASS
            or rc == TosaTestRunner.Result.EXPECTED_FAILURE
        ):
            if args.verbose:
                print("{} {}".format(rc.name, test_name))
        else:
            xt.failed(msg)
            print("{} {}".format(rc.name, test_name))

        xunit_suite.tests.append(xt)
        resultQueue.task_done()

    xunit_result.write_results(args.xunit_file)

    print("Totals: ", end="")
    for result in TosaTestRunner.Result:
        print("{} {}, ".format(results[result], result.name.lower()), end="")
    print()

    return 0


if __name__ == "__main__":
    exit(main())
