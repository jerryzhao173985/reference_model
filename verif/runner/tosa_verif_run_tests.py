"""TOSA verification runner script."""
# Copyright (c) 2020-2022, ARM Limited.
# SPDX-License-Identifier: Apache-2.0
import argparse
import importlib
import os
import queue
import threading
import traceback
from datetime import datetime
from pathlib import Path

from json2numpy import json2numpy
from runner.tosa_test_runner import TosaTestInvalid
from runner.tosa_test_runner import TosaTestRunner
from xunit import xunit

TOSA_REFMODEL_RUNNER = "runner.tosa_refmodel_sut_run"
MAX_XUNIT_TEST_MESSAGE = 1000


def parseArgs(argv):
    """Parse the arguments and return the settings."""
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "-t",
        "--test",
        dest="test",
        type=str,
        nargs="+",
        help="Test(s) to run",
    )
    group.add_argument(
        "-T",
        "--test-list",
        dest="test_list_file",
        type=Path,
        help="File containing list of tests to run (one per line)",
    )
    parser.add_argument(
        "-r",
        "--recursive",
        dest="recursive_tests",
        action="store_true",
        help="Recursively search for tests",
    )
    parser.add_argument(
        "--operator-fbs",
        dest="operator_fbs",
        default="conformance_tests/third_party/serialization_lib/schema/tosa.fbs",
        type=str,
        help="flat buffer syntax file",
    )
    parser.add_argument(
        "--ref-model-path",
        dest="ref_model_path",
        default="reference_model/build/reference_model/tosa_reference_model",
        type=str,
        help="Path to reference model executable",
    )
    parser.add_argument(
        "--flatc-path",
        dest="flatc_path",
        default="reference_model/build/thirdparty/serialization_lib/third_party/flatbuffers/flatc",
        type=str,
        help="Path to flatc compiler executable",
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
        "-b",
        "--binary",
        dest="binary",
        action="store_true",
        help="Convert to using binary flatbuffers instead of JSON",
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
        default=[TOSA_REFMODEL_RUNNER],
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
    parser.add_argument(
        "--test-type",
        dest="test_type",
        type=str,
        default="both",
        choices=["positive", "negative", "both"],
        help="Filter tests based on expected failure status",
    )
    parser.add_argument(
        "--no-color",
        "--no-colour",
        dest="no_color",
        action="store_true",
        help="Disable color output",
    )
    parser.add_argument(
        "--profile",
        dest="profile",
        type=str,
        choices=["tosa-bi", "tosa-mi"],
        help="Filter tests based on profile",
    )
    parser.add_argument(
        "--tosa_level",
        dest="tosa_level",
        default="EIGHTK",
        type=str,
        help="A TOSA level defines operator parameter ranges that an implementation shall support."
        "Config tosa_level for running the reference model only. Default is EIGHTK",
    )
    parser.add_argument(
        "-p",
        "--precise-mode",
        dest="precise_mode",
        action="store_true",
        help="Run the reference model in precise mode (FP64)",
    )

    args = parser.parse_args(argv)

    # Autodetect CPU count
    if args.jobs <= 0:
        args.jobs = os.cpu_count()

    return args


EXCLUSION_PREFIX = ["test", "model", "desc"]


def convert2Numpy(test_path):
    """Convert all the JSON numpy files back into binary numpy."""
    jsons = test_path.glob("*.json")
    for json in jsons:
        for exclude in EXCLUSION_PREFIX:
            if json.name.startswith(exclude):
                json = None
                break
        if json:
            # debug print(f"Converting {json}")
            json2numpy.json_to_npy(json)


def workerThread(task_queue, runnerList, args, result_queue):
    """Worker thread that runs the next test from the queue."""
    while True:
        try:
            test_path = task_queue.get(block=False)
        except queue.Empty:
            break

        if test_path is None:
            break

        msg = ""
        converted = False
        for runnerModule, runnerArgs in runnerList:
            try:
                start_time = datetime.now()
                # Set up system under test runner
                runnerName = runnerModule.__name__
                runner = runnerModule.TosaSUTRunner(args, runnerArgs, test_path)

                skip, reason = runner.skipTest()
                if skip:
                    msg = "Skipping {} test".format(reason)
                    print("{} {}".format(msg, test_path))
                    rc = TosaTestRunner.Result.SKIPPED
                else:
                    # Convert JSON data files into numpy format on first pass
                    if not converted:
                        convert2Numpy(test_path)
                        converted = True

                    if args.verbose:
                        print(
                            "Running runner {} with test {}".format(
                                runnerName, test_path
                            )
                        )
                    try:
                        grc, gmsg = runner.runTestGraph()
                        rc, msg = runner.testResult(grc, gmsg)
                    except Exception as e:
                        msg = "System Under Test error: {}".format(e)
                        print(msg)
                        print(
                            "".join(
                                traceback.format_exception(
                                    etype=type(e), value=e, tb=e.__traceback__
                                )
                            )
                        )
                        rc = TosaTestRunner.Result.INTERNAL_ERROR
            except Exception as e:
                msg = "Internal error: {}".format(e)
                print(msg)
                if not isinstance(e, TosaTestInvalid):
                    # Show stack trace on unexpected exceptions
                    print(
                        "".join(
                            traceback.format_exception(
                                etype=type(e), value=e, tb=e.__traceback__
                            )
                        )
                    )
                rc = TosaTestRunner.Result.INTERNAL_ERROR
            finally:
                end_time = datetime.now()
                result_queue.put(
                    (runnerName, test_path, rc, msg, end_time - start_time)
                )

        task_queue.task_done()

    return True


def loadSUTRunnerModules(args):
    """Load in the system under test modules.

    Returns a list of tuples of (runner_module, [argument list])
    """
    runnerList = []
    # Remove any duplicates from the list
    sut_module_list = list(set(args.sut_module))
    for r in sut_module_list:
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


def createXUnitResults(xunitFile, runnerList, resultLists, verbose):
    """Create the xunit results file."""
    xunit_result = xunit.xunit_results()

    for runnerModule, _ in runnerList:
        # Create test suite per system under test (runner)
        runner = runnerModule.__name__
        xunit_suite = xunit_result.create_suite(runner)

        # Sort by test name
        for test_path, rc, msg, time_delta in sorted(
            resultLists[runner], key=lambda tup: tup[0]
        ):
            test_name = str(test_path)
            xt = xunit.xunit_test(test_name, runner)

            xt.time = str(
                float(time_delta.seconds) + (float(time_delta.microseconds) * 1e-6)
            )

            testMsg = rc.name if not msg else "{}: {}".format(rc.name, msg)

            if (
                rc == TosaTestRunner.Result.EXPECTED_PASS
                or rc == TosaTestRunner.Result.EXPECTED_FAILURE
            ):
                if verbose:
                    print("{} {} ({})".format(rc.name, test_name, runner))
            elif rc == TosaTestRunner.Result.SKIPPED:
                xt.skipped()
                if verbose:
                    print("{} {} ({})".format(rc.name, test_name, runner))
            else:
                xt.failed(testMsg)
                print("{} {} ({})".format(rc.name, test_name, runner))

            xunit_suite.tests.append(xt)

    xunit_result.write_results(xunitFile)


def getTestsInPath(path):
    # Recursively find any tests in this directory
    desc_path = path / "desc.json"
    if desc_path.is_file():
        return [path]
    elif path.is_dir():
        path_list = []
        for p in path.glob("*"):
            path_list.extend(getTestsInPath(p))
        return path_list
    else:
        return []


def main(argv=None):
    """Start worker threads to do the testing and outputs the results."""
    args = parseArgs(argv)

    if (
        TOSA_REFMODEL_RUNNER in args.sut_module
        and not Path(args.ref_model_path).is_file()
    ):
        print(
            "Argument error: Reference Model not found ({})".format(args.ref_model_path)
        )
        exit(2)

    if args.test_list_file:
        try:
            with args.test_list_file.open("r") as f:
                args.test = f.read().splitlines()
        except Exception as e:
            print(
                "Argument error: Cannot read list of tests in {}\n{}".format(
                    args.test_list_file, e
                )
            )
            exit(2)

    runnerList = loadSUTRunnerModules(args)

    threads = []
    taskQueue = queue.Queue()
    resultQueue = queue.Queue()

    for tdir in args.test:
        tpath = Path(tdir)
        if tpath.is_file():
            if tpath.name != "README":
                print(
                    "Warning: Skipping test {} as not a valid directory".format(tpath)
                )
        else:
            if args.recursive_tests:
                tpath_list = getTestsInPath(tpath)
            else:
                tpath_list = [tpath]

            for t in tpath_list:
                taskQueue.put((t))

    print(
        "Running {} tests on {} system{} under test".format(
            taskQueue.qsize(), len(runnerList), "s" if len(runnerList) > 1 else ""
        )
    )

    for i in range(args.jobs):
        t = threading.Thread(
            target=workerThread, args=(taskQueue, runnerList, args, resultQueue)
        )
        t.setDaemon(True)
        t.start()
        threads.append(t)

    taskQueue.join()

    # Set up results lists for each system under test
    resultLists = {}
    results = {}
    for runnerModule, _ in runnerList:
        runner = runnerModule.__name__
        resultLists[runner] = []
        results[runner] = [0] * len(TosaTestRunner.Result)

    while True:
        try:
            runner, test_path, rc, msg, time_delta = resultQueue.get(block=False)
            resultQueue.task_done()
        except queue.Empty:
            break

        # Limit error messages to make results easier to digest
        if msg and len(msg) > MAX_XUNIT_TEST_MESSAGE:
            half = int(MAX_XUNIT_TEST_MESSAGE / 2)
            trimmed = len(msg) - MAX_XUNIT_TEST_MESSAGE
            msg = "{} ...\nskipped {} bytes\n... {}".format(
                msg[:half], trimmed, msg[-half:]
            )
        resultLists[runner].append((test_path, rc, msg, time_delta))
        results[runner][rc] += 1

    createXUnitResults(args.xunit_file, runnerList, resultLists, args.verbose)

    # Print out results for each system under test
    for runnerModule, _ in runnerList:
        runner = runnerModule.__name__
        resultSummary = []
        for result in TosaTestRunner.Result:
            resultSummary.append(
                "{} {}".format(results[runner][result], result.name.lower())
            )
        print("Totals ({}): {}".format(runner, ", ".join(resultSummary)))

    return 0


if __name__ == "__main__":
    exit(main())
