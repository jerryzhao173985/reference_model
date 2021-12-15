#!/usr/bin/env python3
"""Mocked flatc compiler for testing."""
# Copyright (c) 2021-2022, ARM Limited.
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path


def main(argv=None):
    """Mock the required behaviour of the flatc compiler."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o",
        dest="output_dir",
        type=Path,
        help="output directory",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="convert to JSON",
    )
    parser.add_argument(
        "--binary",
        action="store_true",
        help="convert to binary",
    )
    parser.add_argument(
        "--raw-binary",
        action="store_true",
        help="convert from raw-binary",
    )
    parser.add_argument(
        "path",
        type=Path,
        action="append",
        nargs="*",
        help="the path to fbs or files to convert",
    )

    args = parser.parse_args(argv)
    path = args.path
    if len(path) == 0:
        print("ERROR: Missing fbs files and files to convert")
        return 2
    return 0


if __name__ == "__main__":
    exit(main())
