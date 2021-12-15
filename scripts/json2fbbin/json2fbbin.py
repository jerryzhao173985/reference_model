"""Conversion utility from flatbuffer JSON files to binary and the reverse."""
# Copyright (c) 2021-2022, ARM Limited.
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
from typing import Optional

from runner.run_command import run_sh_command, RunShCommandError


def fbbin_to_json(flatc: Path, fbs: Path, t_path: Path, o_path: Optional[Path] = None):
    """Convert the binary flatbuffer to JSON.

    flatc: the Path to the flatc compiler program
    fbs: the Path to the fbs (flatbuffer schema) file
    t_path: the Path to the binary flatbuffer file
    o_path: the output Path where JSON file will be put, if None, it is same as t_path
    """
    if o_path is None:
        o_path = t_path.parent
    cmd = [
        str(flatc.absolute()),
        "-o",
        str(o_path.absolute()),
        "--json",
        "--defaults-json",
        "--raw-binary",
        str(fbs.absolute()),
        "--",
        str(t_path.absolute()),
    ]
    run_sh_command(verbose=False, full_cmd=cmd)


def json_to_fbbin(flatc: Path, fbs: Path, j_path: Path, o_path: Optional[Path] = None):
    """Convert JSON flatbuffer to binary.

    flatc: the Path to the flatc compiler program
    fbs: the Path to the fbs (flatbuffer schema) file
    j_path: the Path to the JSON flatbuffer file
    o_path: the output Path where JSON file will be put, if None, it is same as j_path
    """
    if o_path is None:
        o_path = j_path.parent
    cmd = [
        str(flatc.absolute()),
        "-o",
        str(o_path.absolute()),
        "--binary",
        str(fbs.absolute()),
        str(j_path.absolute()),
    ]
    run_sh_command(verbose=False, full_cmd=cmd)


# ------------------------------------------------------------------------------


def main(argv=None):
    """Load and convert supplied file based on file suffix."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--flatc",
        type=Path,
        default="reference_model/build/thirdparty/serialization_lib/third_party/flatbuffers/flatc",
        help="the path to the flatc compiler program",
    )
    parser.add_argument(
        "--fbs",
        type=Path,
        default="conformance_tests/third_party/serialization_lib/schema/tosa.fbs",
        help="the path to the flatbuffer schema",
    )
    parser.add_argument("path", type=Path, help="the path to the file to convert")
    args = parser.parse_args(argv)
    path = args.path

    if not path.is_file():
        print(f"Invalid file to convert - {path}")
        return 2

    if not args.flatc.is_file():
        print(f"Invalid flatc compiler - {args.flatc}")
        return 2

    if not args.fbs.is_file():
        print(f"Invalid flatbuffer schema - {args.fbs}")
        return 2

    try:
        if path.suffix == ".json":
            json_to_fbbin(args.flatc, args.fbs, path)
        else:
            # Have to assume this is a binary flatbuffer file as could have any suffix
            fbbin_to_json(args.flatc, args.fbs, path)
    except RunShCommandError as e:
        print(e)
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
