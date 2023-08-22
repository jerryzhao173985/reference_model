"""Conversion utility from flatbuffer JSON files to binary and the reverse."""
# Copyright (c) 2021-2023, ARM Limited.
# SPDX-License-Identifier: Apache-2.0
import re
from pathlib import Path
from typing import Optional

from runner.run_command import run_sh_command
from runner.run_command import RunShCommandError

MAX_LINE_LEN = 120
MAX_INDENT_LEN = 20


def json_squeeze(json_path: Path):
    """File compression for JSONs, reducing spaces used for number lists."""
    # Move existing file to a new name
    temp_path = json_path.with_suffix(".json_unsqueezed")
    json_path.rename(temp_path)
    # Now read the original file and write a smaller output with less new lines/spaces
    with temp_path.open("r") as tfd:
        with json_path.open("w") as jfd:
            found = False

            for line in tfd:
                # Find lines that are part of number lists
                match = re.match(r"(\s+)(-?[0-9]+),?", line)
                if match:
                    # Found a line with just a number on it (and optional comma)
                    if not found:
                        # New list of numbers
                        numbers = []
                        # Save indent (upto maximum)
                        indent = match.group(1)[0:MAX_INDENT_LEN]
                        found = True
                    numbers.append(match.group(2))
                else:
                    # Found a line without just a number
                    if found:
                        # Format the list of numbers recorded into a concise output
                        # with multiple numbers on a single line, rather than one per line
                        numbers_str = indent
                        for num in numbers:
                            nums = f"{num},"
                            if len(numbers_str) + len(nums) > MAX_LINE_LEN:
                                print(numbers_str, file=jfd)
                                numbers_str = indent
                            numbers_str += nums
                        # print all but the last comma
                        print(numbers_str[:-1], file=jfd)

                        found = False
                    # print the line we just read (that wasn't just a number)
                    print(line, file=jfd, end="")

    # Remove the uncompressed version
    temp_path.unlink()


def fbbin_to_json(
    flatc: Path,
    fbs: Path,
    t_path: Path,
    o_path: Optional[Path] = None,
    squeeze: Optional[bool] = True,
):
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
        "--strict-json",
        "--defaults-json",
        "--raw-binary",
        str(fbs.absolute()),
        "--",
        str(t_path.absolute()),
    ]
    run_sh_command(verbose=False, full_cmd=cmd)
    if squeeze:
        json_path = (o_path / t_path.name).with_suffix(".json")
        json_squeeze(json_path)


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
        "--no-squeeze", action="store_true", help="no compression of json output"
    )
    parser.add_argument(
        "--flatc",
        type=Path,
        default=(
            "reference_model/build/thirdparty/serialization_lib/"
            "third_party/flatbuffers/flatc"
        ),
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
            fbbin_to_json(args.flatc, args.fbs, path, squeeze=(not args.no_squeeze))
    except RunShCommandError as e:
        print(e)
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
