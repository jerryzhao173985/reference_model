"""Conversion utility from binary numpy files to JSON and the reverse."""
# Copyright (c) 2021-2022, ARM Limited.
# SPDX-License-Identifier: Apache-2.0
import json
from pathlib import Path
from typing import Optional
from typing import Union

import numpy as np


class NumpyArrayEncoder(json.JSONEncoder):
    """A JSON encoder for Numpy data types."""

    def default(self, obj):
        """Encode default operation."""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.float32):
            return float(obj)
        elif isinstance(obj, np.float16):
            return np.float16(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyArrayEncoder, self).default(obj)


def get_shape(t: Union[list, tuple]):
    """Get the shape of an N-Dimensional tensor."""
    # TODO: validate shape is consistent for all rows and ccolumns
    if isinstance(t, (list, tuple)) and t:
        return [len(t)] + get_shape(t[0])
    return []


def npy_to_json(n_path: Path, j_path: Optional[Path] = None):
    """Load a numpy data file and save it as a JSON file.

    n_path: the Path to the numpy file
    j_path: the Path to the JSON file, if None, it is derived from n_path
    """
    if not j_path:
        j_path = n_path.parent / (n_path.stem + ".json")
    with open(n_path, "rb") as fd:
        data = np.load(fd)
    jdata = {
        "type": data.dtype.name,
        "data": data.tolist(),
    }
    with open(j_path, "w") as fp:
        json.dump(jdata, fp, indent=2)


def json_to_npy(j_path: Path, n_path: Optional[Path] = None):
    """Load a JSON file and save it as a numpy data file.

    j_path: the Path to the JSON file
    n_path: the Path to the numpy file, if None, it is derived from j_path
    """
    if not n_path:
        n_path = j_path.parent / (j_path.stem + ".npy")
    with open(j_path, "rb") as fd:
        jdata = json.load(fd)
    raw_data = jdata["data"]
    raw_type = jdata["type"]
    shape = get_shape(raw_data)
    data = np.asarray(raw_data).reshape(shape).astype(raw_type)
    with open(n_path, "wb") as fd:
        np.save(fd, data)


# ------------------------------------------------------------------------------


def test():
    """Test conversion routines."""
    shape = [2, 3, 4]
    elements = 1
    for i in shape:
        elements *= i

    # file names
    n_path = Path("data.npy")
    j_path = Path("data.json")
    j2n_path = Path("data_j2n.npy")

    datatypes = [
        np.bool_,
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        np.uint8,
        np.uint16,
        np.uint32,
        np.uint64,
        np.float16,
        np.float32,
        np.float64,
        # np.float128,
        # np.complex64,
        # np.complex128,
        # np.complex256,
        # np.datetime64,
        # np.str,
    ]

    for data_type in datatypes:
        dt = np.dtype(data_type)
        print(data_type, dt, dt.char, dt.num, dt.name, dt.str)

        # create a tensor of the given shape
        tensor = np.arange(elements).reshape(shape).astype(data_type)
        # print(tensor)

        # save the tensor in a binary numpy file
        with open(n_path, "wb") as fd:
            np.save(fd, tensor)

        # read back the numpy file for verification
        with open(n_path, "rb") as fd:
            tensor1 = np.load(fd)

        # confirm the loaded tensor matches the original
        assert tensor.shape == tensor1.shape
        assert tensor.dtype == tensor1.dtype
        assert (tensor == tensor1).all()

        # convert the numpy file to json
        npy_to_json(n_path, j_path)

        # convert the json file to numpy
        json_to_npy(j_path, j2n_path)

        # read back the json-to-numpy file for verification
        with open(j2n_path, "rb") as fd:
            tensor1 = np.load(fd)

        # confirm the loaded tensor matches the original
        assert tensor.shape == tensor1.shape
        assert tensor.dtype == tensor1.dtype
        assert (tensor == tensor1).all()

    # delete the files, if no problems were found
    # they are left for debugging if any of the asserts failed
    n_path.unlink()
    j_path.unlink()
    j2n_path.unlink()
    return 0


def main(argv=None):
    """Load and convert supplied file based on file suffix."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path", type=Path, help="the path to the file to convert, or 'test'"
    )
    args = parser.parse_args(argv)
    path = args.path
    if str(path) == "test":
        print("test")
        return test()

    if not path.is_file():
        print(f"Invalid file - {path}")
        return 2

    if path.suffix == ".npy":
        npy_to_json(path)
    elif path.suffix == ".json":
        json_to_npy(path)
    else:
        print("Unknown file type - {path.suffix}")
        return 2

    return 0


if __name__ == "__main__":
    exit(main())
