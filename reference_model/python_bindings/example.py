# Copyright (c) 2024, ARM Limited.
# SPDX-License-Identifier: Apache-2.0
import pathlib

import numpy as np
import tosa_reference_model
from serializer import tosa_serializer as ts

# If the tosa_reference_model import fails, navigate to the
# tosa_reference_model.cpython-{version}.so file in the build folder and run:
# "export PYTHONPATH=$PYTHONPATH:$(pwd)"


if __name__ == "__main__":
    # Creating a TOSA graph with the serialization library.
    # Graphs can also be read as binary from flatbuffers files and used directly.
    handler = ts.TosaSerializer(pathlib.Path(__file__).parent)
    handler.addInputTensor(ts.TosaSerializerTensor("input_0", [3, 3], ts.DType.INT32))
    handler.addInputTensor(ts.TosaSerializerTensor("input_1", [3, 3], ts.DType.INT32))
    inter_tensor = handler.addIntermediate([3, 3], ts.DType.INT32)
    handler.addOperator(ts.TosaOp.Op().SUB, ["input_0", "input_1"], [inter_tensor.name])
    output_tensor = handler.addOutput([3, 3], ts.DType.INT32)
    handler.addOperator(
        ts.TosaOp.Op().SUB, [inter_tensor.name, "input_1"], [output_tensor.name]
    )
    graph = handler.serialize()

    # Creating random input arrays
    a0 = np.random.uniform(0, 20, (3, 3)).astype(np.int32)

    # Because of the transpose, this array is not stored in row-major order, so the
    # reference model has to copy it internally to read it correctly.
    a1 = np.random.uniform(0, 20, (3, 3)).astype(np.int32).T

    expected = a0 - a1 - a1

    # Running the reference model through Python/Pybind11
    outputs, status = tosa_reference_model.run(
        graph,
        [a0, a1],
        verbosity="HIGH",
        debug_filename=str(pathlib.Path(__file__).parent / "debug.txt"),
        dump_intermediates=1,
    )

    assert status == tosa_reference_model.GraphStatus.TOSA_VALID

    # This step is only necessary for certain datatypes that the C++ runner can't
    # natively work with; it is not required for int32, but it's good practice.
    out = outputs[0].view(dtype=np.int32)

    print(
        f"""numpy arrays ---
input_0:
{a0}
input_1:
{a1}
expected:
{expected}
raw out (may be bytearray):
{outputs[0]}
out:
{out}
--- """
    )

    assert np.array_equal(out, expected)
