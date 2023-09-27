# Copyright (c) 2024, ARM Limited.
# SPDX-License-Identifier: Apache-2.0
import hashlib
import logging

import generator.tosa_utils as gtu
import numpy as np
from tosa.DType import DType

logging.basicConfig()
logger = logging.getLogger("tosa_verif_build_tests")


class TosaRandomGenerator(np.random.Generator):
    """Equivalent to numpy.default_rng, with support for TOSA data types"""

    def __init__(self, seed, restrict_range_by_type={}):
        """Create random generator with TOSA type support.

        seed: integer seed
        restrict_range_by_type: see TosaHashRandomGenerator.__init__()
        """
        self._restrict_range_by_type = restrict_range_by_type
        self._seed = int(seed)
        self._bitgen = np.random.PCG64(self._seed)
        super().__init__(self._bitgen)

    @property
    def seed(self):
        return self._seed

    @property
    def hexSeed(self):
        return hex(self._seed)

    def dTypeRange(self, dtype, high_inclusive=False):
        """Returns range tuple for given dtype.

        dtype: DType
        high_inclusive: True for inclusive high values
        Returns: dtype value range boundaries tuple (low, high)
            The high boundary is excluded in the range unless high_inclusive is True
        """
        if dtype in self._restrict_range_by_type:
            rng = self._restrict_range_by_type[dtype]
        elif dtype == DType.BOOL:
            rng = (0, 2)
        elif dtype == DType.UINT8:
            rng = (0, 256)
        elif dtype == DType.UINT16:
            rng = (0, 65536)
        elif dtype == DType.INT4:
            # TOSA specific INT4 weight range from -7 to 7
            rng = (-7, 8)
        elif dtype == DType.INT8:
            rng = (-128, 128)
        elif dtype == DType.INT16:
            rng = (-32768, 32768)
        elif dtype == DType.INT32:
            rng = (-(1 << 31), (1 << 31))
        elif dtype == DType.INT48:
            rng = (-(1 << 47), (1 << 47))
        else:
            # Float types and SHAPE should be in _restrict_range_by_type dict
            raise Exception("Unknown supported dtype: {}".format(dtype))

        if dtype in (DType.FP16, DType.BF16, DType.FP32, DType.FP8E4M3, DType.FP8E5M2):
            # Floating point - range is always inclusive
            return rng
        else:
            # Integer
            if not high_inclusive:
                # Exclusive high: low <= range < high
                return rng
            else:
                # Inclusive range: low <= range <= high
                return (rng[0], rng[1] - 1)

    def randInt(self, low=0, high=256):
        return np.int32(self.integers(low=low, high=high, size=1))[0]

    def randNumberDType(self, dtype):
        low, high = self.dTypeRange(dtype)

        if dtype == DType.FP32:
            return np.float32(self.uniform(low=low, high=high))
        elif dtype == DType.FP16:
            return np.float16(self.uniform(low=low, high=high))
        elif dtype == DType.BF16:
            rand_f32 = np.float32(self.uniform(low=low, high=high))
            return gtu.vect_f32_to_bf16(rand_f32)
        elif dtype == DType.FP8E4M3:
            rand_f32 = np.float32(self.uniform(low=low, high=high))
            return gtu.vect_f32_to_fp8e4m3(rand_f32)
        elif dtype == DType.FP8E5M2:
            rand_f32 = np.float32(self.uniform(low=low, high=high))
            return gtu.vect_f32_to_fp8e5m2(rand_f32)
        elif dtype == DType.BOOL:
            return self.choice([False, True])
        elif dtype == DType.INT48 or dtype == DType.SHAPE:
            # Special size
            return np.int64(self.integers(low, high, size=1))[0]

        return np.int32(self.integers(low, high, size=1))[0]

    def randTensor(self, shape, dtype, data_range=None):
        if data_range is None:
            low, high = self.dTypeRange(dtype)
        else:
            low, high = data_range

        if dtype == DType.BOOL:
            return np.bool_(self.choice(a=[False, True], size=shape))
        elif dtype == DType.INT4:
            return np.int8(self.integers(low=low, high=high, size=shape))
        elif dtype == DType.INT8:
            return np.int8(self.integers(low=low, high=high, size=shape))
        elif dtype == DType.UINT8:
            return np.uint8(self.integers(low=low, high=high, size=shape))
        elif dtype == DType.INT16:
            return np.int16(self.integers(low=low, high=high, size=shape))
        elif dtype == DType.UINT16:
            return np.uint16(self.integers(low=low, high=high, size=shape))
        elif dtype in (DType.INT48, DType.SHAPE):
            return np.int64(self.integers(low=low, high=high, size=shape))
        elif dtype in (
            DType.FP16,
            DType.BF16,
            DType.FP32,
            DType.FP8E4M3,
            DType.FP8E5M2,
        ):
            f_tensor = self.uniform(low=low, high=high, size=shape)

            if dtype == DType.FP16:
                return np.float16(f_tensor)
            else:
                f32_tensor = np.float32(f_tensor)
                if dtype == DType.BF16:
                    # Floor the last 16 bits of each f32 value
                    return np.float32(gtu.vect_f32_to_bf16(f32_tensor))
                elif dtype == DType.FP8E4M3:
                    return np.float32(gtu.vect_f32_to_fp8e4m3(f32_tensor))
                elif dtype == DType.FP8E5M2:
                    return np.float32(gtu.vect_f32_to_fp8e5m2(f32_tensor))
                else:
                    return f32_tensor
        else:
            # All other integer types
            return np.int32(self.integers(low=low, high=high, size=shape))


class TosaHashRandomGenerator(TosaRandomGenerator):
    """Hash seeded TOSA random number generator."""

    def __init__(self, seed, seed_list, restrict_range_by_type={}):
        """Create TOSA random generator seeding it with a hashable list.

        seed: integer starting seed
        seed_list: list of hashable items to add to starting seed
        restrict_range_by_type: dictionary of DTypes with (low, high) range tuples
            This must contain entries for SHAPE and all Floating Point data types.
            NOTE: For integers, the high value must be the exclusive value
        """
        # Convert seed_list to strings
        seed_strings_list = [str(s) for s in seed_list]
        # Create a single string and create hash
        self._seed_string = "__".join(seed_strings_list)
        self._hash = hashlib.md5(bytes(self._seed_string, "utf-8"))
        # Add the hash value to the given seed
        seed += int(self._hash.hexdigest(), 16)

        logger.debug(f"Seed={seed} Seed string={self._seed_string}")
        super().__init__(seed, restrict_range_by_type)
