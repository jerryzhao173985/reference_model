# Copyright (c) 2024, ARM Limited.
# SPDX-License-Identifier: Apache-2.0


class TosaProfiles:
    TosaProINT = "tosa-pro-int"
    TosaProFP = "tosa-pro-fp"
    TosaExtInt16 = "tosa-ext-int16"
    TosaExtInt4 = "tosa-ext-int4"
    TosaExtBF16 = "tosa-ext-bf16"
    TosaExtFP8E4M3 = "tosa-ext-fp8e4m3"
    TosaExtFP8E5M2 = "tosa-ext-fp8e5m2"
    TosaExtFFT = "tosa-ext-fft"
    TosaExtVariable = "tosa-ext-variable"
    TosaExtShape = "tosa-ext-shape"

    @staticmethod
    def profiles():
        return [TosaProfiles.TosaProINT, TosaProfiles.TosaProFP]

    @staticmethod
    def extensions():
        return [
            TosaProfiles.TosaExtInt16,
            TosaProfiles.TosaExtInt4,
            TosaProfiles.TosaExtBF16,
            TosaProfiles.TosaExtFP8E4M3,
            TosaProfiles.TosaExtFP8E5M2,
            TosaProfiles.TosaExtFFT,
            TosaProfiles.TosaExtVariable,
            TosaProfiles.TosaExtShape,
        ]

    @staticmethod
    def all():
        return TosaProfiles.profiles() + TosaProfiles.extensions()
