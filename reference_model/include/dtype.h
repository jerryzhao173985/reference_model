// Copyright (c) 2023-2024, ARM Limited.
//
//    Licensed under the Apache License, Version 2.0 (the "License");
//    you may not use this file except in compliance with the License.
//    You may obtain a copy of the License at
//
//         http://www.apache.org/licenses/LICENSE-2.0
//
//    Unless required by applicable law or agreed to in writing, software
//    distributed under the License is distributed on an "AS IS" BASIS,
//    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//    See the License for the specific language governing permissions and
//    limitations under the License.

#ifndef TOSA_REFERENCE_DTYPE_H
#define TOSA_REFERENCE_DTYPE_H

#include "model_common.h"
#include "tosa_generated.h"
#include <cstdint>

using namespace tosa;

namespace TosaReference
{

// Reference Model version of tosa.fbs enum DType
// Plus a FP64 data type for precise mode.
enum TOSA_REF_TYPE : uint32_t
{
    TOSA_REF_TYPE_UNKNOWN = 0,
    TOSA_REF_TYPE_BOOL    = 1,
    TOSA_REF_TYPE_UINT8   = 2,
    TOSA_REF_TYPE_INT4    = 3,
    TOSA_REF_TYPE_INT8    = 4,
    TOSA_REF_TYPE_INT16   = 5,
    TOSA_REF_TYPE_INT32   = 6,
    TOSA_REF_TYPE_INT48   = 7,
    TOSA_REF_TYPE_FP32    = 8,
    TOSA_REF_TYPE_UINT16  = 9,
    TOSA_REF_TYPE_FP16    = 10,
    TOSA_REF_TYPE_BF16    = 11,
    TOSA_REF_TYPE_SHAPE   = 12,
    TOSA_REF_TYPE_FP8E4M3 = 13,
    TOSA_REF_TYPE_FP8E5M2 = 14,
    TOSA_REF_TYPE_FP64    = 99,    // FP64 is special: add new data types above
};

inline const char* EnumNameTOSAREFTYPE(TOSA_REF_TYPE e)
{
    switch (e)
    {
        case TOSA_REF_TYPE_UNKNOWN:
            return EnumNameDType(DType_UNKNOWN);
        case TOSA_REF_TYPE_BOOL:
            return EnumNameDType(DType_BOOL);
        case TOSA_REF_TYPE_UINT8:
            return EnumNameDType(DType_UINT8);
        case TOSA_REF_TYPE_INT4:
            return EnumNameDType(DType_INT4);
        case TOSA_REF_TYPE_INT8:
            return EnumNameDType(DType_INT8);
        case TOSA_REF_TYPE_INT16:
            return EnumNameDType(DType_INT16);
        case TOSA_REF_TYPE_INT32:
            return EnumNameDType(DType_INT32);
        case TOSA_REF_TYPE_INT48:
            return EnumNameDType(DType_INT48);
        case TOSA_REF_TYPE_FP32:
            return EnumNameDType(DType_FP32);
        case TOSA_REF_TYPE_UINT16:
            return EnumNameDType(DType_UINT16);
        case TOSA_REF_TYPE_FP16:
            return EnumNameDType(DType_FP16);
        case TOSA_REF_TYPE_BF16:
            return EnumNameDType(DType_BF16);
        case TOSA_REF_TYPE_SHAPE:
            return EnumNameDType(DType_SHAPE);
        case TOSA_REF_TYPE_FP8E4M3:
            return EnumNameDType(DType_FP8E4M3);
        case TOSA_REF_TYPE_FP8E5M2:
            return EnumNameDType(DType_FP8E5M2);
        case TOSA_REF_TYPE_FP64:
            return "FP64";
        default:
            assert(false);
            return "ERROR";
    }
}

// return corresponding TOSA_REF_TYPE for DType
inline TOSA_REF_TYPE ConvertDType(const DType dtype)
{
    assert(DType_MAX == DType_FP8E5M2);    // must update whenever DType_MAX changes

    if (g_func_config.precise_mode)
    {
        // in precise mode, convert all floating DType to TOSA_REF_TYPE_FP64
        switch (dtype)
        {
            case DType_FP16:
            case DType_FP32:
            case DType_BF16:
            case DType_FP8E4M3:
            case DType_FP8E5M2:
                return TOSA_REF_TYPE_FP64;
            default:
                break;
        }
    }

    switch (dtype)
    {
        case DType_BOOL:
            return TOSA_REF_TYPE_BOOL;
        case DType_UINT8:
            return TOSA_REF_TYPE_UINT8;
        case DType_INT4:
            return TOSA_REF_TYPE_INT4;
        case DType_INT8:
            return TOSA_REF_TYPE_INT8;
        case DType_INT16:
            return TOSA_REF_TYPE_INT16;
        case DType_INT32:
            return TOSA_REF_TYPE_INT32;
        case DType_INT48:
            return TOSA_REF_TYPE_INT48;
        case DType_FP32:
            return TOSA_REF_TYPE_FP32;
        case DType_UINT16:
            return TOSA_REF_TYPE_UINT16;
        case DType_FP16:
            return TOSA_REF_TYPE_FP16;
        case DType_BF16:
            return TOSA_REF_TYPE_BF16;
        case DType_SHAPE:
            return TOSA_REF_TYPE_SHAPE;
        case DType_FP8E4M3:
            return TOSA_REF_TYPE_FP8E4M3;
        case DType_FP8E5M2:
            return TOSA_REF_TYPE_FP8E5M2;
        default:
            break;
    }
    return TOSA_REF_TYPE_UNKNOWN;
}

template <TOSA_REF_TYPE Dtype>
bool IsSignedInt()
{
    switch (Dtype)
    {
        case TOSA_REF_TYPE_INT4:
        case TOSA_REF_TYPE_INT8:
        case TOSA_REF_TYPE_INT16:
        case TOSA_REF_TYPE_INT32:
        case TOSA_REF_TYPE_INT48:
            return true;

        case TOSA_REF_TYPE_UINT8:
        case TOSA_REF_TYPE_UINT16:
            return false;

        case TOSA_REF_TYPE_BOOL:
        case TOSA_REF_TYPE_FP32:
        case TOSA_REF_TYPE_FP16:
        case TOSA_REF_TYPE_BF16:
        case TOSA_REF_TYPE_SHAPE:
        case TOSA_REF_TYPE_FP8E4M3:
        case TOSA_REF_TYPE_FP8E5M2:
        default:
            FATAL_ERROR("dtype is not an integer type");
            break;
    }
}

};    // namespace TosaReference

#endif
