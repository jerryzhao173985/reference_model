
// Copyright (c) 2020-2022, ARM Limited.
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

#include "comparison.h"
#include "arith_util.h"
#include "quant_util.h"
#include "template_types.h"

using namespace TosaReference;
using namespace Eigen;
using namespace tosa;

template <int Rank, DType Dtype>
int OpEqual<Rank, Dtype>::register_fcn()
{
    switch (Dtype)
    {
        case DType_FP16:
        case DType_FLOAT:
        case DType_INT32:
            this->fcn = [](InEigenType a, InEigenType b) -> OutEigenType { return a == b; };
            break;
        default:
            ERROR_IF(true, "unsupported DType %s", EnumNamesDType()[Dtype]);
    }

    return 0;
}

template <int Rank, DType Dtype>
int OpGreater<Rank, Dtype>::register_fcn()
{
    switch (Dtype)
    {
        case DType_FP16:
        case DType_FLOAT:
        case DType_INT32:
            this->fcn = [](InEigenType a, InEigenType b) -> OutEigenType { return a > b; };
            break;
        default:
            ERROR_IF(true, "unsupported DType %s", EnumNamesDType()[Dtype]);
    }

    return 0;
}

template <int Rank, DType Dtype>
int OpGreaterEqual<Rank, Dtype>::register_fcn()
{
    switch (Dtype)
    {
        case DType_FP16:
        case DType_FLOAT:
        case DType_INT32:
            this->fcn = [](InEigenType a, InEigenType b) -> OutEigenType { return a >= b; };
            break;
        default:
            ERROR_IF(true, "unsupported DType %s", EnumNamesDType()[Dtype]);
    }

    return 0;
}

// template explicit instantiation
DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpEqual, FP16);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpEqual, FLOAT);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpEqual, INT32);

DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpGreater, FP16);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpGreater, FLOAT);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpGreater, INT32);

DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpGreaterEqual, FP16);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpGreaterEqual, FLOAT);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpGreaterEqual, INT32);
