
// Copyright (c) 2020-2021, ARM Limited.
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

#include "activation_funcs.h"
#include "quant_util.h"
#include "template_types.h"
#include <cmath>

using namespace TosaReference;
using namespace Eigen;
using namespace tosa;

template <int Rank, DType Dtype>
int OpClamp<Rank, Dtype>::register_fcn()
{
    switch (Dtype)
    {
        case DType_FLOAT:
        {
            InEigenType min = (InEigenType)attribute->min_fp();
            InEigenType max = (InEigenType)attribute->max_fp();
            ERROR_IF(max < min, "OpClamp: max smaller than min");

            this->fcn = [min, max](InEigenType a) -> OutEigenType { return a <= min ? min : a >= max ? max : a; };
        }
        break;
        case DType_INT8:
        case DType_INT16:
        {
            InEigenType min = (InEigenType)attribute->min_int();
            InEigenType max = (InEigenType)attribute->max_int();
            ERROR_IF(max < min, "OpClamp: max smaller than min");
            this->fcn = [min, max](InEigenType a) -> OutEigenType { return a <= min ? min : a >= max ? max : a; };
        }
        break;
        default:
            ERROR_IF(true, "unsupported DType %s", EnumNamesDType()[Dtype]);
    }

    return 0;
}

template <int Rank, DType Dtype>
int OpSigmoid<Rank, Dtype>::register_fcn()
{
    switch (Dtype)
    {
        case DType_FLOAT:
            this->fcn = [](InEigenType a) -> OutEigenType { return (1.0 / (1.0 + (expf(-1.0 * a)))); };
            break;
        default:
            ERROR_IF(true, "unsupported DType %s", EnumNamesDType()[Dtype]);
    }

    return 0;
}

template <int Rank, DType Dtype>
int OpTanh<Rank, Dtype>::register_fcn()
{
    switch (Dtype)
    {
        case DType_FLOAT:
            this->fcn = [](InEigenType a) -> OutEigenType { return tanhf(a); };
            break;
        default:
            ERROR_IF(true, "unsupported DType %s", EnumNamesDType()[Dtype]);
    }

    return 0;
}

// template explicit instantiation
DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpClamp, FLOAT);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpClamp, INT8);
DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpClamp, INT16);

DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpSigmoid, FLOAT);

DEF_INSTANTIATE_RANK0_6_ONE_RANK_ONE_TYPE(OpTanh, FLOAT);
