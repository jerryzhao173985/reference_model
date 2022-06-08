
// Copyright (c) 2020, ARM Limited.
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

#ifndef OPS_OP_FACTORY_H
#define OPS_OP_FACTORY_H

#include "attribute.h"
#include "graph_node.h"
#include "template_types.h"
#include "tosa_serialization_handler.h"

#define DEF_FACTORY_ONE_RANK_ONE_TYPE(OP, RANK, DTYPE)                                                                 \
    case RANK:                                                                                                         \
        return new OP<RANK, DType_##DTYPE>(sgt, attribute, id);

#define DEF_FACTORY_ONE_RANK_TWO_TYPE(OP, RANK, DTYPE1, DTYPE2)                                                        \
    case RANK:                                                                                                         \
        return new OP<RANK, DType_##DTYPE1, DType_##DTYPE2>(sgt, attribute, id);

#define DEF_FACTORY_TWO_RANK_ONE_TYPE(OP, RANK1, RANK2, DTYPE)                                                         \
    case RANK2:                                                                                                        \
        return new OP<RANK1, RANK2, DType_##DTYPE>(sgt, attribute, id);

#define DEF_FACTORY_TWO_RANK_TWO_TYPE(OP, RANK1, RANK2, DTYPE1, DTYPE2)                                                \
    case RANK2:                                                                                                        \
        return new OP<RANK1, RANK2, DType_##DTYPE1, DType_##DTYPE2>(sgt, attribute, id);

#define DEF_FACTORY_ONE_RANK_0_6(OP)                                                                                   \
    switch (inputRank)                                                                                                 \
    {                                                                                                                  \
        case 0:                                                                                                        \
            return new OP<0>(sgt, attribute, id);                                                                      \
        case 1:                                                                                                        \
            return new OP<1>(sgt, attribute, id);                                                                      \
        case 2:                                                                                                        \
            return new OP<2>(sgt, attribute, id);                                                                      \
        case 3:                                                                                                        \
            return new OP<3>(sgt, attribute, id);                                                                      \
        case 4:                                                                                                        \
            return new OP<4>(sgt, attribute, id);                                                                      \
        case 5:                                                                                                        \
            return new OP<5>(sgt, attribute, id);                                                                      \
        case 6:                                                                                                        \
            return new OP<6>(sgt, attribute, id);                                                                      \
    }

#define DEF_FACTORY_ONE_TYPE(OP, DTYPE)                                                                                \
    if (inputDType == DType_##DTYPE)                                                                                   \
    {                                                                                                                  \
        return new OP<DType_##DTYPE>(sgt, attribute, id);                                                              \
    }

#define DEF_FACTORY_TWO_TYPE(OP, DTYPE1, DTYPE2)                                                                       \
    if (inputDType == DType_##DTYPE1 && weightDType == DType_##DTYPE2)                                                 \
    {                                                                                                                  \
        return new OP<DType_##DTYPE1, DType_##DTYPE2>(sgt, attribute, id);                                             \
    }

#define DEF_FACTORY_TWO_TYPE_RESIZE_INT16(OP, DTYPE1, DTYPE2)                                                          \
    if (inputDType == DType_##DTYPE1 && outputDType == DType_##DTYPE2)                                                 \
    {                                                                                                                  \
        return new OP<DType_##DTYPE1, DType_##DTYPE2, int16_t>(sgt, attribute, id);                                    \
    }

#define DEF_FACTORY_TWO_TYPE_RESIZE_FLOAT(OP, DTYPE1, DTYPE2)                                                          \
    if (inputDType == DType_##DTYPE1 && outputDType == DType_##DTYPE2)                                                 \
    {                                                                                                                  \
        return new OP<DType_##DTYPE1, DType_##DTYPE2, float>(sgt, attribute, id);                                      \
    }

#define DEF_FACTORY_RANK0_6_ONE_RANK_ONE_TYPE(OP, DTYPE)                                                               \
    if (inputDType == DType_##DTYPE)                                                                                   \
    {                                                                                                                  \
        switch (inputRank)                                                                                             \
        {                                                                                                              \
            DEF_FACTORY_ONE_RANK_ONE_TYPE(OP, 0, DTYPE)                                                                \
            DEF_FACTORY_ONE_RANK_ONE_TYPE(OP, 1, DTYPE)                                                                \
            DEF_FACTORY_ONE_RANK_ONE_TYPE(OP, 2, DTYPE)                                                                \
            DEF_FACTORY_ONE_RANK_ONE_TYPE(OP, 3, DTYPE)                                                                \
            DEF_FACTORY_ONE_RANK_ONE_TYPE(OP, 4, DTYPE)                                                                \
            DEF_FACTORY_ONE_RANK_ONE_TYPE(OP, 5, DTYPE)                                                                \
            DEF_FACTORY_ONE_RANK_ONE_TYPE(OP, 6, DTYPE)                                                                \
        }                                                                                                              \
    }

#define DEF_FACTORY_RANK1_6_ONE_RANK_ONE_TYPE(OP, DTYPE)                                                               \
    if (inputDType == DType_##DTYPE)                                                                                   \
    {                                                                                                                  \
        switch (inputRank)                                                                                             \
        {                                                                                                              \
            DEF_FACTORY_ONE_RANK_ONE_TYPE(OP, 1, DTYPE)                                                                \
            DEF_FACTORY_ONE_RANK_ONE_TYPE(OP, 2, DTYPE)                                                                \
            DEF_FACTORY_ONE_RANK_ONE_TYPE(OP, 3, DTYPE)                                                                \
            DEF_FACTORY_ONE_RANK_ONE_TYPE(OP, 4, DTYPE)                                                                \
            DEF_FACTORY_ONE_RANK_ONE_TYPE(OP, 5, DTYPE)                                                                \
            DEF_FACTORY_ONE_RANK_ONE_TYPE(OP, 6, DTYPE)                                                                \
        }                                                                                                              \
    }

#define DEF_FACTORY_RANK0_6_ONE_RANK_TWO_TYPE(OP, DTYPE1, DTYPE2)                                                      \
    if (inputDType == DType_##DTYPE1 && outputDType == DType_##DTYPE2)                                                 \
    {                                                                                                                  \
        switch (inputRank)                                                                                             \
        {                                                                                                              \
            DEF_FACTORY_ONE_RANK_TWO_TYPE(OP, 0, DTYPE1, DTYPE2)                                                       \
            DEF_FACTORY_ONE_RANK_TWO_TYPE(OP, 1, DTYPE1, DTYPE2)                                                       \
            DEF_FACTORY_ONE_RANK_TWO_TYPE(OP, 2, DTYPE1, DTYPE2)                                                       \
            DEF_FACTORY_ONE_RANK_TWO_TYPE(OP, 3, DTYPE1, DTYPE2)                                                       \
            DEF_FACTORY_ONE_RANK_TWO_TYPE(OP, 4, DTYPE1, DTYPE2)                                                       \
            DEF_FACTORY_ONE_RANK_TWO_TYPE(OP, 5, DTYPE1, DTYPE2)                                                       \
            DEF_FACTORY_ONE_RANK_TWO_TYPE(OP, 6, DTYPE1, DTYPE2)                                                       \
        }                                                                                                              \
    }

#define DEF_FACTORY_RESHAPE(OP, DTYPE)                                                                                 \
    if (inputDType == DType_##DTYPE && outputDType == DType_##DTYPE)                                                   \
    {                                                                                                                  \
        switch (inputRank)                                                                                             \
        {                                                                                                              \
            case 0:                                                                                                    \
            {                                                                                                          \
                switch (outputRank)                                                                                    \
                {                                                                                                      \
                    DEF_FACTORY_TWO_RANK_ONE_TYPE(OP, 0, 0, DTYPE)                                                     \
                    DEF_FACTORY_TWO_RANK_ONE_TYPE(OP, 0, 1, DTYPE)                                                     \
                    DEF_FACTORY_TWO_RANK_ONE_TYPE(OP, 0, 2, DTYPE)                                                     \
                    DEF_FACTORY_TWO_RANK_ONE_TYPE(OP, 0, 3, DTYPE)                                                     \
                    DEF_FACTORY_TWO_RANK_ONE_TYPE(OP, 0, 4, DTYPE)                                                     \
                    DEF_FACTORY_TWO_RANK_ONE_TYPE(OP, 0, 5, DTYPE)                                                     \
                    DEF_FACTORY_TWO_RANK_ONE_TYPE(OP, 0, 6, DTYPE)                                                     \
                }                                                                                                      \
            }                                                                                                          \
            case 1:                                                                                                    \
            {                                                                                                          \
                switch (outputRank)                                                                                    \
                {                                                                                                      \
                    DEF_FACTORY_TWO_RANK_ONE_TYPE(OP, 1, 0, DTYPE)                                                     \
                    DEF_FACTORY_TWO_RANK_ONE_TYPE(OP, 1, 1, DTYPE)                                                     \
                    DEF_FACTORY_TWO_RANK_ONE_TYPE(OP, 1, 2, DTYPE)                                                     \
                    DEF_FACTORY_TWO_RANK_ONE_TYPE(OP, 1, 3, DTYPE)                                                     \
                    DEF_FACTORY_TWO_RANK_ONE_TYPE(OP, 1, 4, DTYPE)                                                     \
                    DEF_FACTORY_TWO_RANK_ONE_TYPE(OP, 1, 5, DTYPE)                                                     \
                    DEF_FACTORY_TWO_RANK_ONE_TYPE(OP, 1, 6, DTYPE)                                                     \
                }                                                                                                      \
            }                                                                                                          \
            case 2:                                                                                                    \
            {                                                                                                          \
                switch (outputRank)                                                                                    \
                {                                                                                                      \
                    DEF_FACTORY_TWO_RANK_ONE_TYPE(OP, 2, 0, DTYPE)                                                     \
                    DEF_FACTORY_TWO_RANK_ONE_TYPE(OP, 2, 1, DTYPE)                                                     \
                    DEF_FACTORY_TWO_RANK_ONE_TYPE(OP, 2, 2, DTYPE)                                                     \
                    DEF_FACTORY_TWO_RANK_ONE_TYPE(OP, 2, 3, DTYPE)                                                     \
                    DEF_FACTORY_TWO_RANK_ONE_TYPE(OP, 2, 4, DTYPE)                                                     \
                    DEF_FACTORY_TWO_RANK_ONE_TYPE(OP, 2, 5, DTYPE)                                                     \
                    DEF_FACTORY_TWO_RANK_ONE_TYPE(OP, 2, 6, DTYPE)                                                     \
                }                                                                                                      \
            }                                                                                                          \
            case 3:                                                                                                    \
            {                                                                                                          \
                switch (outputRank)                                                                                    \
                {                                                                                                      \
                    DEF_FACTORY_TWO_RANK_ONE_TYPE(OP, 3, 0, DTYPE)                                                     \
                    DEF_FACTORY_TWO_RANK_ONE_TYPE(OP, 3, 1, DTYPE)                                                     \
                    DEF_FACTORY_TWO_RANK_ONE_TYPE(OP, 3, 2, DTYPE)                                                     \
                    DEF_FACTORY_TWO_RANK_ONE_TYPE(OP, 3, 3, DTYPE)                                                     \
                    DEF_FACTORY_TWO_RANK_ONE_TYPE(OP, 3, 4, DTYPE)                                                     \
                    DEF_FACTORY_TWO_RANK_ONE_TYPE(OP, 3, 5, DTYPE)                                                     \
                    DEF_FACTORY_TWO_RANK_ONE_TYPE(OP, 3, 6, DTYPE)                                                     \
                }                                                                                                      \
            }                                                                                                          \
            case 4:                                                                                                    \
            {                                                                                                          \
                switch (outputRank)                                                                                    \
                {                                                                                                      \
                    DEF_FACTORY_TWO_RANK_ONE_TYPE(OP, 4, 0, DTYPE)                                                     \
                    DEF_FACTORY_TWO_RANK_ONE_TYPE(OP, 4, 1, DTYPE)                                                     \
                    DEF_FACTORY_TWO_RANK_ONE_TYPE(OP, 4, 2, DTYPE)                                                     \
                    DEF_FACTORY_TWO_RANK_ONE_TYPE(OP, 4, 3, DTYPE)                                                     \
                    DEF_FACTORY_TWO_RANK_ONE_TYPE(OP, 4, 4, DTYPE)                                                     \
                    DEF_FACTORY_TWO_RANK_ONE_TYPE(OP, 4, 5, DTYPE)                                                     \
                    DEF_FACTORY_TWO_RANK_ONE_TYPE(OP, 4, 6, DTYPE)                                                     \
                }                                                                                                      \
            }                                                                                                          \
            case 5:                                                                                                    \
            {                                                                                                          \
                switch (outputRank)                                                                                    \
                {                                                                                                      \
                    DEF_FACTORY_TWO_RANK_ONE_TYPE(OP, 5, 0, DTYPE)                                                     \
                    DEF_FACTORY_TWO_RANK_ONE_TYPE(OP, 5, 1, DTYPE)                                                     \
                    DEF_FACTORY_TWO_RANK_ONE_TYPE(OP, 5, 2, DTYPE)                                                     \
                    DEF_FACTORY_TWO_RANK_ONE_TYPE(OP, 5, 3, DTYPE)                                                     \
                    DEF_FACTORY_TWO_RANK_ONE_TYPE(OP, 5, 4, DTYPE)                                                     \
                    DEF_FACTORY_TWO_RANK_ONE_TYPE(OP, 5, 5, DTYPE)                                                     \
                    DEF_FACTORY_TWO_RANK_ONE_TYPE(OP, 5, 6, DTYPE)                                                     \
                }                                                                                                      \
            }                                                                                                          \
            case 6:                                                                                                    \
            {                                                                                                          \
                switch (outputRank)                                                                                    \
                {                                                                                                      \
                    DEF_FACTORY_TWO_RANK_ONE_TYPE(OP, 6, 0, DTYPE)                                                     \
                    DEF_FACTORY_TWO_RANK_ONE_TYPE(OP, 6, 1, DTYPE)                                                     \
                    DEF_FACTORY_TWO_RANK_ONE_TYPE(OP, 6, 2, DTYPE)                                                     \
                    DEF_FACTORY_TWO_RANK_ONE_TYPE(OP, 6, 3, DTYPE)                                                     \
                    DEF_FACTORY_TWO_RANK_ONE_TYPE(OP, 6, 4, DTYPE)                                                     \
                    DEF_FACTORY_TWO_RANK_ONE_TYPE(OP, 6, 5, DTYPE)                                                     \
                    DEF_FACTORY_TWO_RANK_ONE_TYPE(OP, 6, 6, DTYPE)                                                     \
                }                                                                                                      \
            }                                                                                                          \
        }                                                                                                              \
    }

namespace TosaReference
{

class SubgraphTraverser;
class GraphNode;

class OpFactory
{
public:
    static GraphNode* newOp(SubgraphTraverser* sgt,
                            tosa::TosaSerializationHandler* tsh,
                            tosa::Op opType,
                            tosa::TosaAttributeBase* attribute,
                            uint64_t id,
                            tosa::DType inputDType,
                            int inputRank,
                            tosa::DType outputDType,
                            int outputRank,
                            tosa::DType weightDType,
                            int weightRank);
};
};    // namespace TosaReference

#endif
