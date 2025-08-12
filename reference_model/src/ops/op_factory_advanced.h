// Copyright (c) 2025 ARM Limited.
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

#ifndef OP_FACTORY_ADVANCED_H
#define OP_FACTORY_ADVANCED_H

#include "op_factory.h"
#include "advanced_tensor_ops.h"

// Advanced operation factory macros
#define DEF_FACTORY_ADVANCED_OP_ONE_TYPE(OP, DTYPE)                                                                    \
    if (inputDTYPE == TOSA_REF_TYPE_##DTYPE)                                                                           \
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

#define DEF_FACTORY_ADVANCED_TENSOR_FUSION()                                                                           \
    DEF_FACTORY_ADVANCED_OP_ONE_TYPE(OpTensorFusion, FP16);                                                            \
    DEF_FACTORY_ADVANCED_OP_ONE_TYPE(OpTensorFusion, FP32);                                                            \
    DEF_FACTORY_ADVANCED_OP_ONE_TYPE(OpTensorFusion, FP64);

#define DEF_FACTORY_ADVANCED_SPECTRAL_TRANSFORM()                                                                      \
    DEF_FACTORY_ADVANCED_OP_ONE_TYPE(OpSpectralTransform, FP32);                                                       \
    DEF_FACTORY_ADVANCED_OP_ONE_TYPE(OpSpectralTransform, FP64);

#define DEF_FACTORY_ADVANCED_ACTIVATION()                                                                              \
    DEF_FACTORY_ADVANCED_OP_ONE_TYPE(OpAdvancedActivation, FP16);                                                      \
    DEF_FACTORY_ADVANCED_OP_ONE_TYPE(OpAdvancedActivation, FP32);                                                      \
    DEF_FACTORY_ADVANCED_OP_ONE_TYPE(OpAdvancedActivation, FP64);

#define DEF_FACTORY_ADVANCED_DECOMPOSITION()                                                                           \
    DEF_FACTORY_ADVANCED_OP_ONE_TYPE(OpTensorDecomposition, FP32);                                                     \
    DEF_FACTORY_ADVANCED_OP_ONE_TYPE(OpTensorDecomposition, FP64);

#define DEF_FACTORY_ADVANCED_STATISTICAL()                                                                             \
    DEF_FACTORY_ADVANCED_OP_ONE_TYPE(OpStatisticalOps, FP32);                                                          \
    DEF_FACTORY_ADVANCED_OP_ONE_TYPE(OpStatisticalOps, FP64);

#define DEF_FACTORY_ADVANCED_GEOMETRIC()                                                                               \
    DEF_FACTORY_ADVANCED_OP_ONE_TYPE(OpGeometricOps, FP16);                                                            \
    DEF_FACTORY_ADVANCED_OP_ONE_TYPE(OpGeometricOps, FP32);                                                            \
    DEF_FACTORY_ADVANCED_OP_ONE_TYPE(OpGeometricOps, FP64);

namespace TosaReference
{

// Extension to OpFactory for advanced operations
class AdvancedOpFactory
{
public:
    static GraphNode* newAdvancedOp(SubgraphTraverser* sgt,
                                    tosa::TosaSerializationHandler* tsh,
                                    const std::string& op_name,
                                    tosa::TosaAttributeBase* attribute,
                                    uint64_t id,
                                    TOSA_REF_TYPE inputDTYPE,
                                    int inputRank,
                                    TOSA_REF_TYPE outputDTYPE,
                                    int outputRank);
    
    // Helper functions for creating advanced operations
    static GraphNode* createTensorFusion(SubgraphTraverser* sgt,
                                          tosa::TosaAttributeBase* attribute,
                                          uint64_t id,
                                          TOSA_REF_TYPE inputDTYPE,
                                          int inputRank);
    
    static GraphNode* createSpectralTransform(SubgraphTraverser* sgt,
                                               tosa::TosaAttributeBase* attribute,
                                               uint64_t id,
                                               TOSA_REF_TYPE inputDTYPE,
                                               int inputRank);
    
    static GraphNode* createAdvancedActivation(SubgraphTraverser* sgt,
                                                tosa::TosaAttributeBase* attribute,
                                                uint64_t id,
                                                TOSA_REF_TYPE inputDTYPE,
                                                int inputRank);
    
    static GraphNode* createTensorDecomposition(SubgraphTraverser* sgt,
                                                 tosa::TosaAttributeBase* attribute,
                                                 uint64_t id,
                                                 TOSA_REF_TYPE inputDTYPE,
                                                 int inputRank);
    
    static GraphNode* createStatisticalOps(SubgraphTraverser* sgt,
                                            tosa::TosaAttributeBase* attribute,
                                            uint64_t id,
                                            TOSA_REF_TYPE inputDTYPE,
                                            int inputRank);
    
    static GraphNode* createGeometricOps(SubgraphTraverser* sgt,
                                          tosa::TosaAttributeBase* attribute,
                                          uint64_t id,
                                          TOSA_REF_TYPE inputDTYPE,
                                          int inputRank);
};

}    // namespace TosaReference

#endif