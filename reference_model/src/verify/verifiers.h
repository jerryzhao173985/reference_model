
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

#ifndef VERIFIERS_H_
#define VERIFIERS_H_

#include "verify_utils.h"

namespace TosaReference
{
/// \brief Perform dot-product based verification
///
/// \param ref    Reference tensor
/// \param refBnd Reference tensor when ran on abs(input)
/// \param imp    Implementation resulting tensor
/// \param dpInfo Dot-product verification meta-data
///
/// \return True if compliant else false
bool verifyDotProduct(const CTensor* ref,
                      const CTensor* refBnd,
                      const CTensor* imp,
                      const DotProductVerifyInfo& dpInfo);

/// \brief Perform exact result verification
///
/// \param referenceTensor    Reference tensor
/// \param implementationTensor    Implementation resulting tensor
///
/// \return True if compliant else false
bool verifyExact(const CTensor* referenceTensor, const CTensor* implementationTensor);

/// \brief Perform reduce product result verification
///
/// \param referenceTensor    Reference tensor
/// \param implementationTensor    Implementation resulting tensor
/// \param m    Number of manisa bits in the floating point representation
/// \param n    Number of elements in the product
///
/// \return True if compliant else false
bool verifyReduceProduct(const CTensor* referenceTensor, const CTensor* implementationTensor, uint64_t m, uint64_t n);

/// \brief Perform ULP result verification
///
/// \param referenceTensor    Reference tensor
/// \param implementationTensor    Implementation resulting tensor
/// \param ulpInfo    The ULP tolerence info for the comparison of the two tensors
///
/// \return True if compliant else false
bool verifyULP(const CTensor* referenceTensor, const CTensor* implementationTensor, const UlpVerifyInfo& ulpInfo);

/// \brief Perform abs-error based verification
///
/// \param ref    Reference tensor
/// \param refBnd Reference bounds tensor (according to op)
/// \param imp    Implementation resulting tensor
/// \param aeInfo Abs-error verification meta-data
///
/// \return True if compliant else false
bool verifyAbsError(const CTensor* ref, const CTensor* refBnd, const CTensor* imp, const AbsErrorVerifyInfo& aeInfo);

};    // namespace TosaReference

#endif    // VERIFIERS_H_
