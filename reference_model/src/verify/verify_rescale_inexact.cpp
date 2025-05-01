// Copyright (c) 2025, ARM Limited.
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

#include <cmath>
#include <vector>

#include "verifiers.h"
#include "verify_utils.h"

namespace TosaReference
{

namespace
{
template <typename DataType>
bool validateRescale(const uint8_t* refData,
                     const uint8_t* refBndData,
                     const uint8_t* impData,
                     const std::vector<int32_t>& shape)
{
    const auto elementCount = numElements(shape);

    const auto* referenceData = reinterpret_cast<const DataType*>(refData);
    TOSA_REF_REQUIRE(referenceData != nullptr, "[RI] Missing data for reference");
    const auto* boundsData = reinterpret_cast<const DataType*>(refBndData);
    TOSA_REF_REQUIRE(boundsData != nullptr, "[RI] Missing data for reference bounds");
    const auto* implementationData = reinterpret_cast<const DataType*>(impData);
    TOSA_REF_REQUIRE(implementationData != nullptr, "[RI] Missing data for implementation");

    for (int64_t i = 0; i < elementCount; i++)
    {
        // Expects the precise reference value to be the minimum allowed value
        // and the bounds reference value to be the maximum allowed value
        DataType minVal = referenceData[i];
        DataType maxVal = boundsData[i];
        DataType outImp = implementationData[i];

        TOSA_REF_REQUIRE(minVal <= maxVal, "[RI] Incorrect bounds values supplied")

        if (outImp < minVal || outImp > maxVal)
        {
            // mismatch found
            auto pos = TosaReference::indexToPosition(i, shape);
            WARNING("[Verifier][RI] Location %s, value %d not within range %d to %d",
                    TosaReference::positionToString(pos).c_str(), outImp, minVal, maxVal);
            return false;
        }
    }
    return true;
}
}    // namespace

bool verifyRescaleInexact(const CTensor* referenceTensor,
                          const CTensor* boundsTensor,
                          const CTensor* implementationTensor,
                          const RescaleInexactVerifyInfo& riInfo)
{
    // Validate that tensors are provided
    TOSA_REF_REQUIRE(referenceTensor != nullptr, "[RI] Reference tensor is missing");
    TOSA_REF_REQUIRE(boundsTensor != nullptr, "[RI] Reference bounds tensor is missing");
    TOSA_REF_REQUIRE(implementationTensor != nullptr, "[RI] Implementation tensor is missing");

    const std::vector<int32_t> refShape(referenceTensor->shape, referenceTensor->shape + referenceTensor->num_dims);

    const bool unsignedData = riInfo.unsignedData;

    switch (implementationTensor->data_type)
    {
        case tosa_datatype_int32_t: {
            if (unsignedData)
            {
                WARNING("[Verifier][RI] Data-type not supported.");
            }
            else
            {
                return validateRescale<int32_t>(referenceTensor->data, boundsTensor->data, implementationTensor->data,
                                                refShape);
            }
        }

        case tosa_datatype_int16_t: {
            if (unsignedData)
            {
                return validateRescale<uint16_t>(referenceTensor->data, boundsTensor->data, implementationTensor->data,
                                                 refShape);
            }
            else
            {
                return validateRescale<int16_t>(referenceTensor->data, boundsTensor->data, implementationTensor->data,
                                                refShape);
            }
        }
        case tosa_datatype_int8_t: {
            if (unsignedData)
            {
                return validateRescale<uint8_t>(referenceTensor->data, boundsTensor->data, implementationTensor->data,
                                                refShape);
            }
            else
            {
                return validateRescale<int8_t>(referenceTensor->data, boundsTensor->data, implementationTensor->data,
                                               refShape);
            }
        }
        default:
            WARNING("[Verifier][RI] Data-type not supported.");
            break;
    }

    return false;
}
}    // namespace TosaReference
