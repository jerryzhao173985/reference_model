// Copyright (c) 2023-2025, ARM Limited.
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
#include "cfloat.h"
#include "tosa_generated.h"
#include "verify.h"

#include <algorithm>
#include <cmath>
#include <cstdint>

#include <array>
#include <iterator>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

// Include this last because it redefines REQUIRE
#include "test_utils.h"

using namespace ct;
using namespace tosa;

namespace
{

// TODO(ITL): DType2tosa_datatype_t could be useful elsewhere, but currently
// it is hard to find a good place for it. Move it somewhere reasonable later.
tosa_datatype_t DType2tosa_datatype_t(DType dtype)
{
    switch (dtype)
    {
        case DType_BOOL:
            return tosa_datatype_bool_t;
        case DType_INT4:
            return tosa_datatype_int4_t;
        case DType_INT8:
            return tosa_datatype_int8_t;
        case DType_INT16:
            return tosa_datatype_int16_t;
        case DType_INT32:
            return tosa_datatype_int32_t;
        case DType_INT48:
            return tosa_datatype_int48_t;
        case DType_FP32:
            return tosa_datatype_fp32_t;
        case DType_FP16:
            return tosa_datatype_fp16_t;
        case DType_BF16:
            return tosa_datatype_bf16_t;
        case DType_FP8E4M3:
            return tosa_datatype_fp8e4m3_t;
        case DType_FP8E5M2:
            return tosa_datatype_fp8e5m2_t;
        case DType_SHAPE:
            return tosa_datatype_shape_t;
        default:
            throw "Unknown Dtype";
    }
}

void update_json_template(std::string& str, const std::string& find, const std::string& change)
{
    // Update the 'str' by looking for instances of 'find' and replacing them with 'change'
    auto pos = str.find(find);
    while (pos != std::string::npos)
    {
        str.replace(pos, find.length(), change);
        pos = str.find(find);
    }
}

class TosaVerifTensor
{
public:
    TosaVerifTensor(std::string name, tosa_datatype_t dataType, std::vector<int32_t> shape, uint8_t* data = nullptr)
        : _name(std::move(name))
        , _shape(std::move(shape))
    {
        _tensor.name      = _name.c_str();
        _tensor.data_type = dataType;
        _tensor.num_dims  = static_cast<int32_t>(_shape.size());
        _tensor.shape     = _shape.data();
        _tensor.data      = data;
        _tensor.size      = static_cast<size_t>(
            std::accumulate(_tensor.shape, std::next(_tensor.shape, _tensor.num_dims), 1, std::multiplies<>()));
    };

    const tosa_tensor_t* cTensor() const
    {
        return &_tensor;
    }

private:
    std::string _name;
    std::vector<int32_t> _shape;
    tosa_tensor_t _tensor;
};

template <typename FP>
std::enable_if_t<std::is_floating_point_v<FP>, FP> increment(FP input, uint64_t steps)
{
    for (uint64_t step = 0; step < steps; ++step)
        input = std::nextafter(input, std::numeric_limits<FP>::infinity());
    return input;
}

template <typename FP>
std::enable_if_t<std::is_floating_point_v<FP>, FP> decrement(FP input, uint64_t steps)
{
    for (uint64_t step = 0; step < steps; ++step)
        input = std::nextafter(input, -std::numeric_limits<FP>::infinity());
    return input;
}

auto& getRandomGenerator()
{
    static std::mt19937 gen(0);
    return gen;
}

template <typename FP>
std::enable_if_t<std::is_floating_point_v<FP>, std::add_lvalue_reference_t<std::uniform_real_distribution<FP>>>
    getUniformRealDist()
{
    // Uniform real distribution generates real values in the range [a, b]
    // and requires that b - a <= std::numeric_limits<FP>::max() so here
    // we choose some arbitrary values that satisfy that condition.
    constexpr auto min = std::numeric_limits<FP>::lowest() / 2;
    constexpr auto max = std::numeric_limits<FP>::max() / 2;
    static_assert(max <= std::numeric_limits<FP>::max() + min);

    static std::uniform_real_distribution<FP> dis(min, max);
    return dis;
}

template <typename FP>
std::enable_if_t<std::is_floating_point_v<FP>, FP> getRandomUniformFloat()
{
    return getUniformRealDist<FP>()(getRandomGenerator());
}

template <typename FP>
std::enable_if_t<std::is_floating_point_v<FP>, std::vector<FP>> generateRandomTensorData(size_t elementCount,
                                                                                         bool includeNans = false)
{
    // Generate some random floats using the full range of fp32.
    auto data = std::vector<FP>(elementCount);
    std::generate(std::begin(data), std::end(data), []() { return getRandomUniformFloat<FP>(); });

    // Include some edge cases.
    auto edgeCases = std::vector<float>{ +0.0f, -0.0f, std::numeric_limits<float>::infinity(),
                                         -std::numeric_limits<float>::infinity() };
    if (includeNans)
    {
        static const auto nans =
            std::vector<float>{ std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::signaling_NaN() };

        std::copy(std::begin(nans), std::end(nans), std::back_inserter(edgeCases));
    }

    if (elementCount >= edgeCases.size())
    {
        // Evenly distribute the edge cases throughout the data, this way for operations like reductions all edge cases won't
        // end up in the same row/column over which a reduction happens.
        const auto stride = (data.size() + (edgeCases.size() - 1)) / edgeCases.size();
        for (unsigned i = 0; i < edgeCases.size(); ++i)
        {
            data[i * stride] = edgeCases[i];
        }
    }

    return data;
}

// Calculates the "error" in the tolerance calculation as: E = pow(1 + pow(2, -M-1), N) - 1.
// where M is the number of mantisa bits in the floating point representation and N is the number
// of elements in the product.
auto reduceProductError(uint64_t M, uint64_t N)
{
    return std::pow(1 + std::pow(2, -static_cast<int64_t>(M) - 1), N) - 1;
}

template <typename FP>
auto reduceProductTolerance(uint64_t M, uint64_t N, const std::vector<FP>& results)
{
    const auto error     = reduceProductError(M, N);
    auto tolerances_fp64 = std::vector<FP>(results.size());
    for (size_t i = 0, end = results.size(); i < end; ++i)
    {
        tolerances_fp64[i] = std::abs(results[i]) * error;
    }
    return tolerances_fp64;
}

template <typename FP>
void checkAbsErrorInjection(std::vector<FP>& data,
                            std::vector<double>& bounds,
                            const std::vector<int32_t>& shape,
                            const std::string& jsonCfg,
                            double errorFactor,
                            bool expectFailure)
{
    auto noisyData = data;
    std::vector<double> data_fp64(data.begin(), data.end());

    std::for_each(std::begin(noisyData), std::end(noisyData), [=](auto& value) {
        if (std::abs(value) != 0.0 && !std::isinf(value) && !std::isnan(value))
        {
            // If we used lower precision here, we would add more error.
            double value64 = static_cast<double>(value);
            value64 += value64 * errorFactor;
            value = static_cast<FP>(value64);
        }
    });
    const auto referenceTensor =
        TosaVerifTensor("out1", tosa_datatype_fp64_t, shape, reinterpret_cast<uint8_t*>(data_fp64.data()));
    const auto boundsTensor =
        TosaVerifTensor("out1", tosa_datatype_fp64_t, shape, reinterpret_cast<uint8_t*>(bounds.data()));
    const auto implementationTensor = TosaVerifTensor("out1", DType2tosa_datatype_t(NativeType2DType<FP>()), shape,
                                                      reinterpret_cast<uint8_t*>(noisyData.data()));
    if (expectFailure)
        REQUIRE_FALSE(tvf_verify_data(referenceTensor.cTensor(), boundsTensor.cTensor(), implementationTensor.cTensor(),
                                      jsonCfg.c_str()));
    else
        REQUIRE(tvf_verify_data(referenceTensor.cTensor(), boundsTensor.cTensor(), implementationTensor.cTensor(),
                                jsonCfg.c_str()));
}

}    // namespace

TEST_SUITE_BEGIN("verify");

TEST_CASE("negative - api")
{
    std::string jsonCfg = R"({
        "tensors" : {
            "out1" : {
                "mode": "DOT_PRODUCT",
                "data_type": "FP32",
                "dot_product_info" : {
                    "s": 2,
                    "ksb": 9
                }
            }
        }
    })";

    SUBCASE("invalid json")
    {
        std::string invalidJsonCfg = R"({
            "tensors" : {
                "out1" : {
                    "mode": DOT_PRODUCT,
                },
            }
        })";

        const TosaVerifTensor ref("out1", tosa_datatype_fp64_t, { 8, 8, 8 });
        const TosaVerifTensor refAbs("out1", tosa_datatype_fp64_t, { 8, 8, 8 });
        const TosaVerifTensor imp("out1", tosa_datatype_fp32_t, { 8, 8, 8 });

        REQUIRE_FALSE(tvf_verify_data(ref.cTensor(), refAbs.cTensor(), imp.cTensor(), invalidJsonCfg.c_str()));
    }
    SUBCASE("unknown mode")
    {
        std::string unknownJsonCfg = R"({
            "tensors" : {
                "out1" : {
                    "mode": "WIND",
                    "data_type": "FP32"
                }
            }
        })";

        const TosaVerifTensor ref("out1", tosa_datatype_fp64_t, { 8 });
        const TosaVerifTensor imp("out1", tosa_datatype_fp32_t, { 8 });

        REQUIRE_FALSE(tvf_verify_data(ref.cTensor(), nullptr, imp.cTensor(), unknownJsonCfg.c_str()));
    }
    SUBCASE("unknown type")
    {
        std::string unknownJsonCfg = R"({
            "tensors" : {
                "out1" : {
                    "mode": "DOT_PRODUCT",
                    "data_type": "JOULES"
                }
            }
        })";

        const TosaVerifTensor ref("out1", tosa_datatype_fp64_t, { 8 });
        const TosaVerifTensor imp("out1", tosa_datatype_fp32_t, { 8 });

        REQUIRE_FALSE(tvf_verify_data(ref.cTensor(), nullptr, imp.cTensor(), unknownJsonCfg.c_str()));
    }
    SUBCASE("mismatching dimensions")
    {
        const TosaVerifTensor ref("out1", tosa_datatype_fp64_t, { 4, 4 });
        const TosaVerifTensor refAbs("out1", tosa_datatype_fp64_t, { 4, 4 });
        const TosaVerifTensor imp("out1", tosa_datatype_fp32_t, { 8, 8, 8 });

        REQUIRE_FALSE(tvf_verify_data(ref.cTensor(), refAbs.cTensor(), imp.cTensor(), jsonCfg.c_str()));
    }
    SUBCASE("mismatching shapes")
    {
        const TosaVerifTensor ref("out1", tosa_datatype_fp64_t, { 8, 8, 8 });
        const TosaVerifTensor refAbs("out1", tosa_datatype_fp64_t, { 8, 8, 8 });
        const TosaVerifTensor imp("out1", tosa_datatype_fp32_t, { 4, 4, 4 });

        REQUIRE_FALSE(tvf_verify_data(ref.cTensor(), refAbs.cTensor(), imp.cTensor(), jsonCfg.c_str()));
    }
    SUBCASE("mismatching data types")
    {
        const TosaVerifTensor ref("out1", tosa_datatype_fp64_t, { 8, 8, 8 });
        const TosaVerifTensor refAbs("out1", tosa_datatype_fp64_t, { 8, 8, 8 });
        const TosaVerifTensor imp("out1", tosa_datatype_fp16_t, { 8, 8, 8 });

        REQUIRE_FALSE(tvf_verify_data(ref.cTensor(), refAbs.cTensor(), imp.cTensor(), jsonCfg.c_str()));
    }
    SUBCASE("missing tensor data")
    {
        const TosaVerifTensor ref("out1", tosa_datatype_fp64_t, { 8, 8, 8 });
        const TosaVerifTensor refAbs("out1", tosa_datatype_fp64_t, { 8, 8, 8 });
        const TosaVerifTensor imp("out1", tosa_datatype_fp32_t, { 8, 8, 8 });

        REQUIRE_FALSE(tvf_verify_data(ref.cTensor(), refAbs.cTensor(), imp.cTensor(), jsonCfg.c_str()));
    }
}

TEST_CASE("positive - exact")
{
    std::string jsonCfg = R"({
        "tensors" : {
            "out1" : {
                "mode": "EXACT",
                "data_type": "FP32"
            }
        }
    })";

    const auto shape = std::vector<int32_t>{ 8, 8, 8 };
    const size_t elementCount =
        static_cast<size_t>(std::accumulate(std::begin(shape), std::end(shape), 1, std::multiplies<>()));

    // Generate some random floats using the full range of fp32.
    auto data_fp32 = generateRandomTensorData<float>(elementCount);
    std::vector<double> data_fp64(data_fp32.begin(), data_fp32.end());
    SUBCASE("same")
    {
        const auto referenceTensor =
            TosaVerifTensor("out1", tosa_datatype_fp64_t, shape, reinterpret_cast<uint8_t*>(data_fp64.data()));
        const auto implementationTensor =
            TosaVerifTensor("out1", tosa_datatype_fp32_t, shape, reinterpret_cast<uint8_t*>(data_fp32.data()));
        REQUIRE(tvf_verify_data(referenceTensor.cTensor(), nullptr, implementationTensor.cTensor(), jsonCfg.c_str()));
    }

    SUBCASE("different")
    {
        // Generate some mismatched tensors by setting every other value to an incrementing counter.
        // In theory this could be the same, but the probability is tiny.
        auto otherData_fp32 = std::vector<float>(elementCount);
        std::generate(std::begin(otherData_fp32), std::end(otherData_fp32), [&, i = 0]() mutable {
            auto oldIndex = i++;
            return oldIndex % 2 ? data_fp32[static_cast<size_t>(oldIndex)] : static_cast<float>(oldIndex);
        });

        const auto referenceTensor =
            TosaVerifTensor("out1", tosa_datatype_fp64_t, shape, reinterpret_cast<uint8_t*>(data_fp64.data()));
        const auto implementationTensor =
            TosaVerifTensor("out1", tosa_datatype_fp32_t, shape, reinterpret_cast<uint8_t*>(otherData_fp32.data()));
        REQUIRE_FALSE(
            tvf_verify_data(referenceTensor.cTensor(), nullptr, implementationTensor.cTensor(), jsonCfg.c_str()));
    }

    SUBCASE("denorm to zero")
    {
        // Denormal / subnormal test
        auto denormData_fp64     = std::vector<double>(elementCount);
        auto denormZeroData_fp32 = std::vector<float>(elementCount);
        float min                = std::numeric_limits<float>::min();
        float denorm             = min;    // Start at minimum and get smaller
        for (size_t idx = 0; idx < elementCount; idx++)
        {
            if (idx % 2)
            {
                denorm                   = std::nextafter(denorm, std::numeric_limits<float>::denorm_min());
                denormData_fp64[idx]     = static_cast<double>(denorm);
                denormZeroData_fp32[idx] = 0.f;
            }
            else
            {
                denormData_fp64[idx]     = static_cast<double>(min);
                denormZeroData_fp32[idx] = min;
                min                      = std::nextafter(min, std::numeric_limits<float>::max());
            }
            if (data_fp32[idx] < 0.f)
            {
                denormData_fp64[idx]     = -denormData_fp64[idx];
                denormZeroData_fp32[idx] = -denormZeroData_fp32[idx];
            }
        }

        const auto referenceTensor =
            TosaVerifTensor("out1", tosa_datatype_fp64_t, shape, reinterpret_cast<uint8_t*>(denormData_fp64.data()));
        const auto implementationTensor = TosaVerifTensor("out1", tosa_datatype_fp32_t, shape,
                                                          reinterpret_cast<uint8_t*>(denormZeroData_fp32.data()));
        REQUIRE(tvf_verify_data(referenceTensor.cTensor(), nullptr, implementationTensor.cTensor(), jsonCfg.c_str()));
    }

    SUBCASE("norm to zero not allowed")
    {
        // Check minimum normal is not allowed flushed to zero
        auto normData_fp64     = std::vector<double>(elementCount);
        auto normZeroData_fp32 = std::vector<float>(elementCount);
        float min              = std::numeric_limits<float>::min();
        for (size_t idx = 0; idx < elementCount; idx++)
        {
            normData_fp64[idx] = static_cast<double>(min);
            if (idx % 2)
            {
                normZeroData_fp32[idx] = 0.f;
            }
            else
            {
                normZeroData_fp32[idx] = min;
            }
            if (data_fp32[idx] < 0.f)
            {
                normData_fp64[idx]     = -normData_fp64[idx];
                normZeroData_fp32[idx] = -normZeroData_fp32[idx];
            }
        }

        const auto referenceTensor =
            TosaVerifTensor("out1", tosa_datatype_fp64_t, shape, reinterpret_cast<uint8_t*>(normData_fp64.data()));
        const auto implementationTensor =
            TosaVerifTensor("out1", tosa_datatype_fp32_t, shape, reinterpret_cast<uint8_t*>(normZeroData_fp32.data()));
        REQUIRE_FALSE(
            tvf_verify_data(referenceTensor.cTensor(), nullptr, implementationTensor.cTensor(), jsonCfg.c_str()));
    }
}

TEST_CASE("positive - reduce product")
{
    std::string jsonCfg = R"({
        "tensors" : {
            "out1" : {
                "mode": "REDUCE_PRODUCT",
                "data_type": "FP32",
                "reduce_product_info": {
                    "n": 8
                }
            }
        }
    })";

    const auto inputShape      = std::vector<int32_t>{ 8, 8, 8 };
    const auto outputShape     = std::vector<int32_t>{ 8, 8, 1 };
    const size_t reductionSize = static_cast<size_t>(inputShape[2]);
    const size_t elementCount =
        static_cast<size_t>(std::accumulate(std::begin(inputShape), std::end(inputShape), 1, std::multiplies<>()));

    // Generate some random floats using the full range of fp32. This will be the "result" of our
    // dot product. Here we "reduced" over the z-axis of our shape.
    auto data_fp32 = generateRandomTensorData<float>(elementCount / reductionSize, false);
    std::vector<double> data_fp64(data_fp32.begin(), data_fp32.end());
    // Calculate the tolerances_fp64 for each element in the result.
    // A float has 23 bit dedicated to the fraction.
    constexpr uint64_t mantisa_count = 23;
    const auto tolerances_fp64       = reduceProductTolerance(mantisa_count, reductionSize, data_fp64);

    SUBCASE("same")
    {
        // Generate some new floats that are as far away as possible from each result without
        // exceeding the tolerance.
        auto otherData_fp32 = std::vector<float>(elementCount / reductionSize);
        for (unsigned i = 0; i < data_fp32.size(); ++i)
        {
            auto newValue       = data_fp32[i];
            const double target = tolerances_fp64[i] + newValue;

            // Here we just increment the value until we exceed the tolerance. For simplicity we go up.
            auto previousValue = newValue;
            while (newValue < target)
            {
                previousValue = newValue;
                newValue      = std::nextafter(newValue, std::numeric_limits<float>::infinity());
            }

            otherData_fp32[i] = previousValue;
        }

        const auto referenceTensor =
            TosaVerifTensor("out1", tosa_datatype_fp64_t, outputShape, reinterpret_cast<uint8_t*>(data_fp64.data()));
        const auto implementationTensor = TosaVerifTensor("out1", tosa_datatype_fp32_t, outputShape,
                                                          reinterpret_cast<uint8_t*>(otherData_fp32.data()));
        REQUIRE(tvf_verify_data(referenceTensor.cTensor(), nullptr, implementationTensor.cTensor(), jsonCfg.c_str()));
    }

    SUBCASE("different")
    {
        // Generate some new floats that exceed the tolerance.
        auto otherData_fp32 = std::vector<float>(elementCount / reductionSize);
        for (unsigned i = 0; i < data_fp32.size(); ++i)
        {
            auto newValue       = data_fp32[i];
            const double target = tolerances_fp64[i] + newValue;

            // Here we just increment the value until we exceed the tolerance. For simplicity we go up.
            while (newValue < target)
            {
                newValue = std::nextafter(newValue, std::numeric_limits<float>::infinity());
            }

            otherData_fp32[i] = newValue;
        }

        const auto referenceTensor =
            TosaVerifTensor("out1", tosa_datatype_fp64_t, outputShape, reinterpret_cast<uint8_t*>(data_fp64.data()));
        const auto implementationTensor = TosaVerifTensor("out1", tosa_datatype_fp32_t, outputShape,
                                                          reinterpret_cast<uint8_t*>(otherData_fp32.data()));
        REQUIRE_FALSE(
            tvf_verify_data(referenceTensor.cTensor(), nullptr, implementationTensor.cTensor(), jsonCfg.c_str()));
    }
}

TEST_CASE("positive - ulp")
{
    std::string jsonCfg = R"({
        "tensors" : {
            "out1" : {
                "mode": "ULP",
                "data_type": "FP32",
                "ulp_info": {
                "ulp": 5
                }
            }
        }
    })";

    const auto shape = std::vector<int32_t>{ 8, 8, 8 };
    const size_t elementCount =
        static_cast<size_t>(std::accumulate(std::begin(shape), std::end(shape), 1, std::multiplies<>()));

    // Generate some random floats using the full range of fp32.
    auto data_fp32 = generateRandomTensorData<float>(elementCount, true);
    std::vector<double> data_fp64(data_fp32.begin(), data_fp32.end());

    SUBCASE("same")
    {
        // Generate some data that meets the ULP requirements of the result.
        auto otherData_fp32 = data_fp32;
        std::for_each(std::begin(otherData_fp32), std::end(otherData_fp32), [](auto& value) {
            if (std::abs(value) != 0.0 && !std::isinf(value) && !std::isnan(value))
                value = increment(value, 5);
        });
        const auto referenceTensor =
            TosaVerifTensor("out1", tosa_datatype_fp64_t, shape, reinterpret_cast<uint8_t*>(data_fp64.data()));
        const auto implementationTensor =
            TosaVerifTensor("out1", tosa_datatype_fp32_t, shape, reinterpret_cast<uint8_t*>(otherData_fp32.data()));
        REQUIRE(tvf_verify_data(referenceTensor.cTensor(), nullptr, implementationTensor.cTensor(), jsonCfg.c_str()));
    }

    SUBCASE("different")
    {
        // Generate some data that exceeds a specified number of ULP for each value in the tensor.
        auto otherData_fp32 = data_fp32;
        std::for_each(std::begin(otherData_fp32), std::end(otherData_fp32), [](auto& value) {
            if (std::abs(value) != 0.0 && !std::isinf(value) && !std::isnan(value))
                value = increment(value, 6);
        });

        const auto referenceTensor =
            TosaVerifTensor("out1", tosa_datatype_fp64_t, shape, reinterpret_cast<uint8_t*>(data_fp64.data()));
        const auto implementationTensor =
            TosaVerifTensor("out1", tosa_datatype_fp32_t, shape, reinterpret_cast<uint8_t*>(otherData_fp32.data()));
        REQUIRE_FALSE(
            tvf_verify_data(referenceTensor.cTensor(), nullptr, implementationTensor.cTensor(), jsonCfg.c_str()));
    }

    SUBCASE("same - small inputs")
    {
        // We have introduced bugs in the past in the computation of ULP for values that are near
        // the normal minimum. Adding a specialized test to capture those errors in the future.

        // Generate random values in [normal_min, 10*normal_min]
        std::uniform_real_distribution<float> smallDis(1.0, 10.0);
        auto smallData_fp32 = std::vector<float>(elementCount);
        std::generate(std::begin(smallData_fp32), std::end(smallData_fp32),
                      [&]() { return std::numeric_limits<float>::min() * smallDis(getRandomGenerator()); });

        std::vector<double> smallData_fp64(smallData_fp32.begin(), smallData_fp32.end());

        std::for_each(std::begin(smallData_fp32), std::end(smallData_fp32), [](auto& value) {
            if (std::abs(value) != 0.0 && !std::isinf(value) && !std::isnan(value))
                value = increment(value, 5);
        });

        const auto referenceTensor =
            TosaVerifTensor("out1", tosa_datatype_fp64_t, shape, reinterpret_cast<uint8_t*>(smallData_fp64.data()));
        const auto implementationTensor =
            TosaVerifTensor("out1", tosa_datatype_fp32_t, shape, reinterpret_cast<uint8_t*>(smallData_fp32.data()));
        REQUIRE(tvf_verify_data(referenceTensor.cTensor(), nullptr, implementationTensor.cTensor(), jsonCfg.c_str()));
    }
    SUBCASE("different - small inputs")
    {
        // We have introduced bugs in the past in the computation of ULP for values that are near
        // the normal minimum. Adding a specialized test to capture those errors in the future.

        // Generate random values in [normal_min, 10*normal_min]
        std::uniform_real_distribution<float> smallDis(1.0, 10.0);
        auto smallData_fp32 = std::vector<float>(elementCount);
        std::generate(std::begin(smallData_fp32), std::end(smallData_fp32),
                      [&]() { return std::numeric_limits<float>::min() * smallDis(getRandomGenerator()); });

        std::vector<double> smallData_fp64(smallData_fp32.begin(), smallData_fp32.end());

        std::for_each(std::begin(smallData_fp32), std::end(smallData_fp32), [](auto& value) {
            if (std::abs(value) != 0.0 && !std::isinf(value) && !std::isnan(value))
                value = increment(value, 6);
        });

        const auto referenceTensor =
            TosaVerifTensor("out1", tosa_datatype_fp64_t, shape, reinterpret_cast<uint8_t*>(smallData_fp64.data()));
        const auto implementationTensor =
            TosaVerifTensor("out1", tosa_datatype_fp32_t, shape, reinterpret_cast<uint8_t*>(smallData_fp32.data()));
        REQUIRE_FALSE(
            tvf_verify_data(referenceTensor.cTensor(), nullptr, implementationTensor.cTensor(), jsonCfg.c_str()));
    }
}

TEST_CASE("positive - abs error")
{
    std::string jsonCfg = R"({
        "tensors" : {
            "out1" : {
                "mode": "ABS_ERROR",
                "data_type": "FP32"
            }
        }
    })";

    const auto shape = std::vector<int32_t>{ 4, 4, 4 };
    const size_t elementCount =
        static_cast<size_t>(std::accumulate(std::begin(shape), std::end(shape), 1, std::multiplies<>()));

    // Generate some random floats using the full range of fp32.
    auto data_fp32 = generateRandomTensorData<float>(elementCount, true);

    // Set up simple bounds of the input to 2.0
    std::vector<double> bounds_fp64(elementCount);
    std::for_each(std::begin(bounds_fp64), std::end(bounds_fp64), [](auto& value) { value = 2.0; });

    // Lower than exp2(-23) * (bounds[] - 0.5)
    // The 0.5 accounts for error when casting the result back to fp32
    // Error in the fp64 operations is too small to be relevant
    constexpr double insideErrBound = 1.7e-7;
    // Greater than exp2(-23) * (bounds[] + 0.5)
    constexpr double outsideErrBound = 3.0e-7;

    SUBCASE("inside")
    {
        // Generate some data that meets the ABS_ERROR requirements of the result.
        checkAbsErrorInjection<float>(data_fp32, bounds_fp64, shape, jsonCfg, /* errorFactor */ insideErrBound,
                                      /* expectFailure */ false);
    }

    SUBCASE("outside")
    {
        // Generate some data that exceeds a requirements for each value in the tensor.
        // Generate some data that meets the ABS_ERROR requirements of the result.
        checkAbsErrorInjection<float>(data_fp32, bounds_fp64, shape, jsonCfg, /* errorFactor */ outsideErrBound,
                                      /* expectFailure */ true);
    }
}

TEST_CASE("positive - abs error with base bound")
{
    std::string jsonCfg = R"({
        "tensors" : {
            "out1" : {
                "mode": "ABS_ERROR",
                "data_type": "FP32",
                "abs_error_info": {
                    "base_bound": 5
                }
            }
        }
    })";

    const auto shape = std::vector<int32_t>{ 4, 4, 4 };
    const auto elementCount =
        static_cast<size_t>(std::accumulate(std::begin(shape), std::end(shape), 1, std::multiplies<>()));

    // Generate some random floats using the full range of fp32.
    auto data_fp32 = generateRandomTensorData<float>(elementCount, true);
    std::vector<double> data_fp64(data_fp32.begin(), data_fp32.end());

    // Set up simple bounds of the input to 2.0
    std::vector<double> bounds_fp64(elementCount);
    std::for_each(std::begin(bounds_fp64), std::end(bounds_fp64), [](auto& value) { value = 2.0; });

    // Lower than exp2(-23) * (base_bound + bounds[] - 0.5)
    // The 0.5 accounts for error when casting the result back to fp32
    // Error in the fp64 operations is too small to be relevant
    constexpr double insideErrBound = 7.7e-7;
    // Greater than exp2(-23) * (base_bound + bounds[] + 0.5)
    constexpr double outsideErrBound = 9.0e-7;

    SUBCASE("inside")
    {
        // Generate some data that meets the ABS_ERROR requirements of the result.
        checkAbsErrorInjection<float>(data_fp32, bounds_fp64, shape, jsonCfg, /* errorFactor */ insideErrBound,
                                      /* expectFailure */ false);
    }

    SUBCASE("outside")
    {
        // Generate some data that exceeds a requirements for each value in the tensor.
        // Generate some data that meets the ABS_ERROR requirements of the result.
        checkAbsErrorInjection<float>(data_fp32, bounds_fp64, shape, jsonCfg, /* errorFactor */ outsideErrBound,
                                      /* expectFailure */ true);
    }
}

TEST_CASE("positive - relative")
{
    std::string templateJsonCfg = R"({
        "tensors" : {
            "out1" : {
                "mode": "RELATIVE",
                "data_type": "FP32",
                "relative_info": {
                    "max": _MAXIMUM_,
                    "scale": _SCALE_,
                    "ulp_bound": _ULP_BOUND_
                }
            }
        }
    })";

    const auto shape = std::vector<int32_t>{ 3, 3, 3 };
    const size_t elementCount =
        static_cast<size_t>(std::accumulate(std::begin(shape), std::end(shape), 1, std::multiplies<>()));

    // Generate some random floats using the full range of fp32.
    auto data_fp32 = generateRandomTensorData<float>(elementCount, true);
    std::vector<double> data_fp64(data_fp32.begin(), data_fp32.end());

    float scale     = 0.0006f;
    float max       = 0.0;
    float ulp_bound = 20.0;

    std::for_each(std::begin(data_fp32), std::end(data_fp32), [&max](auto& value) {
        if (!std::isinf(value) && !std::isnan(value))
        {
            max = std::max(max, std::abs(value));
        }
    });
    std::string jsonCfg = templateJsonCfg;
    update_json_template(jsonCfg, "_MAXIMUM_", std::to_string(max));
    update_json_template(jsonCfg, "_SCALE_", std::to_string(scale));
    update_json_template(jsonCfg, "_ULP_BOUND_", std::to_string(ulp_bound));

    float errBound = max * scale;
    // Use 10% error margin to test due to using v.large values in our random data
    float insideErrBound  = errBound * 0.9f;
    float outsideErrBound = errBound * 1.1f;

    SUBCASE("inside")
    {
        // Generate some data that meets the requirements of the result.
        auto otherData_fp32 = data_fp32;
        std::for_each(std::begin(otherData_fp32), std::end(otherData_fp32), [insideErrBound](auto& value) {
            if (std::abs(value) != 0.0 && !std::isinf(value) && !std::isnan(value))
                value += insideErrBound;
        });
        const auto referenceTensor =
            TosaVerifTensor("out1", tosa_datatype_fp64_t, shape, reinterpret_cast<uint8_t*>(data_fp64.data()));
        const auto implementationTensor =
            TosaVerifTensor("out1", tosa_datatype_fp32_t, shape, reinterpret_cast<uint8_t*>(otherData_fp32.data()));
        REQUIRE(tvf_verify_data(referenceTensor.cTensor(), nullptr, implementationTensor.cTensor(), jsonCfg.c_str()));
    }

    SUBCASE("outside")
    {
        // Generate some data that exceeds the requirements for each value in the tensor.
        auto otherData_fp32 = data_fp32;
        std::for_each(std::begin(otherData_fp32), std::end(otherData_fp32), [outsideErrBound](auto& value) {
            if (std::abs(value) != 0.0 && !std::isinf(value) && !std::isnan(value))
                value += outsideErrBound;
        });

        const auto referenceTensor =
            TosaVerifTensor("out1", tosa_datatype_fp64_t, shape, reinterpret_cast<uint8_t*>(data_fp64.data()));
        const auto implementationTensor =
            TosaVerifTensor("out1", tosa_datatype_fp32_t, shape, reinterpret_cast<uint8_t*>(otherData_fp32.data()));
        REQUIRE_FALSE(
            tvf_verify_data(referenceTensor.cTensor(), nullptr, implementationTensor.cTensor(), jsonCfg.c_str()));
    }
}

/*
 * SPECIAL CASES
 *
 * Testing of special cases like overflow and handling of subnormals
*/

template <typename FP_TYPE>
void checkULPVerification(double ref, FP_TYPE imp, bool should_pass, double ulp = 1.0, double ulp_lower = 0.0)
{
    DType dtype               = NativeType2DType<FP_TYPE>();
    tosa_datatype_t tosaDtype = DType2tosa_datatype_t(dtype);

    std::string jsonCfg = R"({
        "tensors" : {
            "out1" : {
                "mode": "ULP",
                "data_type": "_DATATYPE_",
                "ulp_info": {
                  "ulp": _ULP_
                  _ULPLOWER_ENTRY_ _ULPLOWER_VALUE_
                }
            }
        }
    })";

    update_json_template(jsonCfg, "_DATATYPE_", tosa::EnumNameDType(dtype));
    update_json_template(jsonCfg, "_ULP_", std::to_string(ulp));

    if (ulp_lower != 0.0)
    {
        // Add "ulp_lower": N to JSON with leading comma
        update_json_template(jsonCfg, "_ULPLOWER_ENTRY_", ", \"ulp_lower\": ");
        update_json_template(jsonCfg, "_ULPLOWER_VALUE_", std::to_string(ulp_lower));
    }
    else
    {
        update_json_template(jsonCfg, "_ULPLOWER_ENTRY_", "");
        update_json_template(jsonCfg, "_ULPLOWER_VALUE_", "");
    }

    const auto shape            = std::vector<int32_t>{ 1 };
    std::vector<double> refVec  = { ref };
    std::vector<FP_TYPE> impVec = { imp };

    const auto referenceTensor =
        TosaVerifTensor("out1", tosa_datatype_fp64_t, shape, reinterpret_cast<uint8_t*>(refVec.data()));
    const auto implementationTensor =
        TosaVerifTensor("out1", tosaDtype, shape, reinterpret_cast<uint8_t*>(impVec.data()));

    const bool verified =
        tvf_verify_data(referenceTensor.cTensor(), nullptr, implementationTensor.cTensor(), jsonCfg.c_str());
    INFO("ref: ", ref, " imp: ", double(imp), " should_pass: ", should_pass, " ulp : ", ulp);
    if (should_pass)
        REQUIRE(verified);
    else
        REQUIRE_FALSE(verified);
}

TEST_CASE_TEMPLATE("normal handling - ulp", FP_TYPE, float)
{
    SUBCASE("within upper bounds (same lower)")
    checkULPVerification<FP_TYPE>(134.0, increment<FP_TYPE>(134, 1), true, /* ulp */ 1.0);
    SUBCASE("within lower bounds (same lower)")
    checkULPVerification<FP_TYPE>(134.0, decrement<FP_TYPE>(134, 1), true, /* ulp */ 1.0);
    SUBCASE("outside upper bounds (same lower)")
    checkULPVerification<FP_TYPE>(134.0, increment<FP_TYPE>(134, 2), false, /* ulp */ 1.0);
    SUBCASE("outside lower bounds (same lower)")
    checkULPVerification<FP_TYPE>(134.0, decrement<FP_TYPE>(134, 2), false, /* ulp */ 1.0);

    // Test of the lower bounds setting
    SUBCASE("within upper bounds (different lower)")
    checkULPVerification<FP_TYPE>(134.0, increment<FP_TYPE>(134, 1), true, /* ulp */ 1.0, /* ulp_lower */ 2.0);
    SUBCASE("within lower bounds (different lower)")
    checkULPVerification<FP_TYPE>(134.0, decrement<FP_TYPE>(134, 2), true, /* ulp */ 1.0, /* ulp_lower */ 2.0);
    SUBCASE("outside upper bounds (different lower)")
    checkULPVerification<FP_TYPE>(134.0, increment<FP_TYPE>(134, 2), false, /* ulp */ 1.0, /* ulp_lower */ 2.0);
    SUBCASE("outside lower bounds (different lower)")
    checkULPVerification<FP_TYPE>(134.0, decrement<FP_TYPE>(134, 3), false, /* ulp */ 1.0, /* ulp_lower */ 2.0);
}

TEST_CASE_TEMPLATE("overflow handling - ulp", FP_TYPE, float, binary16, bfloat16, fp8_e4m3, fp8_e5m2)
{
    // Some definitions for readability.
    const bool has_inf           = std::numeric_limits<FP_TYPE>::has_infinity;
    const FP_TYPE dtype_max      = std::numeric_limits<FP_TYPE>::max();
    const FP_TYPE dtype_inf      = std::numeric_limits<FP_TYPE>::infinity();
    const FP_TYPE dtype_qnan     = std::numeric_limits<FP_TYPE>::quiet_NaN();
    const FP_TYPE dtype_overflow = has_inf ? dtype_inf : dtype_qnan;
    const double double_qnan     = std::numeric_limits<double>::quiet_NaN();
    const double double_inf      = std::numeric_limits<double>::infinity();

    SUBCASE("positive overflow of reference allows positive overflow of implementation")
    checkULPVerification<FP_TYPE>(double(dtype_max) * 2, dtype_overflow, true, /* ulp */ 0.5);

    SUBCASE("negative overflow of reference allows negative overflow of implementation")
    checkULPVerification<FP_TYPE>(-double(dtype_max) * 2, -dtype_overflow, true, /* ulp */ 0.5);

    SUBCASE("positive overflow within error bound of reference allows positive overflow of implementation")
    checkULPVerification<FP_TYPE>(double(dtype_max), dtype_overflow, true, /* ulp */ 2.0);

    SUBCASE("negative overflow within error bound of reference allows negative overflow of implementation")
    checkULPVerification<FP_TYPE>(-double(dtype_max), -dtype_overflow, true, /* ulp */ 2.0);

    SUBCASE("positive overflow within error bound of reference allows within bounds finite value in implementation")
    checkULPVerification<FP_TYPE>(double(dtype_max), dtype_max, true, /* ulp */ 2.0);

    SUBCASE("negative overflow within error bound of reference allows within bounds finite value in implementation")
    checkULPVerification<FP_TYPE>(-double(dtype_max), -dtype_max, true, /* ulp */ 2.0);

    SUBCASE("NaN allows NaN")
    checkULPVerification<FP_TYPE>(double_qnan, dtype_qnan, true);

    SUBCASE("+inf disallows 0")
    checkULPVerification<FP_TYPE>(double_inf, 0, false);

    SUBCASE("-inf disallows 0")
    checkULPVerification<FP_TYPE>(-double_inf, 0, false);

    SUBCASE("positive overflow disallows 0")
    checkULPVerification<FP_TYPE>(double(dtype_max) * 2, 0, false);

    SUBCASE("negative overflow disallows 0")
    checkULPVerification<FP_TYPE>(-double(dtype_max) * 2, 0, false);

    SUBCASE("NaN disallows 0")
    checkULPVerification<FP_TYPE>(double_qnan, 0, false);

    if constexpr (std::numeric_limits<FP_TYPE>::has_infinity)
    {
        SUBCASE("positive overflow disallows -inf")
        checkULPVerification<FP_TYPE>(double(dtype_max) * 2, -dtype_inf, false);

        SUBCASE("negative overflow disallows +inf")
        checkULPVerification<FP_TYPE>(-double(dtype_max) * 2, dtype_inf, false);

        SUBCASE("nan disallows +inf")
        checkULPVerification<FP_TYPE>(double_qnan, dtype_inf, false);

        SUBCASE("nan disallows -inf")
        checkULPVerification<FP_TYPE>(double_qnan, -dtype_inf, false);

        SUBCASE("+inf disallows nan")
        checkULPVerification<FP_TYPE>(double_inf, dtype_qnan, false);

        SUBCASE("-inf disallows nan")
        checkULPVerification<FP_TYPE>(-double_inf, dtype_qnan, false);

        SUBCASE("+inf allows +inf")
        checkULPVerification<FP_TYPE>(double_inf, dtype_inf, true);

        SUBCASE("-inf allows -inf")
        checkULPVerification<FP_TYPE>(-double_inf, -dtype_inf, true);
    }
};

TEST_CASE_TEMPLATE("subnormal handling non-fp8 - ulp", FP_TYPE, float, binary16, bfloat16)
{
    // Some definitions for readability.
    const FP_TYPE dtype_denorm_min = std::numeric_limits<FP_TYPE>::denorm_min();
    const FP_TYPE dtype_norm_min   = std::numeric_limits<FP_TYPE>::min();

    SUBCASE("positive subnormal reference allows flush to plus zero")
    checkULPVerification<FP_TYPE>(double(dtype_denorm_min), 0.0, true);
    checkULPVerification<FP_TYPE>(double(dtype_denorm_min), 0.0, true, /* ulp */ 0.5);
    checkULPVerification<FP_TYPE>(double(dtype_norm_min) * .5, 0.0, true);
    checkULPVerification<FP_TYPE>(double(dtype_norm_min) * .5, 0.0, true, /* ulp */ 0.5);
    checkULPVerification<FP_TYPE>(double(dtype_norm_min) * .1, 0.0, true);
    checkULPVerification<FP_TYPE>(double(dtype_norm_min) * .1, 0.0, true, /* ulp */ 0.5);

    SUBCASE("positive subnormal reference allows flush to minus zero")
    checkULPVerification<FP_TYPE>(double(dtype_denorm_min), -0.0, true);
    checkULPVerification<FP_TYPE>(double(dtype_denorm_min), -0.0, true, /* ulp */ 0.5);
    checkULPVerification<FP_TYPE>(double(dtype_norm_min) * .5, -0.0, true);
    checkULPVerification<FP_TYPE>(double(dtype_norm_min) * .5, -0.0, true, /* ulp */ 0.5);
    checkULPVerification<FP_TYPE>(double(dtype_norm_min) * .1, -0.0, true);
    checkULPVerification<FP_TYPE>(double(dtype_norm_min) * .1, -0.0, true, /* ulp */ 0.5);

    SUBCASE("negative subnormal reference allows flush to plus zero")
    checkULPVerification<FP_TYPE>(-double(dtype_denorm_min), 0.0, true);
    checkULPVerification<FP_TYPE>(-double(dtype_denorm_min), 0.0, true, /* ulp */ 0.5);
    checkULPVerification<FP_TYPE>(-double(dtype_norm_min) * .5, 0.0, true);
    checkULPVerification<FP_TYPE>(-double(dtype_norm_min) * .5, 0.0, true, /* ulp */ 0.5);
    checkULPVerification<FP_TYPE>(-double(dtype_norm_min) * .1, 0.0, true);
    checkULPVerification<FP_TYPE>(-double(dtype_norm_min) * .1, 0.0, true, /* ulp */ 0.5);

    SUBCASE("negative subnormal reference allows flush to minus zero")
    checkULPVerification<FP_TYPE>(-double(dtype_denorm_min), -0.0, true);
    checkULPVerification<FP_TYPE>(-double(dtype_denorm_min), -0.0, true, /* ulp */ 0.5);
    checkULPVerification<FP_TYPE>(-double(dtype_norm_min) * .5, -0.0, true);
    checkULPVerification<FP_TYPE>(-double(dtype_norm_min) * .5, -0.0, true, /* ulp */ 0.5);
    checkULPVerification<FP_TYPE>(-double(dtype_norm_min) * .1, -0.0, true);
    checkULPVerification<FP_TYPE>(-double(dtype_norm_min) * .1, -0.0, true, /* ulp */ 0.5);

    // a ULP of denorm_min is denorm_min
    SUBCASE("positive denorm_min allows negative denorm_min with ulp > 2")
    checkULPVerification<FP_TYPE>(double(dtype_denorm_min), -dtype_denorm_min, true, /* ulp */ 2.5);

    // a ULP of denorm_min is denorm_min
    SUBCASE("negative denorm_min allows positive denorm_min with ulp > 2")
    checkULPVerification<FP_TYPE>(-double(dtype_denorm_min), dtype_denorm_min, true, /* ulp */ 2.5);

    SUBCASE("positive subnormal reference disallows negative subnormal")
    checkULPVerification<FP_TYPE>(double(dtype_denorm_min), -dtype_denorm_min, false);
    checkULPVerification<FP_TYPE>(double(dtype_norm_min) * .5, -dtype_norm_min * ct::compat::cast<FP_TYPE>(.5), false);
    checkULPVerification<FP_TYPE>(double(dtype_norm_min) * .5, -dtype_norm_min * ct::compat::cast<FP_TYPE>(.5), false,
                                  /* ulp */ 3);
    checkULPVerification<FP_TYPE>(double(dtype_norm_min) * .1, -dtype_norm_min * ct::compat::cast<FP_TYPE>(.1), false);
    checkULPVerification<FP_TYPE>(double(dtype_norm_min) * .1, -dtype_norm_min * ct::compat::cast<FP_TYPE>(.1), false,
                                  /* ulp */ 3);

    SUBCASE("negative subnormal reference disallows positive subnormal")
    checkULPVerification<FP_TYPE>(-double(dtype_denorm_min), dtype_denorm_min, false);
    checkULPVerification<FP_TYPE>(-double(dtype_norm_min) * .5, dtype_norm_min * ct::compat::cast<FP_TYPE>(.5), false);
    checkULPVerification<FP_TYPE>(-double(dtype_norm_min) * .5, dtype_norm_min * ct::compat::cast<FP_TYPE>(.5), false,
                                  /* ulp */ 3);
    checkULPVerification<FP_TYPE>(-double(dtype_norm_min) * .1, dtype_norm_min * ct::compat::cast<FP_TYPE>(.1), false);
    checkULPVerification<FP_TYPE>(-double(dtype_norm_min) * .1, dtype_norm_min * ct::compat::cast<FP_TYPE>(.1), false,
                                  /* ulp */ 3);

    // Specifically covering a previous bug which allowed all subnormal values
    // when the reference was a subnormal input
    SUBCASE("positive subnormal reference disallows high-error subnormals")
    checkULPVerification<FP_TYPE>(double(dtype_denorm_min), dtype_denorm_min * 8, false, /* ulp */ 4);
    checkULPVerification<FP_TYPE>(double(dtype_denorm_min), -dtype_denorm_min * 8, false, /* ulp */ 4);
    checkULPVerification<FP_TYPE>(double(dtype_denorm_min) * 4, dtype_denorm_min * 8, false, /* ulp */ 2);
    checkULPVerification<FP_TYPE>(double(dtype_denorm_min) * 4, -dtype_denorm_min * 8, false, /* ulp */ 2);

    SUBCASE("negative subnormal reference disallows high-error subnormals")
    checkULPVerification<FP_TYPE>(-double(dtype_denorm_min), dtype_denorm_min * 8, false, /* ulp */ 4);
    checkULPVerification<FP_TYPE>(-double(dtype_denorm_min), -dtype_denorm_min * 8, false, /* ulp */ 4);
    checkULPVerification<FP_TYPE>(-double(dtype_denorm_min) * 4, dtype_denorm_min * 8, false, /* ulp */ 2);
    checkULPVerification<FP_TYPE>(-double(dtype_denorm_min) * 4, -dtype_denorm_min * 8, false, /* ulp */ 2);
};

TEST_CASE_TEMPLATE("subnormal handling fp8 - ulp", FP_TYPE, fp8_e5m2, fp8_e4m3)
{
    // Some definitions for readability.
    const FP_TYPE dtype_denorm_min = std::numeric_limits<FP_TYPE>::denorm_min();
    const FP_TYPE dtype_norm_min   = std::numeric_limits<FP_TYPE>::min();

    SUBCASE("positive subnormal reference allows flush to zero if within bounds")
    checkULPVerification<FP_TYPE>(double(dtype_denorm_min), 0.0, true);
    checkULPVerification<FP_TYPE>(double(dtype_denorm_min), -0.0, true);
    checkULPVerification<FP_TYPE>(double(dtype_denorm_min), 0.0, true, /* ulp */ 2);
    checkULPVerification<FP_TYPE>(double(dtype_denorm_min), -0.0, true, /* ulp */ 2);

    SUBCASE("negative subnormal reference allows flush to zero if within bounds")
    checkULPVerification<FP_TYPE>(-double(dtype_denorm_min), 0.0, true);
    checkULPVerification<FP_TYPE>(-double(dtype_denorm_min), -0.0, true);
    checkULPVerification<FP_TYPE>(-double(dtype_denorm_min), 0.0, true, /* ulp */ 2);
    checkULPVerification<FP_TYPE>(-double(dtype_denorm_min), -0.0, true, /* ulp */ 2);

    SUBCASE("positive subnormal reference disallows flush to zero if not within bounds")
    checkULPVerification<FP_TYPE>(double(dtype_denorm_min), 0.0, false, /* ulp */ 0.5);
    checkULPVerification<FP_TYPE>(double(dtype_denorm_min), -0.0, false, /* ulp */ 0.5);
    checkULPVerification<FP_TYPE>(double(dtype_norm_min) * .5, 0.0, false);
    checkULPVerification<FP_TYPE>(double(dtype_norm_min) * .5, -0.0, false);
    checkULPVerification<FP_TYPE>(double(dtype_norm_min) * .5, 0.0, false, /* ulp */ 0.5);
    checkULPVerification<FP_TYPE>(double(dtype_norm_min) * .5, -0.0, false, /* ulp */ 0.5);

    SUBCASE("negative subnormal reference disallows flush to zero if not within bounds")
    checkULPVerification<FP_TYPE>(-double(dtype_denorm_min), 0.0, false, /* ulp */ 0.5);
    checkULPVerification<FP_TYPE>(-double(dtype_denorm_min), -0.0, false, /* ulp */ 0.5);
    checkULPVerification<FP_TYPE>(-double(dtype_norm_min) * .5, 0.0, false);
    checkULPVerification<FP_TYPE>(-double(dtype_norm_min) * .5, -0.0, false);
    checkULPVerification<FP_TYPE>(-double(dtype_norm_min) * .5, 0.0, false, /* ulp */ 0.5);
    checkULPVerification<FP_TYPE>(-double(dtype_norm_min) * .5, -0.0, false, /* ulp */ 0.5);

    // Specifically covering a previous bug which allowed all subnormal values when the
    // reference was a subnormal input
    SUBCASE("positive subnormal reference disallows high-error subnormals")
    checkULPVerification<FP_TYPE>(double(dtype_denorm_min), dtype_denorm_min * 8, false, /* ulp */ 4);
    checkULPVerification<FP_TYPE>(double(dtype_denorm_min), -dtype_denorm_min * 8, false, /* ulp */ 4);
    checkULPVerification<FP_TYPE>(double(dtype_denorm_min) * 4, dtype_denorm_min * 8, false, /* ulp */ 2);
    checkULPVerification<FP_TYPE>(double(dtype_denorm_min) * 4, -dtype_denorm_min * 8, false, /* ulp */ 2);

    SUBCASE("negative subnormal reference disallows high-error subnormals")
    checkULPVerification<FP_TYPE>(-double(dtype_denorm_min), dtype_denorm_min * 8, false, /* ulp */ 4);
    checkULPVerification<FP_TYPE>(-double(dtype_denorm_min), -dtype_denorm_min * 8, false, /* ulp */ 4);
    checkULPVerification<FP_TYPE>(-double(dtype_denorm_min) * 4, dtype_denorm_min * 8, false, /* ulp */ 2);
    checkULPVerification<FP_TYPE>(-double(dtype_denorm_min) * 4, -dtype_denorm_min * 8, false, /* ulp */ 2);
}

TEST_SUITE_END();    // verify
