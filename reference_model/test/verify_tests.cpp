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
#include "verify.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <doctest.h>

#include <array>
#include <iterator>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

namespace
{
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

class TosaTensor
{
public:
    TosaTensor(std::string name, tosa_datatype_t dataType, std::vector<int32_t> shape, uint8_t* data = nullptr)
        : _name(std::move(name))
        , _shape(std::move(shape))
    {
        _tensor.name      = _name.c_str();
        _tensor.data_type = dataType;
        _tensor.num_dims  = _shape.size();
        _tensor.shape     = _shape.data();
        _tensor.data      = data;
        _tensor.size =
            std::accumulate(_tensor.shape, std::next(_tensor.shape, _tensor.num_dims), 1, std::multiplies<>());
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
constexpr auto reduceProductError(uint64_t M, uint64_t N)
{
    return std::pow(1 + std::pow(2, -static_cast<int64_t>(M) - 1), N) - 1;
}

template <typename FP>
auto reduceProductTolerance(uint64_t M, uint64_t N, const std::vector<FP>& results)
{
    const auto error     = reduceProductError(M, N);
    auto tolerances_fp64 = std::vector<FP>(results.size());
    for (unsigned i = 0, end = results.size(); i < end; ++i)
    {
        tolerances_fp64[i] = std::abs(results[i]) * error;
    }
    return tolerances_fp64;
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
                    "ks": 9
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

        const TosaTensor ref("out1", tosa_datatype_fp64_t, { 8, 8, 8 });
        const TosaTensor refAbs("out1", tosa_datatype_fp64_t, { 8, 8, 8 });
        const TosaTensor imp("out1", tosa_datatype_fp32_t, { 8, 8, 8 });

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

        const TosaTensor ref("out1", tosa_datatype_fp64_t, { 8 });
        const TosaTensor imp("out1", tosa_datatype_fp32_t, { 8 });

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

        const TosaTensor ref("out1", tosa_datatype_fp64_t, { 8 });
        const TosaTensor imp("out1", tosa_datatype_fp32_t, { 8 });

        REQUIRE_FALSE(tvf_verify_data(ref.cTensor(), nullptr, imp.cTensor(), unknownJsonCfg.c_str()));
    }
    SUBCASE("mismatching dimensions")
    {
        const TosaTensor ref("out1", tosa_datatype_fp64_t, { 4, 4 });
        const TosaTensor refAbs("out1", tosa_datatype_fp64_t, { 4, 4 });
        const TosaTensor imp("out1", tosa_datatype_fp32_t, { 8, 8, 8 });

        REQUIRE_FALSE(tvf_verify_data(ref.cTensor(), refAbs.cTensor(), imp.cTensor(), jsonCfg.c_str()));
    }
    SUBCASE("mismatching shapes")
    {
        const TosaTensor ref("out1", tosa_datatype_fp64_t, { 8, 8, 8 });
        const TosaTensor refAbs("out1", tosa_datatype_fp64_t, { 8, 8, 8 });
        const TosaTensor imp("out1", tosa_datatype_fp32_t, { 4, 4, 4 });

        REQUIRE_FALSE(tvf_verify_data(ref.cTensor(), refAbs.cTensor(), imp.cTensor(), jsonCfg.c_str()));
    }
    SUBCASE("mismatching data types")
    {
        const TosaTensor ref("out1", tosa_datatype_fp64_t, { 8, 8, 8 });
        const TosaTensor refAbs("out1", tosa_datatype_fp64_t, { 8, 8, 8 });
        const TosaTensor imp("out1", tosa_datatype_fp16_t, { 8, 8, 8 });

        REQUIRE_FALSE(tvf_verify_data(ref.cTensor(), refAbs.cTensor(), imp.cTensor(), jsonCfg.c_str()));
    }
    SUBCASE("missing tensor data")
    {
        const TosaTensor ref("out1", tosa_datatype_fp64_t, { 8, 8, 8 });
        const TosaTensor refAbs("out1", tosa_datatype_fp64_t, { 8, 8, 8 });
        const TosaTensor imp("out1", tosa_datatype_fp32_t, { 8, 8, 8 });

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

    const auto shape        = std::vector<int32_t>{ 8, 8, 8 };
    const auto elementCount = std::accumulate(std::begin(shape), std::end(shape), 1, std::multiplies<>());

    // Generate some random floats using the full range of fp32.
    auto data_fp32 = generateRandomTensorData<float>(elementCount);
    std::vector<double> data_fp64(data_fp32.begin(), data_fp32.end());
    SUBCASE("same")
    {
        const auto referenceTensor =
            TosaTensor("out1", tosa_datatype_fp64_t, shape, reinterpret_cast<uint8_t*>(data_fp64.data()));
        const auto implementationTensor =
            TosaTensor("out1", tosa_datatype_fp32_t, shape, reinterpret_cast<uint8_t*>(data_fp32.data()));
        REQUIRE(tvf_verify_data(referenceTensor.cTensor(), nullptr, implementationTensor.cTensor(), jsonCfg.c_str()));
    }

    SUBCASE("different")
    {
        // Generate some mismatched tensors by setting every other value to an incrementing counter.
        // In theory this could be the same, but the probability is tiny.
        auto otherData_fp32 = std::vector<float>(elementCount);
        std::generate(std::begin(otherData_fp32), std::end(otherData_fp32), [&, i = 0]() mutable {
            auto oldIndex = i++;
            return oldIndex % 2 ? data_fp32[oldIndex] : static_cast<float>(oldIndex);
        });

        const auto referenceTensor =
            TosaTensor("out1", tosa_datatype_fp64_t, shape, reinterpret_cast<uint8_t*>(data_fp64.data()));
        const auto implementationTensor =
            TosaTensor("out1", tosa_datatype_fp32_t, shape, reinterpret_cast<uint8_t*>(otherData_fp32.data()));
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
        for (auto idx = 0; idx < elementCount; idx++)
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
            TosaTensor("out1", tosa_datatype_fp64_t, shape, reinterpret_cast<uint8_t*>(denormData_fp64.data()));
        const auto implementationTensor =
            TosaTensor("out1", tosa_datatype_fp32_t, shape, reinterpret_cast<uint8_t*>(denormZeroData_fp32.data()));
        REQUIRE(tvf_verify_data(referenceTensor.cTensor(), nullptr, implementationTensor.cTensor(), jsonCfg.c_str()));
    }

    SUBCASE("norm to zero not allowed")
    {
        // Check minimum normal is not allowed flushed to zero
        auto normData_fp64     = std::vector<double>(elementCount);
        auto normZeroData_fp32 = std::vector<float>(elementCount);
        float min              = std::numeric_limits<float>::min();
        for (auto idx = 0; idx < elementCount; idx++)
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
            TosaTensor("out1", tosa_datatype_fp64_t, shape, reinterpret_cast<uint8_t*>(normData_fp64.data()));
        const auto implementationTensor =
            TosaTensor("out1", tosa_datatype_fp32_t, shape, reinterpret_cast<uint8_t*>(normZeroData_fp32.data()));
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

    const auto inputShape    = std::vector<int32_t>{ 8, 8, 8 };
    const auto outputShape   = std::vector<int32_t>{ 8, 8, 1 };
    const auto reductionSize = inputShape[2];
    const auto elementCount  = std::accumulate(std::begin(inputShape), std::end(inputShape), 1, std::multiplies<>());

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
            TosaTensor("out1", tosa_datatype_fp64_t, outputShape, reinterpret_cast<uint8_t*>(data_fp64.data()));
        const auto implementationTensor =
            TosaTensor("out1", tosa_datatype_fp32_t, outputShape, reinterpret_cast<uint8_t*>(otherData_fp32.data()));
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
            TosaTensor("out1", tosa_datatype_fp64_t, outputShape, reinterpret_cast<uint8_t*>(data_fp64.data()));
        const auto implementationTensor =
            TosaTensor("out1", tosa_datatype_fp32_t, outputShape, reinterpret_cast<uint8_t*>(otherData_fp32.data()));
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

    const auto shape        = std::vector<int32_t>{ 8, 8, 8 };
    const auto elementCount = std::accumulate(std::begin(shape), std::end(shape), 1, std::multiplies<>());

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
            TosaTensor("out1", tosa_datatype_fp64_t, shape, reinterpret_cast<uint8_t*>(data_fp64.data()));
        const auto implementationTensor =
            TosaTensor("out1", tosa_datatype_fp32_t, shape, reinterpret_cast<uint8_t*>(otherData_fp32.data()));
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
            TosaTensor("out1", tosa_datatype_fp64_t, shape, reinterpret_cast<uint8_t*>(data_fp64.data()));
        const auto implementationTensor =
            TosaTensor("out1", tosa_datatype_fp32_t, shape, reinterpret_cast<uint8_t*>(otherData_fp32.data()));
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

    const auto shape        = std::vector<int32_t>{ 4, 4, 4 };
    const auto elementCount = std::accumulate(std::begin(shape), std::end(shape), 1, std::multiplies<>());

    // Generate some random floats using the full range of fp32.
    auto data_fp32 = generateRandomTensorData<float>(elementCount, true);
    std::vector<double> data_fp64(data_fp32.begin(), data_fp32.end());

    // Set up simple bounds of the input to 2.0
    std::vector<double> bounds_fp64(elementCount);
    std::for_each(std::begin(bounds_fp64), std::end(bounds_fp64), [](auto& value) { value = 2.0; });
    constexpr float insideErrBound  = 1.0e-7 * 2;    // v.approx exp2(-23) * bounds[]
    constexpr float outsideErrBound = 1.0e-7 * 3;

    SUBCASE("inside")
    {
        // Generate some data that meets the ABS_ERROR requirements of the result.
        auto otherData_fp32 = data_fp32;
        std::for_each(std::begin(otherData_fp32), std::end(otherData_fp32), [insideErrBound](auto& value) {
            if (std::abs(value) != 0.0 && !std::isinf(value) && !std::isnan(value))
                value += value * insideErrBound;
        });
        const auto referenceTensor =
            TosaTensor("out1", tosa_datatype_fp64_t, shape, reinterpret_cast<uint8_t*>(data_fp64.data()));
        const auto boundsTensor =
            TosaTensor("out1", tosa_datatype_fp64_t, shape, reinterpret_cast<uint8_t*>(bounds_fp64.data()));
        const auto implementationTensor =
            TosaTensor("out1", tosa_datatype_fp32_t, shape, reinterpret_cast<uint8_t*>(otherData_fp32.data()));
        REQUIRE(tvf_verify_data(referenceTensor.cTensor(), boundsTensor.cTensor(), implementationTensor.cTensor(),
                                jsonCfg.c_str()));
    }

    SUBCASE("outside")
    {
        // Generate some data that exceeds a requirements for each value in the tensor.
        auto otherData_fp32 = data_fp32;
        std::for_each(std::begin(otherData_fp32), std::end(otherData_fp32), [outsideErrBound](auto& value) {
            if (std::abs(value) != 0.0 && !std::isinf(value) && !std::isnan(value))
                value += value * outsideErrBound;
        });

        const auto referenceTensor =
            TosaTensor("out1", tosa_datatype_fp64_t, shape, reinterpret_cast<uint8_t*>(data_fp64.data()));
        const auto boundsTensor =
            TosaTensor("out1", tosa_datatype_fp64_t, shape, reinterpret_cast<uint8_t*>(bounds_fp64.data()));
        const auto implementationTensor =
            TosaTensor("out1", tosa_datatype_fp32_t, shape, reinterpret_cast<uint8_t*>(otherData_fp32.data()));
        REQUIRE_FALSE(tvf_verify_data(referenceTensor.cTensor(), boundsTensor.cTensor(), implementationTensor.cTensor(),
                                      jsonCfg.c_str()));
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
                    "scale": _SCALE_
                }
            }
        }
    })";

    const auto shape        = std::vector<int32_t>{ 3, 3, 3 };
    const auto elementCount = std::accumulate(std::begin(shape), std::end(shape), 1, std::multiplies<>());

    // Generate some random floats using the full range of fp32.
    auto data_fp32 = generateRandomTensorData<float>(elementCount, true);
    std::vector<double> data_fp64(data_fp32.begin(), data_fp32.end());

    float scale = 0.0006;
    float max   = 0.0;
    std::for_each(std::begin(data_fp32), std::end(data_fp32), [&max](auto& value) {
        if (!std::isinf(value) && !std::isnan(value))
        {
            max = std::max(max, std::abs(value));
        }
    });
    std::string jsonCfg = templateJsonCfg;
    update_json_template(jsonCfg, "_MAXIMUM_", std::to_string(max));
    update_json_template(jsonCfg, "_SCALE_", std::to_string(scale));

    float errBound = max * scale;
    // Use 10% error margin to test due to using v.large values in our random data
    float insideErrBound  = errBound * 0.9;
    float outsideErrBound = errBound * 1.1;

    SUBCASE("inside")
    {
        // Generate some data that meets the requirements of the result.
        auto otherData_fp32 = data_fp32;
        std::for_each(std::begin(otherData_fp32), std::end(otherData_fp32), [insideErrBound](auto& value) {
            if (std::abs(value) != 0.0 && !std::isinf(value) && !std::isnan(value))
                value += insideErrBound;
        });
        const auto referenceTensor =
            TosaTensor("out1", tosa_datatype_fp64_t, shape, reinterpret_cast<uint8_t*>(data_fp64.data()));
        const auto implementationTensor =
            TosaTensor("out1", tosa_datatype_fp32_t, shape, reinterpret_cast<uint8_t*>(otherData_fp32.data()));
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
            TosaTensor("out1", tosa_datatype_fp64_t, shape, reinterpret_cast<uint8_t*>(data_fp64.data()));
        const auto implementationTensor =
            TosaTensor("out1", tosa_datatype_fp32_t, shape, reinterpret_cast<uint8_t*>(otherData_fp32.data()));
        REQUIRE_FALSE(
            tvf_verify_data(referenceTensor.cTensor(), nullptr, implementationTensor.cTensor(), jsonCfg.c_str()));
    }
}
TEST_SUITE_END();    // verify
