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
#include "dtype_limits.h"
#include "generate.h"
#include "generate_utils.h"

#include <algorithm>
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

// Random FP generator
template <typename FP>
class PseudoRandomGeneratorFloat
{
public:
    PseudoRandomGeneratorFloat(uint64_t seed)
        : _gen(seed)
    {
        // Uniform real distribution generates real values in the range [a, b]
        // and requires that b - a <= std::numeric_limits<FP>::max() so here
        // we choose some arbitrary values that satisfy that condition.
        constexpr auto min = std::numeric_limits<FP>::lowest() / 2;
        constexpr auto max = std::numeric_limits<FP>::max() / 2;
        static_assert(max <= std::numeric_limits<FP>::max() + min);

        setDistribution(min, max);
    }

    PseudoRandomGeneratorFloat(uint64_t seed, FP min, FP max)
        : _gen(seed)
    {
        setDistribution(min, max);
    }

    FP getRandomFloat()
    {
        if (_useUniform)
            return _unidis(_gen);
        else
            return _pwcdis(_gen);
    }

private:
    void setDistribution(FP min, FP max)
    {
        _unidis = std::uniform_real_distribution<FP>(min, max);

        // Piecewise Constant distribution for larger ranges
        // The code below needs to be careful with overflows.
        FP mid = (max / 2) + (min / 2);

        // We will give a higher probability for numbers around the mid-point.
        // This looks closer to real world data than a uniform distribution
        // as we'd expect to see much more values in [-1, 1] and [-1000, 1000]
        // than a uniform distribution would suggest
        const FP small_radius = 1.;
        const FP large_radius = 1000.;

        const std::array<FP, 7> intervals{ min, mid - large_radius, mid - small_radius,
                                           mid, mid + small_radius, mid + large_radius,
                                           max };

        // One weight for each interval in-between values in the intervals array
        const std::array<FP, 6> weights{ 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 };

        _pwcdis = std::piecewise_constant_distribution<FP>(intervals.begin(), intervals.end(), weights.begin());

        // Uniform distribution works well on smaller ranges and piecewise-constant will misbehave
        // if we cannot fit four large_radius-sized segments inside the full range.
        const double use_uniform_threshold = large_radius * 4;
        _useUniform                        = (std::abs(max - min) < use_uniform_threshold);
    }

    std::mt19937_64 _gen;
    std::uniform_real_distribution<FP> _unidis;
    std::piecewise_constant_distribution<FP> _pwcdis;
    bool _useUniform;
};

template <typename DataType>
bool generateFP(const TosaReference::GenerateConfig& cfg, DataType* data, size_t size)
{
    const TosaReference::PseudoRandomInfo& prinfo = cfg.pseudoRandomInfo;

    std::unique_ptr<PseudoRandomGeneratorFloat<float>> generator;
    bool roundMode = prinfo.round;

    if (prinfo.range.size() == 2)
    {
        const float min = std::stof(prinfo.range[0]);
        const float max = std::stof(prinfo.range[1]);
        generator       = std::make_unique<PseudoRandomGeneratorFloat<float>>(prinfo.rngSeed, min, max);
    }
    else
    {
        generator = std::make_unique<PseudoRandomGeneratorFloat<float>>(prinfo.rngSeed);
    }

    const auto T = TosaReference::numElementsFromShape(cfg.shape);
    const bool comparisonOp =
        (cfg.opType == Op::Op_EQUAL) || (cfg.opType == Op::Op_GREATER_EQUAL) || (cfg.opType == Op::Op_GREATER);

    if (cfg.dataType == DType::DType_BF16 || cfg.dataType == DType::DType_FP8E4M3 ||
        cfg.dataType == DType::DType_FP8E5M2)
    {
        for (auto t = 0; t < T; ++t)
        {
            auto f = generator->getRandomFloat();
            if (comparisonOp && (t % 4 == 0))
            {
                // Set every 4th value to 0 to enable better comparison testing
                f = 0.f;
            }
            else if (roundMode)
            {
                f = std::roundf(f);
            }
            data[t] = static_cast<DataType>(f);
        }
    }
    else
    {
        for (auto t = 0; t < T; ++t)
        {
            data[t] = static_cast<DataType>(generator->getRandomFloat());
            if (comparisonOp && (t % 4 == 0))
            {
                // Set every 4th value to 0 to enable better comparison testing
                data[t] = static_cast<DataType>(0.f);
            }
            else if (roundMode)
            {
                data[t] = static_cast<DataType>(std::roundf(data[t]));
            }
        }
    }
    return true;
}

// Random INT64 generator
class PseudoRandomGeneratorInteger
{
public:
    PseudoRandomGeneratorInteger(uint64_t seed, int64_t min, int64_t max)
        : _gen(seed)
    {
        setDistribution(min, max);
    }

    int64_t getRandomInteger()
    {
        return _unidis(_gen);
    }

    int64_t getRandomInteger(int64_t min, int64_t max)
    {
        typename std::uniform_int_distribution<int64_t>::param_type range(min, max);
        return _unidis(_gen, range);
    }

private:
    void setDistribution(int64_t min, int64_t max)
    {
        _unidis = std::uniform_int_distribution<int64_t>(min, max);
    }

    std::mt19937_64 _gen;
    std::uniform_int_distribution<int64_t> _unidis;
};

template <typename DataType>
bool shuffleINTbyRow(const TosaReference::GenerateConfig& cfg, DataType* data, size_t size)
{
    const TosaReference::PseudoRandomInfo& prinfo = cfg.pseudoRandomInfo;
    std::unique_ptr<PseudoRandomGeneratorInteger> generator;

    if (cfg.shape.size() != 2)
    {
        WARNING("[Generator][PR][INT] Shuffle only supports 2 dimensional tensors.");
        return false;
    }
    if (prinfo.range.size() != 2)
    {
        WARNING("[Generator][PR][INT] Cannot create un-ranged shuffle data.");
        return false;
    }

    DataType min = static_cast<DataType>(std::stoll(prinfo.range[0]));
    DataType max = static_cast<DataType>(std::stoll(prinfo.range[1]));
    if (min > max)
    {
        std::swap(min, max);
    }
    generator = std::make_unique<PseudoRandomGeneratorInteger>(prinfo.rngSeed, min, max);

    // Work out inclusive range
    const auto range = std::abs(max - min) + 1;
    const auto N     = cfg.shape[0];    // Number of rows
    const auto W     = cfg.shape[1];    // Width of rows
    if (W > range)
    {
        WARNING("[Generator][PR][INT] Cannot fill data size %d with given shuffle range %d.", W, range);
        return false;
    }

    std::vector<DataType> numbers(static_cast<size_t>(range));
    for (int n = 0; n < N; ++n)
    {
        // Fill in the numbers in range
        std::iota(numbers.begin(), numbers.end(), min);

        // Perform random shuffling
        for (auto num = numbers.begin(); num < numbers.end(); ++num)
        {
            std::swap(*num, numbers[static_cast<size_t>(generator->getRandomInteger())]);
        }
        // Copy amount of data required
        for (auto w = 0; w < W; ++w)
        {
            data[(n * W) + w] = numbers[static_cast<size_t>(w)];
        }
    }
    return true;
}

template <typename StorageType, TosaReference::TOSA_REF_TYPE TosaType>
bool generateINT(const TosaReference::GenerateConfig& cfg, StorageType* data, size_t size)
{
    const TosaReference::PseudoRandomInfo& prinfo = cfg.pseudoRandomInfo;
    std::unique_ptr<PseudoRandomGeneratorInteger> generator;

    const auto T = TosaReference::numElementsFromShape(cfg.shape);

    const int64_t dtypeMin = TosaReference::DtypeLimits<TosaType>::min;
    const int64_t dtypeMax = TosaReference::DtypeLimits<TosaType>::max;
    int64_t min, max;
    if (prinfo.range.size() == 2)
    {
        min = std::stoll(prinfo.range[0]);
        max = std::stoll(prinfo.range[1]);
        if (min > max)
        {
            std::swap(min, max);
        }
        min = std::max(std::min(min, dtypeMax), dtypeMin);
        max = std::min(std::max(max, dtypeMin), dtypeMax);
    }
    else
    {
        min = dtypeMin;
        max = dtypeMax;
    }

    generator = std::make_unique<PseudoRandomGeneratorInteger>(prinfo.rngSeed, min, max);

    const bool comparisonOp =
        (cfg.opType == Op::Op_EQUAL) || (cfg.opType == Op::Op_GREATER_EQUAL) || (cfg.opType == Op::Op_GREATER);

    for (auto t = 0; t < T; ++t)
    {
        auto value = generator->getRandomInteger();
        // Make sure we make more values match between tensors for comparison
        // operators.
        if (comparisonOp && (t % 4 == 0))
            value = 0;

        TosaReference::writeValue<StorageType, TosaType>(value, t, data);
    }
    return true;
}
}    // namespace

namespace TosaReference
{
bool generatePseudoRandom(const GenerateConfig& cfg, void* data, size_t size)
{
    // Check we support the operator
    if (cfg.opType == Op::Op_UNKNOWN)
    {
        WARNING("[Generator][PR] Unknown operator.");
        return false;
    }
    if (cfg.pseudoRandomInfo.range.size() != 0 && cfg.pseudoRandomInfo.range.size() != 2)
    {
        WARNING("[Generator][PR] Invalid range");
        return false;
    }
    if (data == nullptr)
    {
        WARNING("[Generator][PR] Generator called with a null pointer.");
        return false;
    }

    switch (cfg.dataType)
    {
        case DType::DType_FP32: {
            float* outData = reinterpret_cast<float*>(data);
            return generateFP(cfg, outData, size);
        }
        case DType::DType_FP16: {
            float16* outData = reinterpret_cast<float16*>(data);
            return generateFP(cfg, outData, size);
        }
        case DType::DType_BF16: {
            bf16* outData = reinterpret_cast<bf16*>(data);
            return generateFP(cfg, outData, size);
        }
        case DType::DType_FP8E4M3: {
            fp8e4m3* outData = reinterpret_cast<fp8e4m3*>(data);
            return generateFP(cfg, outData, size);
        }
        case DType::DType_FP8E5M2: {
            fp8e5m2* outData = reinterpret_cast<fp8e5m2*>(data);
            return generateFP(cfg, outData, size);
        }
        case DType::DType_INT48: {
            int8_t* outData = reinterpret_cast<int8_t*>(data);
            return generateINT<int8_t, TosaReference::TOSA_REF_TYPE_INT48>(cfg, outData, size);
        }
        case DType::DType_SHAPE:
            [[fallthrough]];
        case DType::DType_INT32: {
            int32_t* outData = reinterpret_cast<int32_t*>(data);
            if (cfg.opType == Op::Op_SCATTER && cfg.inputPos == 1)
            {
                // Indices for SCATTER must not repeat - perform data shuffle
                return shuffleINTbyRow(cfg, outData, size);
            }
            else
            {
                return generateINT<int32_t, TosaReference::TOSA_REF_TYPE_INT32>(cfg, outData, size);
            }
        }
        case DType::DType_INT16: {
            if (cfg.unsignedData)
            {
                uint16_t* outData = reinterpret_cast<uint16_t*>(data);
                return generateINT<uint16_t, TosaReference::TOSA_REF_TYPE_UINT16>(cfg, outData, size);
            }
            else
            {
                int16_t* outData = reinterpret_cast<int16_t*>(data);
                return generateINT<int16_t, TosaReference::TOSA_REF_TYPE_INT16>(cfg, outData, size);
            }
        }
        case DType::DType_INT8: {
            if (cfg.unsignedData)
            {
                uint8_t* outData = reinterpret_cast<uint8_t*>(data);
                return generateINT<uint8_t, TosaReference::TOSA_REF_TYPE_UINT8>(cfg, outData, size);
            }
            else
            {
                int8_t* outData = reinterpret_cast<int8_t*>(data);
                return generateINT<int8_t, TosaReference::TOSA_REF_TYPE_INT8>(cfg, outData, size);
            }
        }
        case DType::DType_INT4: {
            int8_t* outData = reinterpret_cast<int8_t*>(data);
            return generateINT<int8_t, TosaReference::TOSA_REF_TYPE_INT4>(cfg, outData, size);
        }
        case DType::DType_BOOL: {
            int8_t* outData = reinterpret_cast<int8_t*>(data);
            return generateINT<int8_t, TosaReference::TOSA_REF_TYPE_BOOL>(cfg, outData, size);
        }
        default:
            WARNING("[Generator][PR] Unsupported type %s.", EnumNameDType(cfg.dataType));
            return false;
    }
}
}    // namespace TosaReference
