// Copyright (c) 2023, ARM Limited.
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
#include "generate.h"
#include "generate_utils.h"
#include "half.hpp"

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
        double mid = (max / 2) + (min / 2);

        const std::array<double, 5> intervals{ min, (min / 2) + (mid / 2), mid, (mid / 2) + (max / 2), max };
        // One weight for each interval in-between values in the intervals array
        const std::array<double, 4> weights{ 1.0, 1.0, 1.0, 1.0 };

        _pwcdis = std::piecewise_constant_distribution<FP>(intervals.begin(), intervals.end(), weights.begin());

        // Uniform distribution works well on smaller ranges
        _useUniform = (std::abs(max - min) < 2000.0);
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

    PseudoRandomGeneratorFloat<float>* generator;
    bool roundMode = prinfo.round;

    if (prinfo.range.size() == 2)
    {
        const float min = std::stof(prinfo.range[0]);
        const float max = std::stof(prinfo.range[1]);
        generator       = new PseudoRandomGeneratorFloat<float>(prinfo.rngSeed, min, max);
    }
    else
    {
        generator = new PseudoRandomGeneratorFloat<float>(prinfo.rngSeed);
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

// Random INT generator
template <typename INT>
class PseudoRandomGeneratorInteger
{
public:
    PseudoRandomGeneratorInteger(uint64_t seed)
        : _gen(seed)
    {
        constexpr auto min = std::numeric_limits<INT>::min();
        constexpr auto max = std::numeric_limits<INT>::max();

        setDistribution(min, max);
    }

    PseudoRandomGeneratorInteger(uint64_t seed, INT min, INT max)
        : _gen(seed)
    {
        setDistribution(min, max);
    }

    INT getRandomInteger()
    {
        return _unidis(_gen);
    }

    INT getRandomInteger(INT min, INT max)
    {
        typename std::uniform_int_distribution<INT>::param_type range(min, max);
        return _unidis(_gen, range);
    }

private:
    void setDistribution(INT min, INT max)
    {
        _unidis = std::uniform_int_distribution<INT>(min, max);
    }

    std::mt19937_64 _gen;
    std::uniform_int_distribution<INT> _unidis;
};

template <typename DataType>
bool shuffleINTbyRow(const TosaReference::GenerateConfig& cfg, DataType* data, size_t size)
{
    const TosaReference::PseudoRandomInfo& prinfo = cfg.pseudoRandomInfo;
    PseudoRandomGeneratorInteger<DataType>* generator;

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

    const int32_t min = std::stoi(prinfo.range[0]);
    const int32_t max = std::stoi(prinfo.range[1]);
    generator         = new PseudoRandomGeneratorInteger<DataType>(prinfo.rngSeed, min, max);

    // Work out inclusive range
    const auto range = std::abs(max - min) + 1;
    const auto N     = cfg.shape[0];    // Number of rows
    const auto W     = cfg.shape[1];    // Width of rows
    if (W > range)
    {
        WARNING("[Generator][PR][INT] Cannot fill data size %d with given shuffle range %d.", W, range);
        return false;
    }

    std::vector<DataType> numbers(range);
    for (int n = 0; n < N; ++n)
    {
        // Fill in the numbers in range
        std::iota(numbers.begin(), numbers.end(), min);

        // Perform random shuffling
        for (auto num = numbers.begin(); num < numbers.end(); ++num)
        {
            std::swap(*num, numbers[generator->getRandomInteger()]);
        }
        // Copy amount of data required
        for (auto w = 0; w < W; ++w)
        {
            data[(n * W) + w] = numbers[w];
        }
    }
    return true;
}

template <typename DataType>
bool generateINT(const TosaReference::GenerateConfig& cfg, DataType* data, size_t size)
{
    const TosaReference::PseudoRandomInfo& prinfo = cfg.pseudoRandomInfo;
    PseudoRandomGeneratorInteger<DataType>* generator;

    const auto T = TosaReference::numElementsFromShape(cfg.shape);

    if (prinfo.range.size() == 2)
    {
        const int32_t min = std::stoi(prinfo.range[0]);
        const int32_t max = std::stoi(prinfo.range[1]);
        generator         = new PseudoRandomGeneratorInteger<DataType>(prinfo.rngSeed, min, max);
    }
    else
    {
        generator = new PseudoRandomGeneratorInteger<DataType>(prinfo.rngSeed);
    }

    for (auto t = 0; t < T; ++t)
    {
        data[t] = generator->getRandomInteger();
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

    switch (cfg.dataType)
    {
        case DType::DType_FP32: {
            float* outData = reinterpret_cast<float*>(data);
            return generateFP(cfg, outData, size);
        }
        case DType::DType_FP16: {
            half_float::half* outData = reinterpret_cast<half_float::half*>(data);
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
        case DType::DType_INT32: {
            int32_t* outData = reinterpret_cast<int32_t*>(data);
            if (cfg.opType == Op::Op_SCATTER && cfg.inputPos == 1)
            {
                // Indices for SCATTER must not repeat - perform data shuffle
                return shuffleINTbyRow(cfg, outData, size);
            }
            else
            {
                return generateINT(cfg, outData, size);
            }
        }
        default:
            WARNING("[Generator][PR] Unsupported type.");
            return false;
    }
}
}    // namespace TosaReference
