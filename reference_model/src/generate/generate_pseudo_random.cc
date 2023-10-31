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

// Random generator
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
        double range = std::abs(max - min);
        double mid;
        if (max == -min)
            mid = 0.f;
        else
            mid = (range / 2) + min;
        double segment = std::min<double>(1000.0, range / 5);

        const std::array<double, 7> intervals{
            min, min + segment, mid - segment, mid, mid + segment, max - segment, max
        };
        const std::array<double, 7> weights{ 1.0, 0.1, 1.0, 2.0, 1.0, 0.1, 1.0 };
        _pwcdis = std::piecewise_constant_distribution<FP>(intervals.begin(), intervals.end(), weights.begin());

        // Uniform distribution works well on smaller ranges
        _useUniform = (range < 2000.0);
    }

    std::mt19937 _gen;
    std::uniform_real_distribution<FP> _unidis;
    std::piecewise_constant_distribution<FP> _pwcdis;
    bool _useUniform;
};

bool generateFP32(const TosaReference::GenerateConfig& cfg, void* data, size_t size)
{
    const TosaReference::PseudoRandomInfo& prinfo = cfg.pseudoRandomInfo;

    PseudoRandomGeneratorFloat<float>* generator;

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

    float* a     = reinterpret_cast<float*>(data);
    const auto T = TosaReference::numElementsFromShape(cfg.shape);
    for (auto t = 0; t < T; ++t)
    {
        a[t] = generator->getRandomFloat();
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
        case DType::DType_FP32:
            return generateFP32(cfg, data, size);
        default:
            WARNING("[Generator][PR] Unsupported type.");
            return false;
    }
}
}    // namespace TosaReference
