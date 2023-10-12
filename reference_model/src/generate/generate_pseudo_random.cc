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
        _unidis = std::uniform_real_distribution<FP>(min, max);

        // Piecewise Constant distribution
        const std::array<double, 7> intervals{ min, min + 1000, -1000.0, 0.0, 1000.0, max - 1000, max };
        const std::array<double, 7> weights{ 1.0, 0.1, 1.0, 2.0, 1.0, 0.1, 1.0 };
        _pwcdis = std::piecewise_constant_distribution<FP>(intervals.begin(), intervals.end(), weights.begin());
    }

    FP getRandomUniformFloat()
    {
        return _unidis(_gen);
    }

    FP getRandomPWCFloat()
    {
        return _pwcdis(_gen);
    }

private:
    std::mt19937 _gen;
    std::uniform_real_distribution<FP> _unidis;
    std::piecewise_constant_distribution<FP> _pwcdis;
};

bool generateFP32(const TosaReference::GenerateConfig& cfg, void* data, size_t size)
{
    const TosaReference::PseudoRandomInfo& prinfo = cfg.pseudoRandomInfo;
    PseudoRandomGeneratorFloat<float> generator(prinfo.rngSeed);

    float* a     = reinterpret_cast<float*>(data);
    const auto T = TosaReference::numElementsFromShape(cfg.shape);
    for (auto t = 0; t < T; ++t)
    {
        a[t] = generator.getRandomPWCFloat();
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
