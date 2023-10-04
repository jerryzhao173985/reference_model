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

#include "generate_dot_product.h"
#include "generate_utils.h"

#include <cstdint>

namespace
{

// Input index global variables
inline constexpr uint32_t P0 = 0;
inline constexpr uint32_t P1 = 1;

// Unused helper function
template <typename... Args>
inline void unused(Args&&...)
{}

// Primitive generator class
//
// Yields a new value on function operator access and increases the
// index by one
class PrimitiveGenerator
{
public:
    PrimitiveGenerator(uint32_t S)
        : _S(S)
        , _m(0)
        , _r(0)
        , _index(0)
    {
        _m = (8 * _S + 1) * 0x705A5E75;
        _r = _m + 1;
    }

    [[nodiscard]] float operator()()
    {
        _r           = _r * _m + 1;
        float sign   = (_r >> 31) == 0 ? +1 : -1;
        float pseudo = sign * (float)(_r & 0x7FFFFFFF) / (float)(0x7FFFFFFF);
        ++_index;

        return pseudo;
    }

    uint32_t index()
    {
        return _index;
    }

private:
    uint32_t _S;
    uint32_t _m;
    uint32_t _r;
    uint32_t _index;
};

//----------------------------------------------------------------------------//
// State generators
//----------------------------------------------------------------------------//

// S0 generator
class GeneratorS0 : public TosaReference::IDotProductGenerator
{
public:
    GeneratorS0(uint32_t p)
        : _p(p)
        , _s0(0)    // set_data(2*S)
        , _s1(1)    // set_data(2*S+1)
    {}
    float operator()(uint32_t k) override
    {
        unused(k);
        const float s0 = _s0();
        const float s1 = _s1();
        if (_p == P0)
            return s0 < 0.f ? 0.f : s1;
        else
            return s0 < 0.f ? s1 : 0.f;
    }

private:
    uint32_t _p;
    PrimitiveGenerator _s0;
    PrimitiveGenerator _s1;
};

}    // namespace

namespace TosaReference
{

std::unique_ptr<IDotProductGenerator> pickDotProductGenerator(const GenerateConfig& cfg)
{
    const DotProductInfo& dpinfo = cfg.dotProductInfo;
    switch (dpinfo.s)
    {
        case 0:
            return std::make_unique<GeneratorS0>(cfg.inputPos);
        default:
            return nullptr;
    }
    return nullptr;
}

}    // namespace TosaReference