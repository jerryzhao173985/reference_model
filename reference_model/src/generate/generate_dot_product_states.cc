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

#include <cmath>
#include <cstdint>

namespace
{

// Input index global variables
inline constexpr uint32_t P0 = 0;
inline constexpr uint32_t P1 = 1;
inline constexpr uint32_t P2 = 2;

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
        float sign   = (_r >> 31) == 0 ? +1 : -1;
        float pseudo = sign * (float)(_r & 0x7FFFFFFF) / (float)(0x7FFFFFFF);

        // Move index and calculate r value for the next index
        ++_index;
        _r = _r * _m + 1;

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
// State generators - equivalent to tosa_mi_data() in the TOSA specification
//
// Each call to the generator returns the next generated value with an
// auto incrementing index
//----------------------------------------------------------------------------//

// Test set 0 generator
// The aim of this generator is to check that sum of products with zero gives zero result.
class GeneratorS0 : public TosaReference::IDotProductGenerator
{
public:
    GeneratorS0(uint32_t p)
        : _p(p)
        , _set_data0(2 * 0)
        , _set_data1(2 * 0 + 1)
    {}
    float operator()(uint32_t k) override
    {
        unused(k);
        const float s0 = _set_data0();
        const float s1 = _set_data1();
        if (_p == P0)
            return s0 < 0.f ? 0.f : s1;
        else if (_p == P1)
            return s0 < 0.f ? s1 : 0.f;
        else
            return 0.f;
    }

private:
    uint32_t _p;
    PrimitiveGenerator _set_data0;
    PrimitiveGenerator _set_data1;
};

// Test set 1 generator
// The aim of this test set is to check values with large exponents.
class GeneratorS1 : public TosaReference::IDotProductGenerator
{
public:
    GeneratorS1(uint32_t p, uint32_t KS, float B)
        : _p(p)
        , _KS(KS)
        , _B(B)
        , _set_data(3 * 1 + p)
    {}
    float operator()(uint32_t k) override
    {
        unused(k);
        const float s = _set_data();
        float v       = 0.75f + 0.25f * s;
        if (_p != P2)
            return (_B / std::sqrt(_KS + 1)) * v;
        else
            return (_B * _B / (_KS + 1)) * v;
    }

private:
    uint32_t _p;
    uint32_t _KS;
    float _B;
    PrimitiveGenerator _set_data;
};

// Test set 2 generator
// The aim of this test set is to check rounding error when accumulating small values
// onto a large value. In this case the small values are of similar magnitude. If the
// implementation changes the order of the sum, then the test data must also be reordered
// so that the largest values occur first in the sum.
class GeneratorS2 : public TosaReference::IDotProductGenerator
{
public:
    GeneratorS2(uint32_t p, uint32_t KS)
        : _p(p)
        , _KS(KS)
        , _set_data(2 * 2 + p)
    {}
    float operator()(uint32_t k) override
    {
        const float s = _set_data();
        if (_p != P2)
            return k == 0 ? 1.f : s / std::sqrt(_KS);
        else
            return 0.f;
    }

private:
    uint32_t _p;
    uint32_t _KS;
    PrimitiveGenerator _set_data;
};

// Test set 3 generator
// The aim of this test set is to check rounding error when accumulating small values
// onto a large value. In this case the small values are of varying magnitude. If the
// implementation changes the order of the sum, then the test data must also be reordered
// so that the largest values occur first in the sum.
class GeneratorS3 : public TosaReference::IDotProductGenerator
{
public:
    GeneratorS3(uint32_t p)
        : _p(p)
        , _set_data(2 * 3 + p)
    {}
    float operator()(uint32_t k) override
    {
        const float s0 = _set_data();
        const float s1 = _set_data();
        if (_p != P2)
            return k == 0 ? 16.f : std::exp(2 * s0) * s1;
        else
            return 0.f;
    }

private:
    uint32_t _p;
    PrimitiveGenerator _set_data;
};

// Test set 4 generator
// The aim of this test set is to check a mixture of zero and non-zero products.
class GeneratorS4 : public TosaReference::IDotProductGenerator
{
public:
    GeneratorS4(uint32_t p, uint32_t KS, float B)
        : _p(p)
        , _KS(KS)
        , _B(B)
        , _set_data0(2 * 4 + 0)
        , _set_data1(2 * 4 + 1)
    {}
    float operator()(uint32_t k) override
    {
        const float s0 = _set_data0();
        const float s1 = _set_data1();
        if (_p == P0)
            return (k == _KS / 2) ? +0.5f : s0 < 0 ? 0.f : (_B / std::sqrt(_KS)) * s1;
        else if (_p == P1)
            return (k == _KS / 2) ? -0.5f : s0 < 0 ? (_B / std::sqrt(_KS)) * s1 : 0.f;
        else
            return 0.f;
    }

private:
    uint32_t _p;
    uint32_t _KS;
    float _B;
    PrimitiveGenerator _set_data0;
    PrimitiveGenerator _set_data1;
};

// Test set 5 generator
// The aim of this test set is to check signed inputs of large range.
class GeneratorS5 : public TosaReference::IDotProductGenerator
{
public:
    GeneratorS5(uint32_t p, uint32_t KS, float B)
        : _p(p)
        , _KS(KS)
        , _B(B)
        , _set_data(3 * 5 + p)
    {}
    float operator()(uint32_t k) override
    {
        unused(k);
        const float s = _set_data();
        if (_p != P2)
            return (_B / std::sqrt(_KS + 1)) * s;
        else
            return 0.f;
    }

private:
    uint32_t _p;
    uint32_t _KS;
    float _B;
    PrimitiveGenerator _set_data;
};

float getBoundParameter(const DType& dataType, const DType& accType)
{
    // Work out the bounds parameter value B for the given data and accumulator types
    // Returns value > 0.f on success
    float B = 0.f;
    if (dataType == DType::DType_FP16)
    {
        if (accType == DType::DType_FP16)
            B = 255.875f;    // (1<<8) - (1/8);
        else if (accType == DType::DType_FP32)
            B = 65504.f;    // (1<<16) - (1<<5);
    }
    else if (dataType == DType::DType_BF16)
    {
        if (accType == DType::DType_FP32)
            B = 18374686479671623680.f;    // (1<<64) - (1<<56)
    }
    else if (dataType == DType::DType_FP32)
    {
        if (accType == DType::DType_FP32)
            B = 18446742974197923840.f;    // (1<<64) - (1<<40)
    }
    return B;
}

}    // namespace

namespace TosaReference
{

std::unique_ptr<IDotProductGenerator> pickDotProductGenerator(const GenerateConfig& cfg)
{
    // Generators can only support 3 inputs
    if (cfg.inputPos > 2)
        return nullptr;

    const DotProductInfo& dpinfo = cfg.dotProductInfo;

    float B = getBoundParameter(cfg.dataType, dpinfo.accType);
    if (B > 0.f)
    {
        // Create the generator
        switch (dpinfo.s)
        {
            case 0:
                return std::make_unique<GeneratorS0>(cfg.inputPos);
            case 1:
                return std::make_unique<GeneratorS1>(cfg.inputPos, dpinfo.ks, B);
            case 2:
                return std::make_unique<GeneratorS2>(cfg.inputPos, dpinfo.ks);
            case 3:
                return std::make_unique<GeneratorS3>(cfg.inputPos);
            case 4:
                return std::make_unique<GeneratorS4>(cfg.inputPos, dpinfo.ks, B);
            case 5:
                return std::make_unique<GeneratorS5>(cfg.inputPos, dpinfo.ks, B);
            default:
                WARNING("[Generator][DP] Unsupported dot product test series for generator.");
                return nullptr;
        }
    }
    WARNING("[Generator][DP] Unsupported data types for generator.");
    return nullptr;
}

}    // namespace TosaReference
