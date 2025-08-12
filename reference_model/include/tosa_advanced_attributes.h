// Copyright (c) 2025 ARM Limited.
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

#ifndef TOSA_ADVANCED_ATTRIBUTES_H
#define TOSA_ADVANCED_ATTRIBUTES_H

#include "attribute.h"
#include <vector>

using namespace tosa;

namespace TosaReference
{

// Fusion operation attribute
class TosaFusionAttribute : public TosaAttributeBase
{
public:
    TosaFusionAttribute() = default;
    TosaFusionAttribute(int strategy, int axis, float weight1, float weight2, const std::vector<int>& shape)
        : _strategy(strategy), _axis(axis), _weight1(weight1), _weight2(weight2), _shape(shape) {}
    
    virtual TosaAttributeType attribute_type() const override {
        return TosaAttributeType_FusionAttribute;
    }
    
    // Getters
    int strategy() const { return _strategy; }
    int axis() const { return _axis; }
    float weight1() const { return _weight1; }
    float weight2() const { return _weight2; }
    const std::vector<int>& shape() const { return _shape; }
    
    // Setters
    void strategy(int value) { _strategy = value; }
    void axis(int value) { _axis = value; }
    void weight1(float value) { _weight1 = value; }
    void weight2(float value) { _weight2 = value; }
    void shape(const std::vector<int>& value) { _shape = value; }
    
private:
    int _strategy = 0;              // Fusion strategy type
    int _axis = -1;                 // Concatenation axis
    float _weight1 = 1.0f;          // Weight for first input
    float _weight2 = 1.0f;          // Weight for second input
    std::vector<int> _shape;        // Output shape
};

// Spectral transform attribute
class TosaSpectralAttribute : public TosaAttributeBase
{
public:
    TosaSpectralAttribute() = default;
    TosaSpectralAttribute(int transform_type, int n_fft, bool normalized, int axis)
        : _transform_type(transform_type), _n_fft(n_fft), _normalized(normalized), _axis(axis) {}
    
    virtual TosaAttributeType attribute_type() const override {
        return TosaAttributeType_SpectralAttribute;
    }
    
    // Getters
    int transform_type() const { return _transform_type; }
    int n_fft() const { return _n_fft; }
    bool normalized() const { return _normalized; }
    int axis() const { return _axis; }
    
    // Setters
    void transform_type(int value) { _transform_type = value; }
    void n_fft(int value) { _n_fft = value; }
    void normalized(bool value) { _normalized = value; }
    void axis(int value) { _axis = value; }
    
private:
    int _transform_type = 0;        // 0=FFT, 1=IFFT, 2=REAL_FFT, 3=DCT
    int _n_fft = 0;                 // FFT size
    bool _normalized = false;       // Whether to normalize output
    int _axis = -1;                 // Axis along which to compute transform
};

// Advanced activation attribute
class TosaActivationAttribute : public TosaAttributeBase
{
public:
    TosaActivationAttribute() = default;
    TosaActivationAttribute(int activation_type, float alpha, float beta)
        : _activation_type(activation_type), _alpha(alpha), _beta(beta) {}
    
    virtual TosaAttributeType attribute_type() const override {
        return TosaAttributeType_ActivationAttribute;
    }
    
    // Getters
    int activation_type() const { return _activation_type; }
    float alpha() const { return _alpha; }
    float beta() const { return _beta; }
    
    // Setters
    void activation_type(int value) { _activation_type = value; }
    void alpha(float value) { _alpha = value; }
    void beta(float value) { _beta = value; }
    
private:
    int _activation_type = 0;       // Activation function type
    float _alpha = 1.0f;           // Alpha parameter for some activations
    float _beta = 1.0f;            // Beta parameter for some activations
};

// Tensor decomposition attribute
class TosaDecompositionAttribute : public TosaAttributeBase
{
public:
    TosaDecompositionAttribute() = default;
    TosaDecompositionAttribute(int decomposition_type, bool compute_uv, bool full_matrices)
        : _decomposition_type(decomposition_type), _compute_uv(compute_uv), _full_matrices(full_matrices) {}
    
    virtual TosaAttributeType attribute_type() const override {
        return TosaAttributeType_DecompositionAttribute;
    }
    
    // Getters
    int decomposition_type() const { return _decomposition_type; }
    bool compute_uv() const { return _compute_uv; }
    bool full_matrices() const { return _full_matrices; }
    
    // Setters
    void decomposition_type(int value) { _decomposition_type = value; }
    void compute_uv(bool value) { _compute_uv = value; }
    void full_matrices(bool value) { _full_matrices = value; }
    
private:
    int _decomposition_type = 0;    // 0=SVD, 1=QR, 2=LU, 3=Cholesky
    bool _compute_uv = true;        // Compute U and V matrices for SVD
    bool _full_matrices = false;    // Compute full-sized matrices
};

// Statistical operations attribute
class TosaStatisticalAttribute : public TosaAttributeBase
{
public:
    TosaStatisticalAttribute() = default;
    TosaStatisticalAttribute(int statistical_type, int axis, int moment_order, bool bias_correction)
        : _statistical_type(statistical_type), _axis(axis), _moment_order(moment_order), _bias_correction(bias_correction) {}
    
    virtual TosaAttributeType attribute_type() const override {
        return TosaAttributeType_StatisticalAttribute;
    }
    
    // Getters
    int statistical_type() const { return _statistical_type; }
    int axis() const { return _axis; }
    int moment_order() const { return _moment_order; }
    bool bias_correction() const { return _bias_correction; }
    
    // Setters
    void statistical_type(int value) { _statistical_type = value; }
    void axis(int value) { _axis = value; }
    void moment_order(int value) { _moment_order = value; }
    void bias_correction(bool value) { _bias_correction = value; }
    
private:
    int _statistical_type = 0;      // Statistical operation type
    int _axis = -1;                 // Axis for reduction operations
    int _moment_order = 2;          // Order for moment calculations
    bool _bias_correction = false;  // Whether to apply bias correction
};

// Geometric operations attribute
class TosaGeometricAttribute : public TosaAttributeBase
{
public:
    TosaGeometricAttribute() = default;
    TosaGeometricAttribute(int geometric_type, const std::vector<float>& transform_matrix, bool normalize)
        : _geometric_type(geometric_type), _transform_matrix(transform_matrix), _normalize(normalize) {}
    
    virtual TosaAttributeType attribute_type() const override {
        return TosaAttributeType_GeometricAttribute;
    }
    
    // Getters
    int geometric_type() const { return _geometric_type; }
    const std::vector<float>& transform_matrix() const { return _transform_matrix; }
    bool normalize() const { return _normalize; }
    
    // Setters
    void geometric_type(int value) { _geometric_type = value; }
    void transform_matrix(const std::vector<float>& value) { _transform_matrix = value; }
    void normalize(bool value) { _normalize = value; }
    
private:
    int _geometric_type = 0;                    // Geometric operation type
    std::vector<float> _transform_matrix;       // Transformation matrix
    bool _normalize = false;                    // Whether to normalize results
};

}    // namespace TosaReference

#endif