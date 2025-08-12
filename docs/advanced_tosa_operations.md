# Advanced TOSA MLIR Dialect Operations

This document describes the comprehensive set of advanced TOSA operations that extend the standard TOSA specification with sophisticated tensor processing capabilities.

## Overview

The advanced TOSA operations provide deep MLIR dialect integration with six major operation categories:

1. **TensorFusion** - Multi-input tensor fusion with configurable strategies
2. **SpectralTransform** - FFT/IFFT and frequency domain operations
3. **AdvancedActivation** - Sophisticated activation functions beyond standard TOSA
4. **TensorDecomposition** - Matrix decomposition operations (SVD, QR, LU, Cholesky)
5. **StatisticalOps** - Advanced statistical analysis operations
6. **GeometricOps** - Geometric transformations and distance metrics

## Operation Details

### TensorFusion Operations

#### Overview
TensorFusion operations enable sophisticated multi-input tensor combination strategies.

#### Supported Strategies
- **CONCATENATE (0)**: Concatenates tensors along specified axis
- **ELEMENT_WISE_ADD (1)**: Element-wise addition with broadcasting
- **WEIGHTED_SUM (2)**: Weighted linear combination of inputs
- **ATTENTION_FUSION (3)**: Attention-based adaptive fusion

#### Attributes
```cpp
TosaFusionAttribute {
    int strategy;           // Fusion strategy type
    int axis;              // Concatenation axis (-1 for last)
    float weight1;         // Weight for first input
    float weight2;         // Weight for second input
    vector<int> shape;     // Output shape
}
```

#### Example Usage
```cpp
// Weighted sum fusion
TosaFusionAttribute attr(2, -1, 0.7f, 0.3f, {4, 4});
OpTensorFusion<2, TOSA_REF_TYPE_FP32> fusion_op(sgt, &attr, id);
```

#### Mathematical Foundation
- **Weighted Sum**: `output = w1 * input1 + w2 * input2`
- **Attention Fusion**: `output = α * input1 + (1-α) * input2` where `α = ||input1||² / (||input1||² + ||input2||²)`

### SpectralTransform Operations

#### Overview
SpectralTransform operations provide frequency domain analysis capabilities.

#### Supported Transforms
- **FFT (0)**: Fast Fourier Transform
- **IFFT (1)**: Inverse Fast Fourier Transform
- **REAL_FFT (2)**: Real-valued FFT optimization
- **DCT (3)**: Discrete Cosine Transform

#### Attributes
```cpp
TosaSpectralAttribute {
    int transform_type;     // Transform type (0-3)
    int n_fft;             // FFT size
    bool normalized;       // Normalize output
    int axis;              // Transform axis (-1 for last)
}
```

#### Implementation Details
- Uses Cooley-Tukey FFT algorithm for optimal O(N log N) performance
- Supports power-of-2 and arbitrary length transforms
- Numerical stability optimizations for edge cases

#### Mathematical Foundation
- **FFT**: `X[k] = Σ(x[n] * e^(-2πikn/N))` for k = 0..N-1
- **DCT**: `X[k] = Σ(x[n] * cos(π*k*(2n+1)/(2N)))` with orthonormal scaling

### AdvancedActivation Operations

#### Overview
AdvancedActivation operations implement sophisticated activation functions.

#### Supported Activations
- **MISH (0)**: `x * tanh(softplus(x))`
- **SWISH (1)**: `x * sigmoid(x)`
- **GELU_TANH (2)**: GELU with tanh approximation
- **GELU_ERF (3)**: GELU with error function
- **SELU (4)**: Self-normalizing activation
- **ELU (5)**: Exponential Linear Unit
- **HARDSWISH (6)**: Hardware-friendly Swish

#### Mathematical Definitions
- **Mish**: `f(x) = x * tanh(ln(1 + e^x))`
- **GELU**: `f(x) = 0.5 * x * (1 + erf(x/√2))`
- **SELU**: `f(x) = λ * (α * (e^x - 1) if x < 0 else x)` with λ=1.0507, α=1.6733

#### Numerical Stability
- Careful handling of exponential overflow/underflow
- Approximations for extreme values
- Gradient-friendly implementations

### TensorDecomposition Operations

#### Overview
TensorDecomposition operations provide matrix factorization capabilities.

#### Supported Decompositions
- **SVD (0)**: Singular Value Decomposition
- **QR (1)**: QR Decomposition
- **LU (2)**: LU Decomposition with partial pivoting
- **CHOLESKY (3)**: Cholesky Decomposition

#### Attributes
```cpp
TosaDecompositionAttribute {
    int decomposition_type; // Decomposition type (0-3)
    bool compute_uv;        // Compute U and V matrices (SVD)
    bool full_matrices;     // Compute full-sized matrices
}
```

#### Mathematical Properties
- **SVD**: `A = U * Σ * V^T` with orthogonal U, V and diagonal Σ
- **QR**: `A = Q * R` with orthogonal Q and upper triangular R
- **LU**: `PA = LU` with permutation P, lower L, upper U
- **Cholesky**: `A = L * L^T` for positive definite A

#### Implementation Details
- Uses Eigen's numerically stable algorithms
- Automatic rank detection and handling
- Memory-efficient implementations

### StatisticalOps Operations

#### Overview
StatisticalOps operations provide advanced statistical analysis capabilities.

#### Supported Operations
- **ENTROPY (0)**: Shannon entropy calculation
- **MUTUAL_INFORMATION (1)**: Mutual information between variables
- **CORRELATION (2)**: Pearson correlation coefficient
- **COVARIANCE (3)**: Covariance matrix computation
- **MOMENT (4)**: Central moments of specified order
- **SKEWNESS (5)**: Third standardized moment
- **KURTOSIS (6)**: Fourth standardized moment (excess)

#### Mathematical Definitions
- **Entropy**: `H(X) = -Σ p(x) * log(p(x))`
- **Correlation**: `ρ = Cov(X,Y) / (σ_X * σ_Y)`
- **Skewness**: `E[((X-μ)/σ)³]`
- **Kurtosis**: `E[((X-μ)/σ)⁴] - 3`

### GeometricOps Operations

#### Overview
GeometricOps operations provide geometric transformations and distance metrics.

#### Supported Operations
- **EUCLIDEAN_DISTANCE (0)**: L2 distance metric
- **MANHATTAN_DISTANCE (1)**: L1 distance metric
- **COSINE_SIMILARITY (2)**: Cosine similarity measure
- **DOT_PRODUCT (3)**: Vector dot product
- **CROSS_PRODUCT (4)**: Vector cross product (3D)
- **AFFINE_TRANSFORM (5)**: Affine transformation
- **PERSPECTIVE_TRANSFORM (6)**: Perspective transformation

#### Mathematical Definitions
- **Euclidean**: `d = √(Σ(x_i - y_i)²)`
- **Cosine Similarity**: `cos(θ) = (A·B) / (||A|| * ||B||)`
- **Cross Product**: `A × B = (a₂b₃ - a₃b₂, a₃b₁ - a₁b₃, a₁b₂ - a₂b₁)`

## Performance Characteristics

### Computational Complexity
- **TensorFusion**: O(N) where N is tensor size
- **SpectralTransform**: O(N log N) for FFT, O(N²) for DCT
- **AdvancedActivation**: O(N) with vectorized implementations
- **TensorDecomposition**: O(N³) for dense matrices
- **StatisticalOps**: O(N) to O(N²) depending on operation
- **GeometricOps**: O(N) for vector operations

### Memory Usage
- Optimized for in-place operations where possible
- Temporary buffer allocation minimized
- Eigen's memory mapping for large tensors

### Numerical Stability
- IEEE 754 compliant floating-point operations
- Overflow/underflow protection
- Iterative refinement for ill-conditioned problems

## Integration with TOSA Ecosystem

### Type System Integration
- Full support for TOSA data types (FP16, FP32, FP64, INT8, INT16, INT32)
- Automatic type promotion and casting
- Rank-generic implementations (0-6 dimensions)

### Attribute System
- Custom attribute classes with validation
- JSON serialization support
- Backward compatibility with existing attributes

### Operation Factory Integration
- Seamless integration with existing op_factory system
- Template-based instantiation
- Runtime type and rank resolution

## Usage Examples

### Basic Operation Usage
```cpp
// Create fusion operation
TosaFusionAttribute fusion_attr(1, -1, 1.0f, 1.0f, {4, 4});
auto fusion_op = OpFactory::newOp<OpTensorFusion<2, TOSA_REF_TYPE_FP32>>(
    sgt, &fusion_attr, id);

// Create spectral transform
TosaSpectralAttribute spectral_attr(0, 64, true, -1);
auto fft_op = OpFactory::newOp<OpSpectralTransform<1, TOSA_REF_TYPE_FP32>>(
    sgt, &spectral_attr, id);
```

### Operation Chaining
```cpp
// Chain operations for complex processing
auto input = createTensor({64, 64});

// 1. Apply advanced activation
auto activated = applyAdvancedActivation(input, MISH);

// 2. Compute spectral transform
auto spectrum = applySpectralTransform(activated, FFT);

// 3. Analyze statistics
auto entropy = computeStatistical(spectrum, ENTROPY);
```

## Best Practices

### Performance Optimization
1. **Choose appropriate data types**: Use FP16 for memory-constrained environments
2. **Optimize tensor layouts**: Ensure data is contiguous for vectorization
3. **Batch operations**: Process multiple samples together when possible
4. **Cache-friendly access**: Access tensors in memory order

### Numerical Stability
1. **Input validation**: Check for NaN, infinity, and out-of-range values
2. **Conditioning**: Monitor condition numbers for matrix operations
3. **Precision**: Use higher precision for accumulation operations
4. **Regularization**: Add small epsilon values to prevent division by zero

### Memory Management
1. **RAII principles**: Use smart pointers and automatic resource management
2. **Buffer reuse**: Reuse temporary buffers across operations
3. **Memory pooling**: Use memory pools for frequent allocations
4. **Lazy evaluation**: Defer computation until results are needed

## Extending the Framework

### Adding New Operations
1. **Define operation class**: Inherit from GraphNode
2. **Create attribute class**: Define operation-specific parameters
3. **Implement eval() method**: Add mathematical implementation
4. **Add factory registration**: Integrate with op_factory system
5. **Create tests**: Add comprehensive unit and integration tests

### Adding New Attributes
1. **Inherit from TosaAttributeBase**: Follow existing patterns
2. **Implement serialization**: Add JSON support
3. **Add validation**: Ensure parameter constraints
4. **Document interface**: Provide clear API documentation

## Troubleshooting

### Common Issues
1. **Compilation errors**: Ensure Eigen3 and C++17 support
2. **Runtime crashes**: Check tensor shapes and data types
3. **Numerical issues**: Validate input ranges and conditioning
4. **Performance problems**: Profile memory access patterns

### Debugging Tools
1. **Debug modes**: Enable detailed logging and validation
2. **Tensor inspection**: Dump intermediate results
3. **Profiling**: Use performance analysis tools
4. **Unit tests**: Validate individual components

## Future Enhancements

### Planned Features
1. **GPU acceleration**: CUDA and OpenCL support
2. **Distributed operations**: Multi-node tensor processing
3. **Quantization support**: INT8/INT4 optimized implementations
4. **Custom kernels**: User-defined operation plugins

### Research Directions
1. **Sparse tensor support**: Efficient sparse matrix operations
2. **Automatic differentiation**: Gradient computation support
3. **Symbolic computation**: Expression optimization
4. **Hardware-specific optimizations**: ARM NEON, Intel AVX support

## References

1. TOSA Specification - https://git.mlplatform.org/tosa/specification.git/
2. MLIR Documentation - https://mlir.llvm.org/docs/
3. Eigen Library - https://eigen.tuxfamily.org/
4. NumPy Documentation - https://numpy.org/doc/
5. IEEE 754 Standard - https://standards.ieee.org/ieee/754/6210/

## License

Copyright (c) 2025 ARM Limited.
Licensed under the Apache License, Version 2.0.