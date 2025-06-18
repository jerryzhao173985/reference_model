#!/usr/bin/env python3

# Copyright (c) 2025 ARM Limited.
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

"""
Advanced TOSA Operation Test Generator

This module generates comprehensive test cases for advanced TOSA MLIR dialect operations.
It creates test data, attributes, and expected results for:
- TensorFusion operations
- SpectralTransform operations  
- AdvancedActivation functions
- TensorDecomposition algorithms
- StatisticalOps calculations
- GeometricOps transformations
"""

import numpy as np
import json
import os
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import scipy.fft
import scipy.linalg
import scipy.stats

# Add parent directory to path for TOSA imports
sys.path.append(str(Path(__file__).parent.parent))
from generator.tosa_utils import *
from generator.tosa_test_gen import TosaTestGen

class AdvancedTosaTestGen(TosaTestGen):
    """Generator for advanced TOSA operation tests"""
    
    def __init__(self, args):
        super().__init__(args)
        self.rng = np.random.default_rng(args.seed if hasattr(args, 'seed') else 42)
        
    def generate_tensor_fusion_test(self, op_name: str, strategy: int, 
                                   input_shapes: List[List[int]], 
                                   dtype: str = "FP32") -> Dict[str, Any]:
        """Generate test case for TensorFusion operation"""
        
        # Generate input tensors
        inputs = []
        for i, shape in enumerate(input_shapes):
            if dtype == "FP32":
                data = self.rng.uniform(-1.0, 1.0, shape).astype(np.float32)
            elif dtype == "FP16":
                data = self.rng.uniform(-1.0, 1.0, shape).astype(np.float16)
            else:
                data = self.rng.uniform(-1.0, 1.0, shape).astype(np.float64)
            inputs.append(data)
        
        # Compute expected output based on strategy
        if strategy == 0:  # CONCATENATE
            axis = 0  # Concatenate along first axis
            expected = np.concatenate(inputs, axis=axis)
        elif strategy == 1:  # ELEMENT_WISE_ADD
            expected = inputs[0] + inputs[1]
        elif strategy == 2:  # WEIGHTED_SUM
            weight1, weight2 = 0.7, 0.3
            expected = weight1 * inputs[0] + weight2 * inputs[1]
        elif strategy == 3:  # ATTENTION_FUSION
            # Simplified attention mechanism
            norm1 = np.sum(inputs[0] ** 2)
            norm2 = np.sum(inputs[1] ** 2)
            attention_weight = norm1 / (norm1 + norm2)
            expected = attention_weight * inputs[0] + (1 - attention_weight) * inputs[1]
        else:
            raise ValueError(f"Unknown fusion strategy: {strategy}")
        
        # Create test descriptor
        test_desc = {
            "op": op_name,
            "attribute": {
                "strategy": strategy,
                "axis": 0,
                "weight1": 0.7,
                "weight2": 0.3
            },
            "inputs": [f"input_{i}.npy" for i in range(len(inputs))],
            "output": "expected.npy",
            "dtype": dtype,
            "shapes": input_shapes
        }
        
        return {
            "test_desc": test_desc,
            "input_data": inputs,
            "expected_output": expected
        }
    
    def generate_spectral_transform_test(self, op_name: str, transform_type: int,
                                        input_shape: List[int], 
                                        dtype: str = "FP32") -> Dict[str, Any]:
        """Generate test case for SpectralTransform operation"""
        
        # Generate input signal
        if dtype == "FP32":
            input_data = self.rng.uniform(-1.0, 1.0, input_shape).astype(np.float32)
        else:
            input_data = self.rng.uniform(-1.0, 1.0, input_shape).astype(np.float64)
        
        # Compute expected output based on transform type
        if transform_type == 0:  # FFT
            expected = np.abs(scipy.fft.fft(input_data, axis=-1))
        elif transform_type == 1:  # IFFT
            # For IFFT, start with frequency domain data
            freq_data = scipy.fft.fft(input_data, axis=-1)
            expected = np.real(scipy.fft.ifft(freq_data, axis=-1))
        elif transform_type == 2:  # REAL_FFT
            expected = np.abs(scipy.fft.rfft(input_data, axis=-1))
        elif transform_type == 3:  # DCT
            expected = scipy.fft.dct(input_data, axis=-1, norm='ortho')
        else:
            raise ValueError(f"Unknown transform type: {transform_type}")
        
        test_desc = {
            "op": op_name,
            "attribute": {
                "transform_type": transform_type,
                "n_fft": input_shape[-1],
                "normalized": True,
                "axis": -1
            },
            "inputs": ["input.npy"],
            "output": "expected.npy",
            "dtype": dtype,
            "shape": input_shape
        }
        
        return {
            "test_desc": test_desc,
            "input_data": [input_data],
            "expected_output": expected
        }
    
    def generate_advanced_activation_test(self, op_name: str, activation_type: int,
                                         input_shape: List[int],
                                         dtype: str = "FP32") -> Dict[str, Any]:
        """Generate test case for AdvancedActivation operation"""
        
        # Generate input data
        if dtype == "FP32":
            input_data = self.rng.uniform(-3.0, 3.0, input_shape).astype(np.float32)
        else:
            input_data = self.rng.uniform(-3.0, 3.0, input_shape).astype(np.float64)
        
        # Compute expected output based on activation type
        if activation_type == 0:  # MISH
            softplus = np.log(1.0 + np.exp(input_data))
            expected = input_data * np.tanh(softplus)
        elif activation_type == 1:  # SWISH
            expected = input_data / (1.0 + np.exp(-input_data))
        elif activation_type == 2:  # GELU_TANH
            sqrt_2_over_pi = np.sqrt(2.0 / np.pi)
            tanh_arg = sqrt_2_over_pi * (input_data + 0.044715 * input_data**3)
            expected = 0.5 * input_data * (1.0 + np.tanh(tanh_arg))
        elif activation_type == 3:  # GELU_ERF
            from scipy.special import erf
            expected = 0.5 * input_data * (1.0 + erf(input_data / np.sqrt(2.0)))
        elif activation_type == 4:  # SELU
            alpha = 1.6732632423543772848170429916717
            scale = 1.0507009873554804934193349852946
            expected = np.where(input_data >= 0, 
                               scale * input_data,
                               scale * alpha * (np.exp(input_data) - 1.0))
        elif activation_type == 5:  # ELU
            alpha = 1.0
            expected = np.where(input_data >= 0,
                               input_data,
                               alpha * (np.exp(input_data) - 1.0))
        elif activation_type == 6:  # HARDSWISH
            relu6 = np.clip(input_data + 3.0, 0.0, 6.0)
            expected = input_data * relu6 / 6.0
        else:
            raise ValueError(f"Unknown activation type: {activation_type}")
        
        test_desc = {
            "op": op_name,
            "attribute": {
                "activation_type": activation_type,
                "alpha": 1.0,
                "beta": 1.0
            },
            "inputs": ["input.npy"],
            "output": "expected.npy",
            "dtype": dtype,
            "shape": input_shape
        }
        
        return {
            "test_desc": test_desc,
            "input_data": [input_data],
            "expected_output": expected
        }
    
    def generate_decomposition_test(self, op_name: str, decomp_type: int,
                                   matrix_shape: Tuple[int, int],
                                   dtype: str = "FP32") -> Dict[str, Any]:
        """Generate test case for TensorDecomposition operation"""
        
        m, n = matrix_shape
        
        # Generate input matrix
        if dtype == "FP32":
            input_matrix = self.rng.uniform(-1.0, 1.0, (m, n)).astype(np.float32)
        else:
            input_matrix = self.rng.uniform(-1.0, 1.0, (m, n)).astype(np.float64)
        
        # Ensure matrix properties for certain decompositions
        if decomp_type == 3:  # CHOLESKY - needs positive definite
            # Create a positive definite matrix
            A = self.rng.uniform(-1.0, 1.0, (m, m)).astype(input_matrix.dtype)
            input_matrix = A @ A.T + np.eye(m) * 0.1  # Add small diagonal for numerical stability
            n = m  # Cholesky requires square matrix
        
        # Compute expected outputs based on decomposition type
        if decomp_type == 0:  # SVD
            U, s, Vt = scipy.linalg.svd(input_matrix, full_matrices=False)
            expected_outputs = [U, s, Vt.T]  # Return V, not V^T
        elif decomp_type == 1:  # QR
            Q, R = scipy.linalg.qr(input_matrix)
            expected_outputs = [Q, R, np.zeros((min(m,n), min(m,n)))]  # Dummy third output
        elif decomp_type == 2:  # LU
            P, L, U = scipy.linalg.lu(input_matrix)
            expected_outputs = [L, U, P]  # Return L, U, and permutation
        elif decomp_type == 3:  # CHOLESKY
            L = scipy.linalg.cholesky(input_matrix, lower=True)
            expected_outputs = [L, np.zeros_like(L), np.zeros_like(L)]  # Only L is meaningful
        else:
            raise ValueError(f"Unknown decomposition type: {decomp_type}")
        
        test_desc = {
            "op": op_name,
            "attribute": {
                "decomposition_type": decomp_type,
                "compute_uv": True,
                "full_matrices": False
            },
            "inputs": ["input.npy"],
            "outputs": ["output_u.npy", "output_s.npy", "output_v.npy"],
            "dtype": dtype,
            "shape": [m, n]
        }
        
        return {
            "test_desc": test_desc,
            "input_data": [input_matrix],
            "expected_outputs": expected_outputs
        }
    
    def generate_statistical_test(self, op_name: str, stat_type: int,
                                 input_shape: List[int],
                                 dtype: str = "FP32") -> Dict[str, Any]:
        """Generate test case for StatisticalOps operation"""
        
        # Generate input data
        if dtype == "FP32":
            input_data = self.rng.uniform(0.1, 1.0, input_shape).astype(np.float32)
        else:
            input_data = self.rng.uniform(0.1, 1.0, input_shape).astype(np.float64)
        
        # For operations requiring two inputs
        input_data2 = None
        if stat_type in [1, 2]:  # MUTUAL_INFORMATION, CORRELATION
            if dtype == "FP32":
                input_data2 = self.rng.uniform(0.1, 1.0, input_shape).astype(np.float32)
            else:
                input_data2 = self.rng.uniform(0.1, 1.0, input_shape).astype(np.float64)
        
        # Compute expected output
        if stat_type == 0:  # ENTROPY
            # Normalize to probabilities
            probs = input_data / np.sum(input_data)
            expected = -np.sum(probs * np.log(probs + 1e-12))  # Add small epsilon
        elif stat_type == 1:  # MUTUAL_INFORMATION
            # Simplified mutual information
            x_flat = input_data.flatten()
            y_flat = input_data2.flatten()
            correlation = np.corrcoef(x_flat, y_flat)[0, 1]
            expected = -0.5 * np.log(1 - correlation**2 + 1e-12)
        elif stat_type == 2:  # CORRELATION
            x_flat = input_data.flatten()
            y_flat = input_data2.flatten()
            expected = np.corrcoef(x_flat, y_flat)[0, 1]
        elif stat_type == 4:  # MOMENT
            order = 2
            mean = np.mean(input_data)
            expected = np.mean((input_data - mean) ** order)
        elif stat_type == 5:  # SKEWNESS
            expected = scipy.stats.skew(input_data.flatten())
        elif stat_type == 6:  # KURTOSIS
            expected = scipy.stats.kurtosis(input_data.flatten())
        else:
            expected = np.mean(input_data)  # Default fallback
        
        inputs = ["input.npy"]
        input_list = [input_data]
        if input_data2 is not None:
            inputs.append("input2.npy")
            input_list.append(input_data2)
        
        test_desc = {
            "op": op_name,
            "attribute": {
                "statistical_type": stat_type,
                "axis": -1,
                "moment_order": 2,
                "bias_correction": False
            },
            "inputs": inputs,
            "output": "expected.npy",
            "dtype": dtype,
            "shape": input_shape
        }
        
        return {
            "test_desc": test_desc,
            "input_data": input_list,
            "expected_output": np.array([expected])  # Scalar output
        }
    
    def generate_geometric_test(self, op_name: str, geom_type: int,
                               input_shape: List[int],
                               dtype: str = "FP32") -> Dict[str, Any]:
        """Generate test case for GeometricOps operation"""
        
        # Generate two input vectors
        if dtype == "FP32":
            input1 = self.rng.uniform(-1.0, 1.0, input_shape).astype(np.float32)
            input2 = self.rng.uniform(-1.0, 1.0, input_shape).astype(np.float32)
        else:
            input1 = self.rng.uniform(-1.0, 1.0, input_shape).astype(np.float64)
            input2 = self.rng.uniform(-1.0, 1.0, input_shape).astype(np.float64)
        
        # Compute expected output
        if geom_type == 0:  # EUCLIDEAN_DISTANCE
            expected = np.linalg.norm(input1 - input2)
        elif geom_type == 1:  # MANHATTAN_DISTANCE
            expected = np.sum(np.abs(input1 - input2))
        elif geom_type == 2:  # COSINE_SIMILARITY
            norm1 = np.linalg.norm(input1)
            norm2 = np.linalg.norm(input2)
            expected = np.dot(input1.flatten(), input2.flatten()) / (norm1 * norm2 + 1e-12)
        elif geom_type == 3:  # DOT_PRODUCT
            expected = np.dot(input1.flatten(), input2.flatten())
        elif geom_type == 4:  # CROSS_PRODUCT (3D only)
            if len(input_shape) == 1 and input_shape[0] == 3:
                expected = np.cross(input1, input2)
            else:
                expected = np.zeros(3)  # Default for non-3D inputs
        else:
            expected = np.array([0.0])  # Default
        
        # Ensure expected is at least 1D
        if np.isscalar(expected):
            expected = np.array([expected])
        
        test_desc = {
            "op": op_name,
            "attribute": {
                "geometric_type": geom_type,
                "transform_matrix": [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],  # Identity transform
                "normalize": False
            },
            "inputs": ["input1.npy", "input2.npy"],
            "output": "expected.npy",
            "dtype": dtype,
            "shape": input_shape
        }
        
        return {
            "test_desc": test_desc,
            "input_data": [input1, input2],
            "expected_output": expected
        }
    
    def save_test_case(self, test_data: Dict[str, Any], output_dir: str, test_name: str):
        """Save a test case to the specified directory"""
        
        # Create output directory
        test_dir = Path(output_dir) / test_name
        test_dir.mkdir(parents=True, exist_ok=True)
        
        # Save test descriptor
        with open(test_dir / "desc.json", "w") as f:
            json.dump(test_data["test_desc"], f, indent=2)
        
        # Save input data
        input_data = test_data["input_data"]
        if isinstance(input_data, list):
            for i, data in enumerate(input_data):
                if i == 0:
                    filename = "input.npy" if len(input_data) == 1 else "input1.npy"
                elif i == 1:
                    filename = "input2.npy"
                else:
                    filename = f"input_{i}.npy"
                np.save(test_dir / filename, data)
        else:
            np.save(test_dir / "input.npy", input_data)
        
        # Save expected output(s)
        if "expected_outputs" in test_data:  # Multiple outputs (e.g., decomposition)
            for i, output in enumerate(test_data["expected_outputs"]):
                if i == 0:
                    filename = "output_u.npy"
                elif i == 1:
                    filename = "output_s.npy"
                elif i == 2:
                    filename = "output_v.npy"
                else:
                    filename = f"output_{i}.npy"
                np.save(test_dir / filename, output)
        else:  # Single output
            np.save(test_dir / "expected.npy", test_data["expected_output"])

def main():
    parser = argparse.ArgumentParser(description="Generate advanced TOSA operation tests")
    parser.add_argument("--output-dir", default="./advanced_tests", 
                       help="Output directory for generated tests")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--operations", nargs="+", 
                       choices=["tensor_fusion", "spectral_transform", "advanced_activation",
                               "tensor_decomposition", "statistical_ops", "geometric_ops", "all"],
                       default=["all"], help="Operations to generate tests for")
    parser.add_argument("--dtypes", nargs="+", choices=["FP16", "FP32", "FP64"],
                       default=["FP32"], help="Data types to test")
    
    args = parser.parse_args()
    generator = AdvancedTosaTestGen(args)
    
    operations = args.operations
    if "all" in operations:
        operations = ["tensor_fusion", "spectral_transform", "advanced_activation",
                     "tensor_decomposition", "statistical_ops", "geometric_ops"]
    
    for dtype in args.dtypes:
        for op in operations:
            print(f"Generating tests for {op} with dtype {dtype}...")
            
            if op == "tensor_fusion":
                for strategy in range(4):  # 4 fusion strategies
                    test_data = generator.generate_tensor_fusion_test(
                        "TensorFusion", strategy, [[4, 4], [4, 4]], dtype)
                    generator.save_test_case(test_data, args.output_dir, 
                                           f"tensor_fusion_strategy_{strategy}_{dtype}")
            
            elif op == "spectral_transform":
                for transform_type in range(4):  # 4 transform types
                    test_data = generator.generate_spectral_transform_test(
                        "SpectralTransform", transform_type, [16], dtype)
                    generator.save_test_case(test_data, args.output_dir,
                                           f"spectral_transform_{transform_type}_{dtype}")
            
            elif op == "advanced_activation":
                for activation_type in range(7):  # 7 activation types
                    test_data = generator.generate_advanced_activation_test(
                        "AdvancedActivation", activation_type, [8, 8], dtype)
                    generator.save_test_case(test_data, args.output_dir,
                                           f"advanced_activation_{activation_type}_{dtype}")
            
            elif op == "tensor_decomposition":
                for decomp_type in range(4):  # 4 decomposition types
                    if decomp_type == 3 and dtype == "FP16":  # Skip Cholesky for FP16
                        continue
                    test_data = generator.generate_decomposition_test(
                        "TensorDecomposition", decomp_type, (6, 6), dtype)
                    generator.save_test_case(test_data, args.output_dir,
                                           f"tensor_decomposition_{decomp_type}_{dtype}")
            
            elif op == "statistical_ops":
                for stat_type in [0, 2, 4, 5, 6]:  # Selected statistical operations
                    test_data = generator.generate_statistical_test(
                        "StatisticalOps", stat_type, [10, 10], dtype)
                    generator.save_test_case(test_data, args.output_dir,
                                           f"statistical_ops_{stat_type}_{dtype}")
            
            elif op == "geometric_ops":
                for geom_type in range(5):  # 5 geometric operations
                    shape = [3] if geom_type == 4 else [8]  # Cross product needs 3D
                    test_data = generator.generate_geometric_test(
                        "GeometricOps", geom_type, shape, dtype)
                    generator.save_test_case(test_data, args.output_dir,
                                           f"geometric_ops_{geom_type}_{dtype}")
    
    print(f"Test generation complete. Tests saved to {args.output_dir}")

if __name__ == "__main__":
    main()