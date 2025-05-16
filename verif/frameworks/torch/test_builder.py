# Copyright (c) 2024-2025, ARM Limited.
# SPDX-License-Identifier: Apache-2.0
import torch
import torch.nn.functional as F


class TBuilder:
    """Container of PyTorch modules that will be used to build into small
    networks for our tests"""

    def __init__(self):
        pass

    class Log(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return torch.log(x)

    class Exp(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return torch.exp(x)

    class Neg(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return torch.neg(x)

    class Floor(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return torch.floor(x)

    class Rsqrt(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return torch.rsqrt(x)

    class BitwiseNot(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return torch.bitwise_not(x)

    class Ceil(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return torch.ceil(x)

    class Rcp(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return torch.reciprocal(x)

    class Minimum(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, a, b):
            return torch.minimum(a, b)

    class Maximum(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, a, b):
            return torch.maximum(a, b)

    class LogicalOr(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, a, b):
            return torch.logical_or(a, b)

    class Add(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x, y):
            return torch.add(x, y)

    class Sub(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x, y):
            return torch.sub(x, y, alpha=1)

    class Equal(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, a, b):
            return torch.eq(a, b)

    class GreaterEqual(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, a, b):
            return torch.ge(a, b)

    class Greater(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, a, b):
            return torch.gt(a, b)

    class Less(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, a, b):
            return torch.lt(a, b)

    class LessEqual(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, a, b):
            return torch.le(a, b)

    class BitwiseAnd(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, a, b):
            return torch.bitwise_and(a, b)

    class BitwiseOr(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, a, b):
            return torch.bitwise_or(a, b)

    class BitwiseXor(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, a, b):
            return torch.bitwise_xor(a, b)

    class Mul(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x, y):
            return torch.mul(x, y)

    class Div(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x, y):
            return torch.div(x, y)

    class ReduceMean(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return torch.mean(x, 0, False)

    class ReduceSum(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return torch.sum(x, 0, False)

    class ReduceAny(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return torch.any(x, 0, False)

    class ReduceAll(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return torch.all(x)

    class Squeeze(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return torch.squeeze(x)

    class SqueezeDim(torch.nn.Module):
        def __init__(self, dim):
            self.dim = dim
            super().__init__()

        def forward(self, x):
            return torch.squeeze(x, self.dim)

    class MatMul(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x, y):
            return torch.matmul(x, y)

    class Mm(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x, y):
            return torch.mm(x, y)

    class Bmm(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x, y):
            return torch.bmm(x, y)

    class Linear(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(3, 5)

        def forward(self, x):
            return self.fc(x)

    class FillInt(torch.nn.Module):
        def __init__(self, shape, value):
            super().__init__()
            self.shape = shape
            self.value = torch.tensor(value, dtype=torch.int32).item()

        def forward(self, x):
            return torch.full(
                self.shape,
                self.value,
                dtype=torch.int32,
            )

    class FillFloat(torch.nn.Module):
        def __init__(self, shape, value):
            super().__init__()
            self.shape = shape
            self.value = torch.tensor(value, dtype=torch.float32).item()

        def forward(self, x):
            return torch.full(
                self.shape,
                self.value,
                dtype=torch.float32,
            )

    class Tanh(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return torch.tanh(x)

    class Sigmoid(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return torch.sigmoid(x)

    class Relu(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return torch.relu(x)

    class LeakyRelu(torch.nn.Module):
        def __init__(self, alpha):
            super().__init__()
            self.alpha = torch.tensor(alpha, dtype=torch.float32).item()

        def forward(self, x):
            return F.leaky_relu(x, negative_slope=self.alpha)

    class Gelu(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return F.gelu(x)

    class Abs(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return torch.abs(x)

    class Concat(torch.nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.dim = dim

        def forward(self, a, b):
            return torch.cat([a, b], self.dim) if a.shape != () else torch.stack([a, b])

    class AdaptiveAvgPool2d(torch.nn.Module):
        def __init__(self, output_size):
            super().__init__()
            self.adaptive_avgpool2d = torch.nn.AdaptiveAvgPool2d(
                output_size=output_size
            )

        def forward(self, x):
            return self.adaptive_avgpool2d(x)

    class AvgPool2d(torch.nn.Module):
        def __init__(self, stride, kernel_size, padding):
            super().__init__()
            self.stride = stride
            self.kernel_size = kernel_size
            self.padding = padding
            self.avg_pool2d = torch.nn.AvgPool2d(
                stride=self.stride,
                kernel_size=self.kernel_size,
                padding=self.padding,
                count_include_pad=False,
            )

        def forward(self, x):
            return self.avg_pool2d(x)

    class Conv2d(torch.nn.Module):
        def __init__(self, weight, stride, padding, dilation):
            super().__init__()
            torch.manual_seed(0)

            # weight = (out_channels, in_channels / groups, kernel_size[0], kernel_size[1])
            out_channels, in_channels, *kernel_size = weight.shape
            self.conv2d = torch.nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=False,
            )

            self.conv2d.weight = torch.nn.Parameter(weight)

        def forward(self, x):
            return self.conv2d(x)

    class Conv2dWithBias(torch.nn.Module):
        def __init__(self, weight, bias, stride, padding, dilation):
            super().__init__()
            torch.manual_seed(0)

            # weight = (out_channels, in_channels / groups, kernel_size[0], kernel_size[1])
            out_channels, in_channels, *kernel_size = weight.shape
            self.conv2d = torch.nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=True,
            )

            self.conv2d.weight = torch.nn.Parameter(weight)
            self.conv2d.bias = torch.nn.Parameter(bias)

        def forward(self, x):
            return self.conv2d(x)

    class MaxPool2d(torch.nn.Module):
        def __init__(self, stride, kernel_size, padding):
            super().__init__()
            self.maxpool2d = torch.nn.MaxPool2d(
                kernel_size=kernel_size,
                padding=padding,
                stride=stride,
                dilation=[1, 1],
            )

        def forward(self, x):
            return self.maxpool2d(x)
