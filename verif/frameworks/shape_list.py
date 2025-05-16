# Copyright (c) 2025, ARM Limited.
# SPDX-License-Identifier: Apache-2.0
# Shapes to be tested; default can be overwritten
shape_list = [
    (1,),
    (64,),
    (14, 19),
    (13, 21, 3),
    (1, 8, 16),
    (1, 4, 4, 4),
    (1, 8, 4, 17),
    (1, 4, 8, 19),
    (1, 32, 32, 8),
    (1, 7, 7, 9),
    (3, 1, 1, 7),
    (2, 2, 7, 7, 2),
    (1, 4, 8, 21, 17),
    (3, 32, 16, 16, 5),
]

# Custom shape lists
shape_list_conv2d = [
    (1, 1, 16, 16),
    (5, 2, 32, 32),
]

shape_list_linear = [
    (5, 3),
    (1, 3),
]

shape_list_matmul_2d = [
    ((16, 32), (32, 17)),
]

shape_list_matmul_3d = [
    ((8, 16, 32), (8, 32, 17)),
]
