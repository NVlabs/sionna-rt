#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Utilities for computation with complex numbers"""

import drjit as dr
import mitsuba as mi
from typing import Tuple, Literal

def cpx_add(
    a: Tuple[mi.TensorXf, mi.TensorXf],
    b: Tuple[mi.TensorXf, mi.TensorXf]
) -> Tuple[mi.TensorXf, mi.TensorXf]:
    r"""Element-wise addition of two complex-valued tensors

    Each tensor is represented as a tuple of two real-valued tensors,
    corresponding to the real and imaginary part, respectively.

    :param a: First tensor
    :param b: Second tensor
    """
    return (a[0] + b[0], a[1] + b[1])

def cpx_sub(
    a: Tuple[mi.TensorXf, mi.TensorXf],
    b: Tuple[mi.TensorXf, mi.TensorXf]
) -> Tuple[mi.TensorXf, mi.TensorXf]:
    r"""Element-wise subtraction of a complex-valued tensor from another

    Each tensor is represented as a tuple of two real-valued tensors,
    corresponding to the real and imaginary part, respectively.

    :param a: First tensor
    :param b: Second tensor which is subtracted from the first
    """
    return (a[0] - b[0], a[1] - b[1])

def cpx_mul(
    a: Tuple[mi.TensorXf, mi.TensorXf],
    b: Tuple[mi.TensorXf, mi.TensorXf]
) -> Tuple[mi.TensorXf, mi.TensorXf]:
    r"""Element-wise multiplication of two complex-valued tensors

    Each tensor is represented as a tuple of two real-valued tensors,
    corresponding to the real and imaginary part, respectively.

    :param a: First tensor
    :param b: Second tensor
    """
    re = a[0]*b[0] - a[1]*b[1]
    im = a[0]*b[1] + a[1]*b[0]
    return (re, im)

def cpx_div(
    a: Tuple[mi.TensorXf, mi.TensorXf],
    b: Tuple[mi.TensorXf, mi.TensorXf]
) -> Tuple[mi.TensorXf, mi.TensorXf]:
    r"""Element-wise division of a complex-valued tensor by another

    Each tensor is represented as a tuple of two real-valued tensors,
    corresponding to the real and imaginary part, respectively.

    :param a: First tensor
    :param b: Second tensor by which the first is divided
    """
    d = dr.rcp(dr.square(b[0]) + dr.square(b[1]))
    re = (a[0]*b[0] + a[1]*b[1]) * d
    im = (a[1]*b[0] - a[0]*b[1]) * d
    return (re, im)

def cpx_exp(
    x: Tuple[mi.TensorXf, mi.TensorXf]
) -> Tuple[mi.TensorXf, mi.TensorXf]:
    r"""Element-wise exponential of a complex-valued tensor

    The tensor is represented as a tuple of two real-valued tensors,
    corresponding to the real and imaginary part, respectively.

    :param x: A tensor
    """
    exp_re = dr.exp(x[0])
    sin_im, cos_im = dr.sincos(x[1])
    return (exp_re*cos_im, exp_re*sin_im)

def cpx_abs(
    x: Tuple[mi.TensorXf, mi.TensorXf]
) -> mi.TensorXf:
    r"""Element-wise absolute value of a complex-valued tensor

    The tensor is represented as a tuple of two real-valued tensors,
    corresponding to the real and imaginary part, respectively.

    :param x: A tensor
    """
    return dr.safe_sqrt(cpx_abs_square(x))

def cpx_abs_square(
    x: Tuple[mi.TensorXf, mi.TensorXf]
) -> mi.TensorXf:
    r"""Element-wise absolute squared value of a complex-valued tensor

    The tensor is represented as a tuple of two real-valued tensors,
    corresponding to the real and imaginary part, respectively.

    :param x: A tensor
    """
    return dr.square(x[0]) + dr.square(x[1])

def cpx_sqrt(
    x: Tuple[mi.TensorXf, mi.TensorXf]
) -> Tuple[mi.TensorXf, mi.TensorXf]:
    r"""Element-wise square root of a complex-valued tensor

    The tensor is represented as a tuple of two real-valued tensors,
    corresponding to the real and imaginary part, respectively.

    The following formula is implemented to compute the square roots of complex
    numbers:
    https://en.wikipedia.org/wiki/Square_root#Algebraic_formula

    :param x: A tensor
    """
    x_r = x[0]
    x_i = x[1]
    r = dr.safe_sqrt(dr.square(x_r) + dr.square(x_i))
    y_r = dr.safe_sqrt(0.5*(r + x_r))
    y_i = dr.sign(x_i)*dr.safe_sqrt(0.5*(r - x_r))
    return (y_r, y_i)

def cpx_convert(
    x: Tuple[mi.TensorXf, mi.TensorXf],
    out_type: Literal["numpy", "jax", "tf", "torch"]
):
    r"""
    Converts a complex-valued tensor to any of the supported frameworks

    The tensor is represented as a tuple of two real-valued tensors,
    corresponding to the real and imaginary part, respectively.

    Note that the chosen framework must be installed for this function
    to work.

    :param x: A tensor
    :param out_type: Name of the target framework. Currently supported
        are `Numpy <https://numpy.org>`_ ("numpy"),
        `Jax <https://jax.readthedocs.io/en/latest/index.html>`_ ("jax"),
        `TensorFlow <https://www.tensorflow.org>`_ ("tf"),
        and `PyTorch <https://pytorch.org>`_ ("torch").

    :return type: :py:class:`np.array` | :py:class:`jax.array` |
        :py:class:`tf.Tensor` | :py:class:`torch.tensor`
    """
    x0, x1 = x

    if out_type == "numpy":
        return x0.numpy() + 1j * x1.numpy()
    elif out_type == "tf":
        try:
            import tensorflow as tf # pylint: disable=import-outside-toplevel
        except ImportError as e:
            raise ImportError("Please install TensorFlow to use this feature.")\
                  from e
        return tf.complex(x0.tf(), x1.tf())
    elif out_type == "torch":
        try:
            import torch # pylint: disable=import-outside-toplevel
        except ImportError as e:
            raise ImportError("Please install PyTorch to use this feature.") \
                  from e
        return torch.complex(x0.torch(), x1.torch())
    elif out_type == "jax":
        try:
            from jax import lax # pylint: disable=import-outside-toplevel
        except ImportError as e:
            raise ImportError("Please install Jax to use this feature.") from e
        return lax.complex(x0.jax(), x1.jax())
    else:
        raise ValueError(f"Unsupported target: {out_type}")
