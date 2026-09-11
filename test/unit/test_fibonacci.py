import time
from typing import Tuple

import drjit as dr
import mitsuba as mi
import numpy as np
import pytest

from sionna.rt.utils.fibonacci import fibonacci_lattice, _fibonacci_lattice_f32


def fibonacci_lattice_np(n: int) -> Tuple[np.ndarray, np.ndarray]:
    golden_ratio = (1. + np.sqrt(np.float64(5.))) / 2.
    ns = np.arange(0, n, dtype=np.float64)

    x = ns / golden_ratio
    x = x - np.floor(x)
    y = ns / (n - 1)
    return x.astype(np.float32), y.astype(np.float32)


def most_equal(a: np.ndarray, b: np.ndarray):
    n = a.size
    allowed_mismatches = n * 0.0004

    n_mismatches = np.sum(a != b)
    assert n_mismatches <= allowed_mismatches, f"Found {n_mismatches} mismatches (> {allowed_mismatches} allowed)"

    if n_mismatches > 0:
        diff = np.abs(a - b)
        assert np.sum(np.square(diff)) < 1e-8

    return True


@pytest.mark.parametrize("n", [1000, 10000, 100000])
def test01_fibonacci_lattice(n: int):
    """Check that the DrJit Fibonacci lattice implementation matches the numpy reference.
    This is especially important for the Metal backend, which does not support Float64.
    """
    x_np, y_np = fibonacci_lattice_np(n)
    x, y = fibonacci_lattice(n)

    most_equal(x_np, x.numpy())
    most_equal(y_np, y.numpy())

    if dr.backend_v(mi.Float) != dr.JitBackend.Metal:
        x_emulated, y_emulated = _fibonacci_lattice_f32(n)
        most_equal(x_emulated.numpy(), x.numpy())
        most_equal(y_emulated.numpy(), y.numpy())


def test_fibonacci_lattice_single_point():
    point = fibonacci_lattice(1)

    assert dr.allclose(point, mi.Point2f(0.5, 0.5))


@pytest.mark.parametrize("n", [0, -1])
def test_fibonacci_lattice_rejects_nonpositive_size(n):
    with pytest.raises(ValueError, match="greater than or equal to one"):
        fibonacci_lattice(n)


@pytest.mark.parametrize("n", [1.5, True])
def test_fibonacci_lattice_rejects_noninteger_size(n):
    with pytest.raises(TypeError, match="must be an integer"):
        fibonacci_lattice(n)



def test02_emulated_fibonacci_benchmark():
    if dr.backend_v(mi.Float) == dr.JitBackend.Metal:
        pytest.skip("Only the emulated backend is supported on Metal.")

    n = 1000000
    n_iters = 300

    def bench_ms(fn):
        # Warmup
        dr.eval(fn(n))
        dr.eval(fn(n))

        dr.sync_thread()
        t0 = time.time()

        for _ in range(n_iters):
            dr.eval(fn(n))

        dr.sync_thread()
        t1 = time.time()
        return 1000 * (t1 - t0) / n_iters

    t_np = bench_ms(fibonacci_lattice_np)
    t_dr = bench_ms(fibonacci_lattice)
    t_emulated = bench_ms(_fibonacci_lattice_f32)

    print(f"Benchmarking Fibonacci lattice generation with {n} points and {n_iters} iterations")
    print(f"Numpy: {t_np:.3f} ms / iter")
    print(f"DrJit: {t_dr:.3f} ms / iter")
    print(f"Emulated: {t_emulated:.3f} ms / iter")
