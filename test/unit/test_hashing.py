import pytest
import mitsuba as mi
import numpy as np
import drjit as dr

from sionna.rt.utils.hashing import *

@pytest.mark.parametrize("op", ["round", "floor"])
def test_consistency(op):
    """
    Check that the encoding of planes is resistant to small perturbations
    around 0.
    """
    hash_fn = PlaneHasher(op=op)
    n1 = mi.Normal3f(0.2, 0.9, -1e-6)
    n2 = mi.Normal3f(0.2, 0.9, 1e-6)
    p1 = mi.Point3f(1e-6, -1e-6, 1e-6)
    p2 = mi.Point3f(1., 2., 3.)
    for p in [p1, p2]:
        assert hash_fn(n1, p) == hash_fn(n2, p)
        assert hash_fn(n1, p) == hash_fn(-n1, p)
    assert hash_fn(n1, p1) == hash_fn(n1, -p1)
    assert hash_fn(n1, p2) != hash_fn(n1, -p2)

    n1 = mi.Normal3f(0.2, 1e-6, 0.9)
    n2 = mi.Normal3f(0.2, -1e-6, 0.9)
    for p in [p1, p2]:
        assert hash_fn(n1, p) == hash_fn(n2, p)
        assert hash_fn(n1, p) == hash_fn(-n1, p)
    assert hash_fn(n1, p1) == hash_fn(n1, -p1)
    assert hash_fn(n1, p2) != hash_fn(n1, -p2)

    n1 = mi.Normal3f(1e-6, 0.2, 0.9)
    n2 = mi.Normal3f(-1e-6, 0.2, 0.9)
    for p in [p1, p2]:
        assert hash_fn(n1, p) == hash_fn(n2, p)
        assert hash_fn(n1, p) == hash_fn(-n1, p)
    assert hash_fn(n1, p1) == hash_fn(n1, -p1)
    assert hash_fn(n1, p2) != hash_fn(n1, -p2)

    n1 = mi.Normal3f(0.999, 1e-6, 1e-6)
    n2 = mi.Normal3f(0.999, -1e-6, 1e-6)
    for p in [p1, p2]:
        assert hash_fn(n1, p) == hash_fn(n2, p)
        assert hash_fn(n1, p) == hash_fn(-n1, p)
    assert hash_fn(n1, p1) == hash_fn(n1, -p1)
    assert hash_fn(n1, p2) != hash_fn(n1, -p2)

    # Planes passing through the origin
    n = mi.Normal3f(np.random.uniform(-1, 1, (3,100)))
    assert len(np.unique(hash_fn(n, p1).numpy())) == dr.width(n)
    assert dr.allclose(hash_fn(n, p1), hash_fn(n, -p1))


@pytest.mark.parametrize("op", ["round", "floor"])
def test_fibonacci(op):
    from sionna.rt.utils import fibonacci_lattice
    hash_fn = PlaneHasher(op=op)
    uv = fibonacci_lattice(10**7)
    pts = mi.warp.square_to_uniform_sphere(uv)

    hashes = hash_fn(pts, pts)
    assert len(np.unique(hashes)) == dr.width(pts)

@pytest.mark.parametrize("op", ["round", "floor"])
@pytest.mark.parametrize("n,m", [(10**5, 100), (10**6, 1000), (10**7, 10000)])
def test_chi_squared(n, m, op):
    r"""
    Check that the distribution of hashes is uniform.

    param n: Number of hashes to generate
    param m: Size of the hash table (number of bins)
    """
    from scipy.stats import chisquare
    hash_fn = PlaneHasher(op=op)
    sequence_hash = mi.UInt64(0)
    for i in range(10):
        np.random.seed(i)
        v = mi.Vector3f(np.random.uniform(-1000, 1000, (3, n)))
        p = mi.Vector3f(np.random.uniform(-1000, 1000, (3, n)))
        # Compute the hash
        inter_hash = hash_fn(v, p)
        sequence_hash = hash_fnv1a(inter_hash, h=sequence_hash)

        # Check that the hashes are uniformly distributed
        frequencies = np.bincount(sequence_hash % m, minlength=m)

        res = chisquare(frequencies)
        assert res.pvalue > 0.01, f"Test failed for i={i}, p-value={res.pvalue}"
