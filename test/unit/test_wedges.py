#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

import drjit as dr
import mitsuba as mi
import numpy as np

from sionna.rt.utils.wedges import wedge_geometry


def test_swap_faces():
    """Swapping faces updates all face-dependent wedge data."""
    mesh = mi.Mesh("two-triangles", 4, 2)
    params = mi.traverse(mesh)
    params["vertex_positions"] = np.array(
        [[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., -1., 0.]],
        dtype=np.float32,
    ).ravel()
    # The shared edge is local edge 0 of the first face and local edge 1
    # of the second face.
    params["faces"] = [0, 1, 2, 3, 1, 0]
    params.update()
    mesh.build_directed_edges()

    wedge = wedge_geometry(mi.MeshPtr(mesh), mi.UInt(0), mi.UInt(0))
    old_o = mi.Point3f(wedge.o)
    old_e_hat = mi.Vector3f(wedge.e_hat)
    old_n0 = mi.Normal3f(wedge.n0)
    old_nn = mi.Normal3f(wedge.nn)
    old_local_edge = mi.UInt(wedge.local_edge)
    old_endpoint = old_o + wedge.length * old_e_hat

    wedge.swap_faces(mi.Bool(True))

    assert dr.all(wedge.prim0 == 1)
    assert dr.all(wedge.primn == 0)
    # `swap_faces` does not update `local_edge`.
    assert dr.all(wedge.local_edge == old_local_edge)
    assert dr.allclose(wedge.n0, old_nn)
    assert dr.allclose(wedge.nn, old_n0)
    assert dr.allclose(wedge.o, old_endpoint)
    assert dr.allclose(wedge.e_hat, -old_e_hat)

    # Swapping a second time must restore the original representation.
    wedge.swap_faces(mi.Bool(True))
    assert dr.all(wedge.prim0 == 0)
    assert dr.all(wedge.primn == 1)
    assert dr.all(wedge.local_edge == old_local_edge)
    assert dr.allclose(wedge.n0, old_n0)
    assert dr.allclose(wedge.nn, old_nn)
    assert dr.allclose(wedge.o, old_o)
    assert dr.allclose(wedge.e_hat, old_e_hat)
