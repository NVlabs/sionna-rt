#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

import os
import tempfile

import mitsuba as mi
import numpy as np
import drjit as dr
import pytest
import sionna.rt as rt
from sionna.rt import load_scene, Transmitter, PlanarArray, ITURadioMaterial,\
    Receiver, PathSolver, RadioMapSolver, BackscatteringPattern
from sionna.rt.utils import dbm_to_watt, load_mesh, transform_mesh


####################################################
# Utilities
####################################################

def paths_to_coverage_map(paths, is_mesh=False):
    """
    Converts paths into the equivalent coverage map values.
    The coverage map is assumed to be a ssquare.
    """
    # [num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths]
    a_real, a_imag = paths.a
    a = a_real.numpy() + 1j*a_imag.numpy()

    # Transmit precoding
    # Assume default precoding
    # [num_rx, num_rx_ant, num_tx, num_paths]
    a /= np.sqrt(a.shape[3])
    a = np.sum(a, axis=3)

    # Sum energy of paths
    a = np.square(np.abs(a))
    # [num_rx, num_tx]
    a = np.sum(a, axis=(1, 3))

    # Swap dims
    # [num_tx, num_rx]
    a = a.T

    if not is_mesh:
        # Reshape to coverage map
        n = int(np.sqrt(a.shape[1]))
        shape = [a.shape[0], n, n]
        a = np.reshape(a, shape)

    return a

def default_array(num_rows: int = 1, num_cols: int = 1,
                  vertical_spacing: float = 0.5, horizontal_spacing: float = 0.5,
                  pattern: str = "iso", polarization: str = "V"):
    return PlanarArray(num_rows=num_rows,
                       num_cols=num_cols,
                       vertical_spacing=vertical_spacing,
                       horizontal_spacing=horizontal_spacing,
                       pattern=pattern,
                       polarization=polarization)

def validate_cm(los=False,
                specular_reflection=False,
                diffuse_reflection=False,
                refraction=False,
                diffraction=False,
                rm_center_z=1,
                tx_pattern="iso",
                rx_pattern="iso",
                tx_pol="V"):
    """Compares coverage map against exact path calculation for
    different propagation phenomena.
    """

    scene = load_scene(rt.scene.simple_reflector, merge_shapes=False)

    scene.get("reflector").radio_material = ITURadioMaterial("mat-concrete", "concrete", thickness=0.1)
    rm = scene.get("reflector").radio_material
    scene.get("reflector").radio_material.scattering_coefficient = mi.Float(np.sqrt(0.5))
    scene.get("reflector").radio_material.xpd_coefficient = mi.Float(0.3)

    scene.tx_array = default_array(pattern=tx_pattern, polarization=tx_pol)
    scene.rx_array = default_array(pattern=rx_pattern, polarization="VH")

    delta = 0.1
    width = 3
    tx = Transmitter(name="tx",
                     position=mi.Point3f(0, 0, .1),
                     orientation=mi.Point3f(0, 0, 0))
    scene.add(tx)

    rm_solver = RadioMapSolver()
    rm = rm_solver(scene,
                   max_depth=1,
                   cell_size=mi.Point2f(delta, delta),
                   center=mi.Point3f(0, 0, rm_center_z),
                   orientation=mi.Point3f(0., 0., 0.),
                   size=mi.Point2f(width, width),
                   samples_per_tx=int(1e7),
                   los=los,
                   specular_reflection=specular_reflection,
                   diffuse_reflection=diffuse_reflection,
                   refraction=refraction,
                   diffraction=diffraction,
                   edge_diffraction=diffraction,
                   diffraction_lit_region=diffraction)

    cell_centers = rm.cell_centers.numpy()
    cell_centers = np.reshape(cell_centers, [-1,3])
    for i, pos in enumerate(cell_centers):
        scene.add(Receiver(name=f"rx-{i}",
                           position=mi.Point3f(pos),
                           orientation=mi.Point3f(0., 0., 0.)))

    solver = PathSolver()
    paths = solver(scene,
                   samples_per_src=int(1e4),
                   max_num_paths_per_src=int(1e7),
                   max_depth=1,
                   los=los,
                   specular_reflection=specular_reflection,
                   diffuse_reflection=diffuse_reflection,
                   refraction=refraction,
                   diffraction=diffraction,
                   edge_diffraction=diffraction,
                   diffraction_lit_region=diffraction)

    rm_theo = paths_to_coverage_map(paths)[0]
    rm_rt = rm.path_gain.numpy()[0]

    err = np.where(rm_theo == 0.0,
                   0.0,
                   np.abs(rm_rt - rm_theo) / rm_theo)

    nmse_db = 10*np.log10(np.mean(np.abs(err)**2))
    return rm, rm_theo, nmse_db


#############################################################
# Tests
#############################################################

def test_random_positions():
    """test that random positions have a valid path loss and min/max
    distance is correctly set."""

    cell_size = mi.Point2f([4., 5.])
    batch_size = 100
    tx_pos = mi.Point3f(-210,73,105) # Top of Frauenkirche

    scene = load_scene(rt.scene.munich)

    tx = Transmitter(name="tx", position=tx_pos)
    scene.add(tx)

    scene.tx_array = default_array(num_rows=4, num_cols=4)
    scene.rx_array = default_array(num_rows=4, num_cols=4)

    # Position of the measurement plane
    radio_map_pos = dr.copy(scene.transmitters["tx"].position)
    radio_map_pos.z = 1.5

    ### Check with centering set to True

    # Generate the radio map
    rm_solver = RadioMapSolver()
    rm = rm_solver(scene,
                   max_depth=5,
                   center=radio_map_pos,
                   orientation=(0., 0., 0.),
                   size=(500., 500.),
                   cell_size=cell_size,
                   los=True,
                   specular_reflection=True,
                   diffuse_reflection=True,
                   refraction=True,
                   diffraction=True,
                   samples_per_tx=int(1e7))

    samples_pos, samples_cell_ind = rm.sample_positions(batch_size,
                                                        min_val_db=-110,
                                                        center_pos=True)
    samples_pos = samples_pos.numpy()
    cell_centers = rm.cell_centers.numpy()
    samples_pos = np.squeeze(samples_pos, axis=0) # Only a single transmitter
    samples_cell_ind = samples_cell_ind.numpy()
    samples_cell_ind = np.squeeze(samples_cell_ind, axis=0)
    # Check that the transmitter is always at the center of the cell
    for p, cell_ind in zip(samples_pos, samples_cell_ind):
        cell_center = cell_centers[cell_ind[0], cell_ind[1]]
        d = np.linalg.norm(p - cell_center)
        assert d == 0.

    ### Check with centering set to False

    # Generate the radio map
    rm_solver = RadioMapSolver()
    rm = rm_solver(scene,
                   max_depth=5,
                   center=radio_map_pos,
                   orientation=(0., 0., 0.),
                   size=(500., 500.),
                   cell_size=cell_size,
                   los=True,
                   specular_reflection=True,
                   diffuse_reflection=True,
                   refraction=True,
                   diffraction=True,
                   samples_per_tx=int(1e7))
    samples_pos, samples_cell_ind = rm.sample_positions(batch_size,
                                                        min_val_db=-110,
                                                        center_pos=False)
    samples_pos = samples_pos.numpy()
    cell_centers = rm.cell_centers.numpy()
    samples_pos = np.squeeze(samples_pos, axis=0) # Only a single transmitter
    samples_cell_ind = samples_cell_ind.numpy()
    samples_cell_ind = np.squeeze(samples_cell_ind, axis=0)
    # Check that the transmitter is always in the cell
    for p, cell_ind in zip(samples_pos, samples_cell_ind):
        cell_center = cell_centers[cell_ind[0], cell_ind[1]]
        d = np.abs(p - cell_center)
        assert (d[0] <= cell_size.x*0.5) and (d[1] <= cell_size.y*0.5)\
                and (d[2] == 0.0)

    ### Test min and max distance

    batch_size = 1000
    d_min = 150
    d_max = 300

    # max distance offset due to cell size quantization
    # dist can be off at most by factor 0.5 of diagonal
    d_cell = 0.5*dr.norm(cell_size).numpy()[0]
    low = d_min - d_cell
    high = d_max + d_cell

    samples_pos, _ = rm.sample_positions(batch_size,
                                        min_dist=d_min,
                                        max_dist=d_max,
                                        center_pos=False)
    samples_pos = samples_pos.numpy()
    samples_pos = np.squeeze(samples_pos, axis=0) # Only a single transmitter
    tx_pos = tx.position.numpy().T[0]
    for p in samples_pos:
        d = np.linalg.norm(p - tx_pos)
        assert (d > low) and (d < high)

    ### Test TX associations

    # Remove transmitter
    scene.remove("tx")

    # Add the first transmitter
    tx0 = Transmitter(name='tx0',
                      position=mi.Point3f(150, -100, 20),
                      orientation=mi.Point3f(0., 0., dr.pi*5/6),
                      power_dbm=44)
    scene.add(tx0)

    # Add the second transmitter
    tx1 = Transmitter(name='tx1',
                      position=mi.Point3f(-150, -100, 20),
                      orientation=mi.Point3f(0., 0., dr.pi/60),
                      power_dbm=44)
    scene.add(tx1)

    # Add the third transmitter
    tx2 = Transmitter(name='tx2',
                      position=mi.Point3f(0, 150 * dr.tan(dr.pi/3) - 100, 20),
                      orientation=mi.Point3f(0., 0., -dr.pi/2),
                      power_dbm=44)
    scene.add(tx2)

    # Compute radio map
    rm_solver = RadioMapSolver()
    rm = rm_solver(scene,
                   max_depth=5,
                   cell_size=cell_size,
                   los=True,
                   specular_reflection=True,
                   diffuse_reflection=True,
                   refraction=True,
                   diffraction=True,
                   samples_per_tx=int(1e7))

    metric = 'sinr'
    sm = getattr(rm, 'sinr').numpy()
    cell_to_tx_ideal = np.argmax(sm, axis=0)
    cell_to_tx_ideal[np.max(sm, axis=0)==0] = -1

    _, samples_cell_ind = rm.sample_positions(batch_size,
                                              center_pos=False,
                                              tx_association=True,
                                              metric=metric)
    samples_cell_ind = samples_cell_ind.numpy()

    for tx in range(sm.shape[0]):
        for cell_ind in samples_cell_ind[tx]:
            assert cell_to_tx_ideal[cell_ind[0], cell_ind[1]] == tx

def test_sinr_map():
    """Test SINR map"""

    cm_cell_size = np.array([4., 5.])

    scene = load_scene(rt.scene.munich)

    # Add the first transmitter
    tx0 = Transmitter(name='tx0',
                        position=mi.Point3f(150, -100, 20),
                        orientation=mi.Point3f(0., 0., dr.pi*5/6),
                        power_dbm=44)
    scene.add(tx0)

    # Add the second transmitter
    tx1 = Transmitter(name='tx1',
                    position=mi.Point3f(-150, -100, 20),
                    orientation=mi.Point3f(0., 0., dr.pi/60),
                    power_dbm=44)
    scene.add(tx1)

    # Add the third transmitter
    tx2 = Transmitter(name='tx2',
                    position=mi.Point3f(0, 150 * dr.tan(dr.pi/3) - 100, 20),
                    orientation=mi.Point3f(0., 0., -dr.pi/2),
                    power_dbm=44)
    scene.add(tx2)

    scene.tx_array = default_array(num_rows=4, num_cols=4)
    scene.rx_array = default_array(num_rows=4, num_cols=4)

    rx_pos = dr.copy(scene.transmitters["tx0"].position)
    rx_pos.z = 1.5

    # generate coverage map
    rm_solver = RadioMapSolver()
    rm = rm_solver(scene,
                   max_depth=5,
                   center=rx_pos,
                   orientation=mi.Point3f(0., 0., 0.),
                   size=mi.Point2f(500., 500.),
                   cell_size=cm_cell_size,
                   los=True,
                   specular_reflection=True,
                   diffuse_reflection=True,
                   refraction=True,
                   samples_per_tx=int(1e7))

    # retrieve path gain map
    path_gain = rm.path_gain.numpy()

    # retrieve transmit power
    tx_power = [dbm_to_watt(tx.power_dbm).numpy()[0]
                for tx in scene.transmitters.values()]

    # retrieve noise power
    n0 = scene.thermal_noise_power.numpy()[0]

    num_tx = rm.num_tx
    sinr_map_per_tx_ideal = np.zeros(path_gain.shape)

    # compute the SINR via Numpy operations
    for tx in range(num_tx):
        interf_mat = np.zeros(path_gain.shape[1:])
        for tx_interf in range(num_tx):
            # compute interference
            if tx_interf != tx:
                interf_mat += tx_power[tx_interf] * \
                    path_gain[tx_interf, ::]
        # SINR per tx,  assuming that all tiles connect to transmitter tx
        sinr_map_per_tx_ideal[tx, ::] = tx_power[tx] * \
            path_gain[tx, ::] / (interf_mat + n0)

    # Tile to tx association
    tile_to_tx_ideal = np.argmax(sinr_map_per_tx_ideal, axis=0)
    tile_to_tx_ideal[np.max(sinr_map_per_tx_ideal, axis=0) == 0] = -1

    # Get SINR map and RT association from RT
    sinr_map_per_tx = rm.sinr.numpy()
    tile_to_tx = rm.tx_association('sinr').numpy()

    err_sinr_map_per_tx = np.sort(np.abs((sinr_map_per_tx - sinr_map_per_tx_ideal)\
                                        /sinr_map_per_tx_ideal).flatten())
    err_sinr_map_per_tx = err_sinr_map_per_tx[np.isfinite(err_sinr_map_per_tx)]

    err_tile_to_tx = np.max(abs(tile_to_tx_ideal - tile_to_tx))

    # check that 90-th percentile SINR has small error
    assert err_sinr_map_per_tx[int(.99*len(err_sinr_map_per_tx))] < .01
    assert err_tile_to_tx == 0

def test_los():
    """Test that LoS pathgain map is close to exact path calculation"""
    _, _, nmse_db = validate_cm(los=True, tx_pol="V")
    assert nmse_db <  -20
    _, _, nmse_db = validate_cm(los=True, tx_pol="cross")
    assert nmse_db < -20

def test_specular_reflection():
    """Test that reflection pathgain map is close to exact path calculation"""
    _, _, nmse_db = validate_cm(specular_reflection=True, tx_pol="V")
    assert nmse_db <  -20
    _, _, nmse_db = validate_cm(specular_reflection=True, tx_pol="cross")
    assert nmse_db <  -20

def test_scattering():
    """Test that scattering pathgain map is close to exact path calculation"""
    _, _, nmse_db = validate_cm(diffuse_reflection=True, tx_pol="V")
    assert nmse_db <  -20
    _, _, nmse_db = validate_cm(diffuse_reflection=True, tx_pol="V")
    assert nmse_db <  -20

def test_refraction():
    """Test that refraction pathgain map is close to exact path calculation"""
    _, _, nmse_db = validate_cm(refraction=True, tx_pol="V", rm_center_z=-1)
    assert nmse_db <  -20
    _, _, nmse_db = validate_cm(refraction=True, tx_pol="cross", rm_center_z=-1)
    assert nmse_db <  -20

def test_edge_diffraction():
    """Test that diffraction pathgain map is close to exact path calculation"""
    _, _, nmse_db = validate_cm(diffraction=True, tx_pol="V", rm_center_z=-1)
    assert nmse_db <  -20
    _, _, nmse_db = validate_cm(diffraction=True, tx_pol="cross", rm_center_z=-1)
    assert nmse_db <  -20

def test_wedge_diffraction():
    """Test that diffraction pathgain map is close to exact path calculation
    for a wedge"""

    scene = load_scene(rt.scene.simple_wedge)

    scene.tx_array = PlanarArray(num_rows=1,
                num_cols=1,
                vertical_spacing=0.5,
                horizontal_spacing=0.5,
                pattern="iso",
                polarization="V")
    scene.rx_array = PlanarArray(num_rows=1,
                                num_cols=1,
                                vertical_spacing=0.5,
                                horizontal_spacing=0.5,
                                pattern="iso",
                                polarization="VH")

    # Unique transmitter
    tx = Transmitter(name="tx",
                    position=[1.0, 1.0, 0.0],
                    orientation=[0,0,0])
    scene.add(tx)

    # Compute the diffracted field energy using the radio map
    rm = RadioMapSolver()(scene=scene,
                        center=mi.Point3f(-1., -2.5, -10.),
                        orientation=mi.Point3f(0., 0., 0.),
                        size=mi.Point2f(1., 1.),
                        cell_size=mi.Point2f(0.1, 0.1),
                        samples_per_tx=10**8,
                        los=False,
                        specular_reflection=False,
                        diffuse_reflection=False,
                        refraction=False,
                        diffraction=True,
                        diffraction_lit_region=True)
    rm_np = rm.path_gain.numpy()[0]

    # Compute the diffracted field using the paths solver
    cell_centers = np.reshape(rm.cell_centers.numpy(), [-1,3])
    for i, rx_pos in enumerate(cell_centers):
        scene.add(Receiver(name=f"rx-{i}",
                        position=rx_pos,
                        orientation=mi.Point3f(0., 0., 0.)))
    paths = PathSolver()(scene,
                        samples_per_src=10**5,
                        max_num_paths_per_src=1000,
                        max_depth=1,
                        los=False,
                        specular_reflection=False,
                        diffuse_reflection=False,
                        refraction=False,
                        diffraction=True)
    a, _ = paths.cir(out_type="numpy")
    paths_rm = np.sum(np.square(np.abs(a)), axis=1)
    paths_rm = np.squeeze(paths_rm)
    paths_rm = np.reshape(paths_rm, [10, 10])

    err = np.abs(rm_np - paths_rm) / paths_rm
    err_db = 10.*np.log10(err)
    assert np.max(err_db) < -10

def test_diffraction_street_canyon():
    """Test that diffraction pathgain map is close to exact path calculation
    for the simple_street_canyon scene"""

    scene = load_scene(rt.scene.simple_street_canyon)

    tx = Transmitter(name="tx",
                     position=[-45, 11.8, 35],
                     orientation=[0, 0, 0])
    scene.add(tx)

    scene.tx_array = PlanarArray(num_rows=1,
                                 num_cols=1,
                                 vertical_spacing=0.5,
                                 horizontal_spacing=0.5,
                                 pattern="iso",
                                 polarization="V")

    rmap = RadioMapSolver()(scene=scene,
                            samples_per_tx=10**9,
                            cell_size=(0.5, 0.5),
                            los=False,
                            specular_reflection=False,
                            refraction=False,
                            diffraction=True,
                            diffraction_lit_region=True)

    rx_indices = [(120, 120), (170, 50), (160, 60), (120, 200), (220, 100),
                  (175, 350)]
    for i, p in enumerate(rx_indices):
        rx_pos = rmap.cell_centers[p[0], p[1]]
        scene.add(
            Receiver(name=f"rx-{i}", position=rx_pos.numpy(),
                     display_radius=3.0)
        )

    scene.rx_array = PlanarArray(num_rows=1,
                                 num_cols=1,
                                 vertical_spacing=0.5,
                                 horizontal_spacing=0.5,
                                 pattern="iso",
                                 polarization="VH")

    paths = PathSolver()(scene=scene,
                         los=False,
                         samples_per_src=10**6,
                         specular_reflection=False,
                         refraction=False,
                         diffraction=True,
                         diffraction_lit_region=True)

    a, _= paths.cir(out_type="numpy")
    a = np.squeeze(a)
    paths_en = np.sum(np.square(np.abs(a)), axis=(1,2))

    for i, ind in enumerate(rx_indices):
        g_rm = rmap.path_gain.numpy()[0, ind[0], ind[1]]
        g_paths = paths_en[i]
        err = np.abs(g_rm - g_paths)/g_paths
        err_db = 10.*np.log10(err)
        assert np.max(err_db) < -5

def test_box_01():
    """Test that field scattered and reflected fields jointly match for
    max_depth=1 in the box scene.
    This test also applies orientation, directive antenna patterns, as well as
    depolariation during scattering.
    """
    los = False
    specular_reflection = True
    diffuse_reflection = True
    refraction = False
    max_depth = 1 # Test only works for max_depth=1
    delta = 1

    scene = load_scene(rt.scene.box, merge_shapes=False)
    scene.objects["box"].radio_material = ITURadioMaterial("concrete", "concrete", 0.1)
    scene.objects["box"].radio_material.scattering_coefficient = dr.sqrt(0.5)
    scene.objects["box"].radio_material.scattering_pattern = \
        BackscatteringPattern(alpha_r=30, alpha_i=10, lambda_=0.5)

    scene.tx_array = default_array(pattern="tr38901")
    scene.rx_array = default_array(polarization="VH")

    scene.add(Transmitter(name="tx",
                          position=mi.Point3f(1.1, 0.8, 2),
                          orientation=mi.Point3f(0,0,0)))
    scene.get("tx").look_at(mi.Point3f(5,5,5))

    rm_solver = RadioMapSolver()
    rm = rm_solver(scene,
                   max_depth=max_depth,
                   cell_size=mi.Point2f(delta, delta),
                   samples_per_tx=int(1e8),
                   los=los,
                   specular_reflection=specular_reflection,
                   diffuse_reflection=diffuse_reflection,
                   refraction=refraction)

    for i, pos in enumerate(np.reshape(rm.cell_centers, [-1,3])):
        scene.add(Receiver(name=f"rx-{i}",
                           position=pos,
                           orientation=mi.Point3f(0, 0, 0)))

    solver = PathSolver()
    paths = solver(scene,
                   samples_per_src = 10000,
                   max_num_paths_per_src=int(1e7),
                   max_depth=max_depth,
                   los=los,
                   specular_reflection=specular_reflection,
                   diffuse_reflection=diffuse_reflection,
                   refraction=refraction,
                   diffraction=False)

    a = paths_to_coverage_map(paths)[0]
    nmse_db = 10 * np.log10(np.mean( ((rm.path_gain[0] - a) / a) ** 2 ))
    assert nmse_db < -20

def test_box_02():
    """Test a multiple reflections and LoS in the box scene.
    It includes directive antenna pattern and a complex material"""
    scene = load_scene(rt.scene.box, merge_shapes=False)
    scene.objects["box"].radio_material = ITURadioMaterial("concrete", "concrete", 0.1)
    scene.objects["box"].radio_material.scattering_coefficient = 0.2


    scene.tx_array = default_array(pattern="tr38901")
    scene.rx_array = default_array(polarization="VH")

    los = True
    specular_reflection = True
    diffuse_reflection = True
    refraction = False
    width = 9
    num_cells_x = 20
    delta = width / num_cells_x
    max_depth = 5

    scene.add(Transmitter(name="tx",
                          position=mi.Point3f(-3, -0.3, 4.),
                          orientation=mi.Point3f(0,0,0)))

    rm_solver = RadioMapSolver()
    rm = rm_solver(scene,
                   max_depth=max_depth,
                   cell_size=mi.Point2f(delta, delta),
                   center=mi.Point3f(0.0,0.0,0.5),
                   orientation=mi.Point3f(0.,0.,0),
                   size=mi.Point2f(width, width),
                   samples_per_tx=int(1e7),
                   los=los,
                   specular_reflection=specular_reflection,
                   diffuse_reflection=diffuse_reflection,
                   refraction=refraction)


    for i, pos in enumerate(np.reshape(rm.cell_centers, [-1,3])):
        scene.add(Receiver(name=f"rx-{i}",
                           position=pos,
                           orientation=mi.Point3f(0, 0, 0)))

    solver = PathSolver()
    paths = solver(scene,
                   samples_per_src = int(1e4),
                   max_num_paths_per_src=int(2e7),
                   max_depth=max_depth,
                   los=los,
                   specular_reflection=specular_reflection,
                   diffuse_reflection=diffuse_reflection,
                   refraction=refraction,
                   diffraction=False)

    a = paths_to_coverage_map(paths)[0]
    nmse_db = 10*np.log10(np.mean( ((rm.path_gain[0]-a)/a)**2 ))
    assert nmse_db < -20.

@pytest.mark.parametrize("los", [True, False])
def test_mesh_radio_map(los):

    specular_reflection = True
    diffuse_reflection = True
    refraction=False
    max_depth = 2

    # Setup the scene
    scene = load_scene(rt.scene.box, merge_shapes=False)
    scene.objects["box"].radio_material = ITURadioMaterial("concrete", "concrete", 0.1)
    scene.objects["box"].radio_material.scattering_coefficient = dr.sqrt(0.5)
    scene.objects["box"].radio_material.scattering_pattern =\
        BackscatteringPattern(alpha_r=30, alpha_i=10, lambda_=0.5)

    scene.tx_array = default_array()
    scene.rx_array = default_array(polarization="VH")

    scene.add(Transmitter(name="tx",
                position=mi.Point3f(4, 3, 4.0),
                orientation=mi.Point3f(0,0,0),
                display_radius=0.1))
    scene.get("tx").look_at(mi.Point3f(0,0,0))


    # Load the measurement surface
    fname = os.path.join(os.path.dirname(__file__),
                         "../data/subdivided_cube.ply")
    ms = load_mesh(fname)
    transform_mesh(ms,
                   translation=mi.Point3f(0, 0, 2.5),
                   scale=mi.Point3f(1.5, 1.5, 1))

    # Compute the radio map
    rm_solver = RadioMapSolver()
    rm_solver.loop_mode = "evaluated"
    rm = rm_solver(scene,
                measurement_surface=ms,
                samples_per_tx=int(1e7),
                max_depth=max_depth,
                los=los,
                specular_reflection=specular_reflection,
                diffuse_reflection=diffuse_reflection,
                refraction=refraction)

    # Check for NaN or Inf
    assert not (dr.any(dr.isinf(rm.path_gain)) or dr.any(dr.isnan(rm.path_gain)))

    # Add receivers at the cell centers
    for i, pos in enumerate(rm.cell_centers.numpy().T):
        scene.remove(f"rx-{i}")
        scene.add(Receiver(name=f"rx-{i}",
                            position=pos,
                            display_radius=0.1,
                            orientation=mi.Point3f(0, 0, 0)))

    # Compute radio map using the path solver
    solver = PathSolver()
    paths = solver(scene,
                    samples_per_src = 10000,
                    max_num_paths_per_src=int(1e7),
                    max_depth=max_depth,
                    los=los,
                    specular_reflection=specular_reflection,
                    diffuse_reflection=diffuse_reflection,
                    refraction=refraction,
                    diffraction=False)
    a = paths_to_coverage_map(paths, is_mesh=True)[0]

    nmse_db = 10*np.log10(np.mean( ((rm.path_gain[0]-a)/a)**2 ))
    assert nmse_db < -20

@pytest.mark.parametrize("color_map", [None, np.random.rand(30, 3)])
def test_show_association(color_map):
    scene = load_scene(rt.scene.box_two_screens)
    scene.tx_array = default_array()
    scene.rx_array = default_array(polarization="VH")
    bbox = scene.mi_scene.bbox()
    scene_size = bbox.extents()

    # Many transmitters to make sure that we are not limited by the number of
    # colors in the colormap.
    safe_range = 0.8
    rng = np.random.default_rng(seed=3445)
    for tx_i in range(20):
        p = mi.Point3f(rng.uniform(safe_range * bbox.min.x, safe_range * bbox.max.x),
                       rng.uniform(safe_range * bbox.min.y, safe_range * bbox.max.y),
                       safe_range * bbox.max.z)
        tx = Transmitter(name=f"tx-{tx_i}", position=p)
        scene.add(tx)

    rm_solver = RadioMapSolver()
    rm = rm_solver(scene,
                   cell_size=mi.Point2f(0.1, 0.1),
                   center=mi.Point3f(0, 0, 0.2 * bbox.center().z),
                   orientation=mi.Point3f(0., 0., 0.),
                   size=mi.Point2f(scene_size.x, scene_size.y),
                   samples_per_tx=int(1e4),
                   max_depth=5)

    if color_map is None:
        suffix = "default"
        # Check error handling if given an insufficiently large color map
        with pytest.raises(ValueError, match=r"The color map has 8 entries.*"):
            rm.show_association(color_map="Dark2")
    else:
        suffix = "custom"

    fig = rm.show_association(color_map=color_map)
    fig.tight_layout()

    # For visual inspection
    if False:
        fname = os.path.join(tempfile.gettempdir(),
                             f"test_show_association_{suffix}.png")
        fig.savefig(fname)
        print(f"Saved figure to: {fname}")

    # For visual inspection
    if False:
        fname = os.path.join(tempfile.gettempdir(),
                             f"test_show_association_preview_{suffix}.png")
        bbox = scene.mi_scene.bbox()
        to_world = mi.ScalarTransform4f().look_at(
            origin=mi.ScalarVector3f(2, 2, 3) * bbox.max,
            target=mi.ScalarVector3f(1, 1, 0) * bbox.center(),
            up=[0, 0, 1],
        )
        scene.render_to_file(camera=to_world, filename=fname, radio_map=rm,
                             clip_at=0.5 * bbox.max.z)
        print(f"Saved rendering to: {fname}")
