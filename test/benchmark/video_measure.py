#!/bin/env python3

import drjit as dr
import mitsuba as mi
import numpy as np

import sys

sys.path.append("../src")
import sionna
from sionna.rt import (
    load_scene,
    PlanarArray,
    Transmitter,
    ITURadioMaterial,
    SceneObject,
    Receiver,
    Paths,
    PathSolver,
    RadioMapSolver,
    subcarrier_frequencies,
)
import time
import argparse
import pickle

import nvtx


###########################################
# Arguments
###########################################

parser = argparse.ArgumentParser(description="Extract RT parameters")
parser.add_argument("eval", type=str, choices=["path", "rm"], help="'path' | 'rm'")
parser.add_argument("--diffraction", action="store_true", help="Enable diffraction.")
parser.add_argument(
    "--deterministic",
    action="store_true",
    help="Use deterministic path solver (only valid for `eval=path`).",
)
parser.add_argument(
    "--verify",
    action="store_true",
    help="Verify results against previously-computed reference results.",
)
args = parser.parse_args()
use_diffraction = args.diffraction
eval_target = args.eval
verify = args.verify
use_deterministic = args.deterministic

if use_deterministic:
    assert (
        eval_target == "path"
    ), "Deterministic path solver is only valid for `eval=path`."

samples_per_tx = int(1e8)


###########################################
# Correctness checking
###########################################

reference_fname = f"{eval_target}-reference.pkl"
if verify:
    print("[!] Results checking mode: do not use for actual timing measurements!")


def compare_paths_cfr(expected, actual):
    # Compute frequencies of subcarriers relative to the carrier frequency
    frequencies = subcarrier_frequencies(num_subcarriers=1024, subcarrier_spacing=30e3)

    # Compute channel frequency response
    ref_real, ref_imag = expected.cfr(frequencies=frequencies, normalize=True)
    actual_real, actual_imag = actual.cfr(frequencies=frequencies, normalize=True)

    assert dr.allclose(
        ref_real, actual_real, atol=1e-2, rtol=1e-3
    ), f"Real part max diff: {dr.max(dr.abs(ref_real - actual_real))}"
    assert dr.allclose(
        ref_imag, actual_imag, atol=1e-2, rtol=1e-3
    ), f"Imaginary part max diff: {dr.max(dr.abs(ref_imag - actual_imag))}"


def compare_structs(name, expected, actual):
    for k in expected.DRJIT_STRUCT:
        expected_v = getattr(expected, k)
        actual_v = getattr(actual, k)
        msg = (
            f"Results do not match reference paths for {name}.{k}.\n"
            f"Expected: {expected_v}\n"
            f"Actual: {actual_v}"
        )
        if isinstance(expected_v, np.ndarray):
            assert np.allclose(expected_v, actual_v), msg
        elif hasattr(expected_v, "DRJIT_STRUCT"):
            compare_structs(f"{name}.{k}", expected_v, actual_v)
        else:
            assert expected_v == actual_v, msg


###########################################
# Config
###########################################

frame_rate = 40
num_points = frame_rate * 3

num_ant = 32

# Paths
car_start_point = np.array([-86, -205, 1.5])
car_end_point = np.array([-163, -230, 1.5])
car_dir = car_end_point - car_start_point
car_dist = np.linalg.norm(car_dir)
car_dir /= car_dist
car_positions = np.linspace(0.0, car_dist, frame_rate * 3)
car_positions = np.expand_dims(car_positions, axis=1)
car_positions = np.expand_dims(
    car_start_point, axis=0
) + car_positions * np.expand_dims(car_dir, axis=0)
car_look_at = car_positions + np.expand_dims(car_dir, axis=0)

# Radio matps
steering_angles = dr.linspace(mi.Float, -0.3 * np.pi, 0.3 * np.pi, frame_rate * 3)

###########################################
# Setup scene
############################################

# Load scene but exclude some objects from merging for editing demo
scene = load_scene(sionna.rt.scene.munich, merge_shapes=True)
scene.radio_materials["marble"].thickness = 0.05

# Radio material constituing the cars
# We use ITU metal, and use red color for visualization to
# make the cars easily discernible
car_material = ITURadioMaterial(
    "car-material", "metal", thickness=0.01, color=(0.8, 0.1, 0.1)
)

# Instantiate the car objects
car = SceneObject(
    fname=sionna.rt.scene.low_poly_car, name="car", radio_material=car_material
)
scene.edit(add=[car])

# Set antenna arrays
scene.tx_array = PlanarArray(
    num_rows=1, num_cols=num_ant, pattern="tr38901", polarization="V"
)
scene.rx_array = PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")

# Add transmitter
tx = Transmitter("tx", position=(-111, -191, 25))
scene.add(tx)
tx.look_at(mi.Point3f(-151.2, -214.3, 1.5))

# Add receivers on top of each car
rx_car = Receiver("rx-car", position=(0, 0, 0))
scene.add(rx_car)

##########################################
# Paths
##########################################

if eval_target == "path":

    print("Measure path compute runtime")

    # Compute paths
    p_solver = PathSolver(deterministic=use_deterministic)

    # Warm-up
    paths1 = p_solver(scene, max_depth=5)
    paths2 = p_solver(scene, max_depth=5)
    # TODO: eval all fields
    dr.eval(paths1.a, paths2.a)
    del paths1, paths2
    dr.sync_thread()

    def measure_paths(num_it=10):
        start = time.time()
        for _ in range(num_it):
            with nvtx.annotate("measure_paths"):
                paths: Paths = p_solver(
                    scene,
                    max_depth=5,
                    diffraction=use_diffraction,
                )
                dr.eval(paths)

            if verify:  # # TODO: remove this
                # with dr.scoped_set_flag(dr.JitFlag.KernelFreezing, False):
                ref_paths = p_solver(
                    scene,
                    max_depth=5,
                    diffraction=use_diffraction,
                )
                dr.eval(ref_paths)
                compare_paths_cfr(ref_paths, paths)

        dr.sync_thread()
        end = time.time()
        elapsed_ms = 1e3 * (end - start) / num_it
        return elapsed_ms, paths

    runtimes = []
    all_paths = []
    for i, (p, a) in enumerate(zip(car_positions, car_look_at)):

        # Move car
        car.position = mi.Point3f(p)
        car.look_at(mi.Point3f(a))
        # Move receiver
        rx_car.position = mi.Point3f(p)
        rx_car.position.z += 1.5
        # TODO: remove this?
        dr.make_opaque(rx_car.position, car.position, car.orientation)

        # Measure
        t, res = measure_paths()
        runtimes.append(t)
        all_paths.append(res)
        extra = "Verification mode | " if verify else ""
        print(f"\t{i}/{num_points} | {extra}Runtime [ms]: {t:.3f}", end="\r")

    print()

    # Save runtimes for later plotting.
    if not verify:
        with open("path-runtime.pkl", "wb") as f:
            pickle.dump(runtimes, f)
else:
    car.position = mi.Point3f(car_positions[-1])
    car.look_at(mi.Point3f(car_look_at[-1]))


###############################################
# Radio maps
###############################################

if eval_target == "rm":
    assert not verify, "Not implemented yet."

    print("Measure radio map compute runtime")

    rm_solver = RadioMapSolver()
    scene.radio_materials["marble"].thickness = 1.0

    ns = dr.linspace(mi.Float, 0.0, num_ant, num_ant, endpoint=False)
    nf = dr.rsqrt(num_ant)

    # Warm-up
    theta = steering_angles[0]
    dphi = ns * dr.pi * dr.sin(theta)
    precoding_vec = (dr.cos(dphi) * nf, dr.sin(dphi) * nf)
    rm = rm_solver(
        scene,
        max_depth=5,
        cell_size=(0.5, 0.5),
        samples_per_tx=samples_per_tx,
        center=(-133, -200, 1.5),
        orientation=(0, 0, 0),
        size=(300, 300),
        precoding_vec=precoding_vec,
        diffraction=use_diffraction,
    )
    pg1 = rm.path_gain
    rm = rm_solver(
        scene,
        max_depth=5,
        cell_size=(0.5, 0.5),
        samples_per_tx=samples_per_tx,
        center=(-133, -200, 1.5),
        orientation=(0, 0, 0),
        size=(300, 300),
        precoding_vec=precoding_vec,
        diffraction=use_diffraction,
    )
    pg2 = rm.path_gain
    dr.eval(pg1, pg2)
    del pg1, pg2, rm, dphi, precoding_vec
    dr.sync_thread()

    def measure_radio_map(theta, num_it=20):
        dphi = ns * dr.pi * dr.sin(theta)
        precoding_vec = (dr.cos(dphi) * nf, dr.sin(dphi) * nf)
        dr.make_opaque(precoding_vec)
        start = time.time()
        for _ in range(num_it):
            rm = rm_solver(
                scene,
                max_depth=5,
                cell_size=(0.5, 0.5),
                samples_per_tx=samples_per_tx,
                center=(-133, -200, 1.5),
                orientation=(0, 0, 0),
                size=(300, 300),
                precoding_vec=precoding_vec,
                diffraction=use_diffraction,
            )
            dr.eval(rm.path_gain)
        dr.sync_thread()
        end = time.time()
        elapsed_ms = 1e3 * (end - start) / num_it
        return elapsed_ms

    runtimes = []
    for i, theta in enumerate(steering_angles):
        # Measure
        t = measure_radio_map(theta)
        runtimes.append(t)
        print(f"\t{i}/{len(steering_angles)} | Runtime [ms]: {t:.3f}", end="\r")

    print()

    with open("rm-runtime.pkl", "wb") as f:
        pickle.dump(runtimes, f)

print("Done.")
