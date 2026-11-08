#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Benchmark of RadioMapSolver with several precoding vectors

Compares the computation of one radio map per precoding vector of a DFT
codebook by (i) calling the solver once per precoding vector, and (ii) a single
call with all precoding vectors.

The GPU is warmed up and the two approaches are timed in an interleaved
fashion, such that clock boosting does not bias the comparison. JIT
compilation is excluded from the timings.

Example:

    python radio_map_precoding.py --scene munich --samples-per-tx 1e9 \
        --num-vecs 8 --num-vecs 32
"""

import argparse
import statistics
import time

import drjit as dr
import mitsuba as mi
import numpy as np

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--scene", default="munich")
parser.add_argument("--samples-per-tx", type=float, action="append",
                    help="Can be given several times (default: 1e6 to 1e9)")
parser.add_argument("--num-vecs", type=int, action="append",
                    help="Can be given several times (default: 8)")
parser.add_argument("--num-ant", type=int, default=8,
                    help="Number of antennas of the ULA")
parser.add_argument("--polarization", default="V")
parser.add_argument("--cell-size", type=float, default=5.)
parser.add_argument("--max-depth", type=int, default=3)
parser.add_argument("--rounds", type=int, default=3)
parser.add_argument("--warmup", type=float, default=5.,
                    help="GPU warm-up duration [s]")
parser.add_argument("--cpu", action="store_true")
args = parser.parse_args()

mi.set_variant("llvm_ad_mono_polarized" if args.cpu
               else "cuda_ad_mono_polarized")

# pylint: disable=wrong-import-position
from sionna import rt
from sionna.rt import load_scene, PlanarArray, Transmitter, RadioMapSolver

scene = load_scene(getattr(rt.scene, args.scene), merge_shapes=True)
scene.tx_array = PlanarArray(num_rows=1, num_cols=args.num_ant,
                             pattern="iso", polarization=args.polarization)
bbox = scene.mi_scene.bbox()
center = 0.5 * (bbox.min + bbox.max)
scene.add(Transmitter("tx", position=mi.Point3f(center.x, center.y,
                                                bbox.max.z + 5.)))
num_tx_ant = scene.tx_array.num_ant
solver = RadioMapSolver()


def codebook(num_vecs):
    """DFT codebook steering over [-60, 60] deg

    :return: Real and imaginary components with shape
        `[num_vecs, 1, num_tx_ant]`
    """
    theta = np.deg2rad(np.linspace(-60., 60., num_vecs))
    phase = np.pi * np.arange(num_tx_ant)[None] * np.sin(theta)[:, None]
    real = np.cos(phase) / np.sqrt(num_tx_ant)
    imag = np.sin(phase) / np.sqrt(num_tx_ant)
    return (mi.TensorXf(real[:, None].astype(np.float32)),
            mi.TensorXf(imag[:, None].astype(np.float32)))


def run(mode, precoding_vecs, samples_per_tx, seed):
    kwargs = {"cell_size": mi.Point2f(args.cell_size, args.cell_size),
              "samples_per_tx": int(samples_per_tx),
              "max_depth": args.max_depth,
              "seed": seed}
    real, imag = precoding_vecs
    t0 = time.perf_counter()
    if mode == "separate":
        rms = [solver(scene, precoding_vec=(real[v], imag[v]), **kwargs)
               for v in range(dr.shape(real)[0])]
    else:
        rms = solver(scene, precoding_vec=precoding_vecs, **kwargs)
    # Return only the gains so each run releases its measurement surfaces.
    # Keeping those surfaces alive changes the shape IDs in later JIT kernels.
    path_gains = [rm.path_gain for rm in rms]
    for gain in path_gains:
        dr.eval(gain)
    dr.sync_thread()
    return time.perf_counter() - t0, path_gains


def time_pair(precoding_vecs, samples_per_tx, seed):
    """Time both modes, retrying the pair if either warms the JIT cache."""
    for _ in range(2):
        # The last two counters track disk-cache loads and new compilations.
        misses = dr.detail.launch_stats()[1:]
        t_sep = run("separate", precoding_vecs, samples_per_tx, seed)[0]
        t_bat = run("batched", precoding_vecs, samples_per_tx, seed)[0]
        if dr.detail.launch_stats()[1:] == misses:
            return t_sep, t_bat
    raise RuntimeError("Repeated timing pair still incurred JIT cache misses")


def main():
    """Runs the benchmark for all requested configurations"""
    print(f"scene={args.scene}, {num_tx_ant} antennas ({args.polarization}), "
          f"cell size={args.cell_size} m, max_depth={args.max_depth}")
    print(f"{'rays/tx':>8} {'N':>4} | {'separate [s]':>12} {'batched [s]':>12} "
          f"{'speedup':>8} | {'max rel. diff':>13}")
    for num_vecs in args.num_vecs or [8]:
        precoding_vecs = codebook(num_vecs)
        for samples_per_tx in args.samples_per_tx or [1e6, 1e7, 1e8, 1e9]:
            # JIT compilation and correctness check
            _, gains_sep = run("separate", precoding_vecs, samples_per_tx, 1)
            _, gains_bat = run("batched", precoding_vecs, samples_per_tx, 1)
            pg_sep = np.stack([gain.numpy() for gain in gains_sep])
            pg_bat = np.stack([gain.numpy() for gain in gains_bat])
            err = np.max(np.abs(pg_sep - pg_bat)) / np.max(pg_sep)
            # GPU warm-up
            t_end = time.perf_counter() + args.warmup
            while time.perf_counter() < t_end:
                run("separate", precoding_vecs, samples_per_tx, 2)
                run("batched", precoding_vecs, samples_per_tx, 2)
            separate_times, batched_times = [], []
            for r in range(args.rounds):
                t_sep, t_bat = time_pair(precoding_vecs, samples_per_tx, 10 + r)
                separate_times.append(t_sep)
                batched_times.append(t_bat)
            t_sep = statistics.median(separate_times)
            t_bat = statistics.median(batched_times)
            print(f"{samples_per_tx:8.0e} {num_vecs:4d} | {t_sep:12.3f} "
                  f"{t_bat:12.3f} {t_sep / t_bat:7.2f}x | {err:13.1e}",
                  flush=True)


if __name__ == "__main__":
    main()
