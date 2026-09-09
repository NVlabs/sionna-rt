from __future__ import annotations

import time

import drjit as dr
import matplotlib.pyplot as plt

import mitsuba as mi
import numpy as np
import pytest

import sionna.rt as rt
from sionna.rt.path_solvers import (
    SBCandidateGenerator,
    SBDeterministicCandidateGenerator,
)
from sionna.rt.utils import spawn_ray_from_sources, fibonacci_lattice


def get_paths_summary(paths: rt.PathsBuffer):
    return {
        "paths_counter": paths.paths_counter,
        "primitives": paths.primitives,
    }


def get_example_sb_setup(use_diffuse: bool, use_diffraction: bool):
    scene = rt.load_scene(rt.scene.munich)
    sources = mi.Point3f([-3.0, -3.5], [1.0, -1.0], [2.5, 2.5])
    targets = mi.Point3f(
        [3.0, 0.0, 3.0, 10.0], [1.0, 0.0, -1.0, 10.0], [2.5, 1.5, 2.5, 10.0]
    )

    max_depth = 8
    samples_per_src = int(1e6)
    max_num_paths = int(1e6)

    sb_args = {
        "mi_scene": scene.mi_scene,
        "src_positions": sources,
        "tgt_positions": targets,
        "samples_per_src": samples_per_src,
        "max_num_paths_per_src": max_num_paths,
        "max_depth": max_depth,
        "los": True,
        "specular_reflection": True,
        "refraction": True,
        "diffuse_reflection": use_diffuse,
        "diffraction": use_diffraction,
        "edge_diffraction": use_diffraction,
        "seed": 123,
    }

    return scene, sb_args


@pytest.mark.parametrize("use_diffuse", [False, True])
@pytest.mark.parametrize("use_diffraction", [False, True])
def test01_determinism_check(use_diffuse, use_diffraction):
    n_runs = 10
    scene, sb_args = get_example_sb_setup(use_diffuse, use_diffraction)

    reference_count = None
    timings: dict[bool, list[float]] = {False: [], True: []}
    for deterministic in (False, True):
        path_counts = set[int]()
        if deterministic:
            sb = SBDeterministicCandidateGenerator()
        else:
            sb = SBCandidateGenerator()

        # print(f"---------- {deterministic=} ----------")
        for run_i in range(n_runs):
            start = time.time()
            paths: rt.PathsBuffer = sb(**sb_args)
            paths.schedule()
            dr.eval()
            dr.sync_thread()
            end = time.time()
            timings[deterministic].append(end - start)

            paths.shrink()
            summary = get_paths_summary(paths)
            print(f"{summary['paths_counter']=}")
            path_counts.add(summary["paths_counter"])

        if deterministic:
            assert (
                len(path_counts) == 1
            ), "Path counts should be the same for deterministic runs"
            assert reference_count is not None
            diff = next(iter(path_counts)) - reference_count
            assert (
                np.abs(diff) / reference_count
            ) < 0.02, f"Path counts are expected to be close to the reference count, but got a difference of {diff} paths"

        else:
            assert (
                len(path_counts) > 1
            ), "Path counts are expected to be different for non-deterministic runs"
            reference_count = np.mean(list(path_counts))

    print(
        "Timings:\n"
        f"- Original: {1e3 * np.mean(timings[False]):.3f} ms\n"
        f"- Deterministic: {1e3 * np.mean(timings[True]):.3f} ms"
    )


def test02_atomic_min_reduction():
    min_indices = dr.full(mi.UInt32, 0xFFFFFFFF, 10)
    thread_hash = mi.UInt32([1, 1, 1, 0, 3, 3, 4, 5, 1, 3, 8, 9, 9])
    output_buffer = dr.zeros_like(thread_hash)
    dr.eval(min_indices, thread_hash, output_buffer)

    with dr.scoped_set_flag(dr.JitFlag.KernelHistory, True):
        thread_idx = dr.arange(mi.UInt32, dr.width(thread_hash))
        dr.scatter_reduce(
            dr.ReduceOp.Min, target=min_indices, index=thread_hash, value=thread_idx
        )

        dr.eval(min_indices)
        race_won = dr.gather(mi.UInt32, min_indices, thread_hash) == thread_idx

        # We may have false positives, if a new min thread arrives later.
        dr.scatter(output_buffer, 1, thread_idx, active=race_won)
        dr.eval(output_buffer)

        history = dr.kernel_history([dr.KernelType.JIT])

    assert dr.all(output_buffer == [1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0])
    assert len(history) == 2


def test03_kernel_count():
    n_runs = 10
    scene, sb_args = get_example_sb_setup(use_diffuse=True, use_diffraction=True)
    sb = SBDeterministicCandidateGenerator()

    # Same seed should yield the same set of kernels.
    for change_seed in (False, True):
        seen_kernels = set[str]()
        with dr.scoped_set_flag(dr.JitFlag.KernelHistory, True):
            for run_i in range(n_runs):
                if change_seed:
                    sb_args["seed"] = run_i + 1
                paths: rt.PathsBuffer = sb(**sb_args)
                paths.schedule()
                dr.eval()

                history = dr.kernel_history()
                for kernel in history:
                    if "hash" in kernel:
                        seen_kernels.add(kernel["hash"])

        print(
            f"With {'varying' if change_seed else 'identical'} seed: saw {len(seen_kernels)} unique kernels"
        )
        # Note: this particular kernel count may change with implementation changes,
        # but we expect it to be the same when keeping or changing the seed.
        assert len(seen_kernels) == 16


def test04_thread_index_computation():
    samples_per_src = 3919

    for max_depth in (1, 2, 5):
        for n_sources in (1, 2, 5):
            for num_targets in (1, 2, 5):
                num_samples = samples_per_src * n_sources
                expected_total = max_depth * num_samples * num_targets

                ref = dr.tile(
                    dr.repeat(dr.arange(mi.UInt32, num_samples), num_targets),
                    reps=max_depth,
                )
                assert dr.width(ref) == expected_total

                lane = dr.arange(mi.UInt32, max_depth * num_samples * num_targets)
                simpler = (lane // mi.UInt32(num_targets)) % mi.UInt32(num_samples)
                assert dr.width(simpler) == expected_total

                assert dr.all(ref == simpler)

@pytest.mark.skip(reason="Only needed for visualization purposes")
def test05_primary_ray_ordering():
    _, sb_args = get_example_sb_setup(use_diffuse=True, use_diffraction=True)
    # Full solvers use sb_args["samples_per_src"] (~1e6); keep a modest count for the plot.
    n_plot = min(1000, int(sb_args["samples_per_src"]))
    ray = spawn_ray_from_sources(fibonacci_lattice, n_plot, mi.Point3f([1, 2, 3]))
    dr.eval(ray.d)

    # Layout is [source0 × n_plot, source1 × n_plot, …]; directions match per source.
    d = np.transpose(ray.d.numpy(), (1, 0))
    assert d.shape == (n_plot, 3)
    idx = np.arange(d.shape[0], dtype=np.float64)

    assert np.allclose(np.linalg.norm(d, axis=1), 1.0, atol=1e-5)

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")

    u = np.linspace(0, 2 * np.pi, 48)
    v = np.linspace(0, np.pi, 24)
    uu, vv = np.meshgrid(u, v)
    xs = np.sin(vv) * np.cos(uu)
    ys = np.sin(vv) * np.sin(uu)
    zs = np.cos(vv)
    ax.plot_surface(
        xs,
        ys,
        zs,
        rstride=2,
        cstride=2,
        color="0.85",
        alpha=0.12,
        linewidth=0.2,
        edgecolor="0.6",
    )

    sc = ax.scatter(
        d[:, 0],
        d[:, 1],
        d[:, 2],
        c=idx,
        cmap="turbo",
        s=6,
        depthshade=False,
    )
    fig.colorbar(sc, ax=ax, shrink=0.55, label="Ray index")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("Primary ray directions (Fibonacci lattice)")
    ax.set_box_aspect((1, 1, 1))
    plt.tight_layout()

    plt.show()
    plt.close(fig)
