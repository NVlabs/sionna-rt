import drjit as dr
import mitsuba as mi
import numpy as np
import pytest

from sionna import rt
from sionna.rt import (
    load_scene,
    PathSolver,
    PlanarArray,
    RadioMapSolver,
    RadioMaterial,
    Receiver,
    Transmitter,
)
from sionna.rt.utils import sigmoid


OUTPUT_PLOTS = False


def example_scene(mat_name: str):
    scene = load_scene(rt.scene.simple_reflector, merge_shapes=False)

    scene.add(Transmitter("tx", position=(-2.0, 0.0, 1)))
    scene.add(Receiver("rx", position=(2.0, 0.0, 1)))

    scene.tx_array = PlanarArray(
        num_rows=1, num_cols=1, pattern="iso", polarization="V"
    )
    scene.rx_array = PlanarArray(
        num_rows=1, num_cols=1, pattern="iso", polarization="V"
    )

    # Instantiate the radio material
    my_mat = RadioMaterial(
        mat_name, thickness=0.1, relative_permittivity=5.0, conductivity=1
    )
    # Assign the radio material to the reflector
    scene.objects["reflector"].radio_material = my_mat
    # To avoid confusion, discard the radio material initially loaded with the scene
    scene.remove("reflector-mat")

    return scene


def logit_2_value(logit, max_value, min_value=0.0):
    return min_value + sigmoid(logit) * (max_value - min_value)


def normalized_absolute_error(x, y):
    """
    Normalized absolute error
    The `dr.detach()` function stops gradient from
    propagating through its input
    """
    return dr.abs(x - y) * dr.detach(dr.rcp(y))


def paths_gain_fn():
    """
    Returns a callable mapping a scene to a differentiable scalar gain
    computed from the coherent paths.
    """
    # Switch the computation of field loop to "evaluated" mode to
    # enable gradient backpropagation through the loop
    solver = PathSolver()
    solver.loop_mode = "evaluated"

    def gain(scene):
        paths = solver(scene, seed=1234)
        a_real, a_imag = paths.a
        return dr.sum(dr.square(a_real) + dr.square(a_imag))

    return gain


def radio_map_gain_fn():
    """
    Returns a callable mapping a scene to a differentiable scalar gain
    computed from a radio map.

    The line-of-sight component is disabled so that the total energy
    captured by the measurement plane only depends on the reflection off
    the material under calibration.
    """
    solver = RadioMapSolver()
    solver.loop_mode = "evaluated"

    def gain(scene):
        rm = solver(
            scene,
            max_depth=1,
            cell_size=mi.Point2f(0.25, 0.25),
            center=mi.Point3f(0, 0, 1),
            orientation=mi.Point3f(0, 0, 0),
            size=mi.Point2f(8, 8),
            samples_per_tx=int(1e6),
            los=False,
            specular_reflection=True,
            refraction=False,
            seed=1234,
        )
        return dr.sum(rm.path_gain)

    return gain


def plot_calibration(history, num_iterations, ref_value, label, fname):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.grid(True)
    ax.plot(np.arange(1, num_iterations + 1), history, label="Calibrated")
    ax.hlines(
        [ref_value],
        1,
        num_iterations + 1,
        color="k",
        label="Ground-truth",
    )
    ax.set_xlabel("Iteration")
    ax.set_ylabel(label)
    ax.legend()
    fig.tight_layout()
    fig.savefig(fname, dpi=150)
    print(f"Plot saved to: {fname}")


def calibrate_material(
    param,
    gain_fn,
    num_iterations,
    rtol,
    loss_threshold,
    plot_name,
):
    """
    Retrieves a radio material parameter through gradient descent.

    The ground-truth gain is computed by ``gain_fn`` for a reference scene,
    and the selected ``param`` of a trainable scene is optimized so that its
    gain matches the reference. All other material parameters are fixed to
    their ground-truth values.
    """
    params = {
        "conductivity": {
            "ref_value": 20.0,
            "label": "Conductivity [S/m]",
            "max_value": 100.0,
            "min_value": 0.0,
            "init_logit": -5.0,
        },
        "relative_permittivity": {
            "ref_value": 5.0,
            "label": "Relative permittivity",
            "max_value": 20.0,
            "min_value": 1.0,
            "init_logit": -2.0,
        },
    }
    param_cfg = params[param]

    def to_value(logit):
        return logit_2_value(
            logit,
            param_cfg["max_value"],
            param_cfg.get("min_value", 0.0),
        )

    # Ground-truth
    scene = example_scene("my-mat")
    ref_mat = scene.radio_materials["my-mat"]
    for name, cfg in params.items():
        setattr(ref_mat, name, cfg["ref_value"])
    ref_gain = dr.detach(gain_fn(scene))
    del scene

    # Trainable scene setup
    trainable_scene = example_scene("my-trainable-mat")
    trainable_mat = trainable_scene.radio_materials["my-trainable-mat"]

    # Fix non-optimized parameters to their ground-truth values
    for name, cfg in params.items():
        if name != param:
            setattr(trainable_mat, name, cfg["ref_value"])

    # Adam optimizer
    learning_rate = 4e-2
    opt = mi.ad.Adam(lr=learning_rate)

    # Trainable variable is the logit of the selected material parameter
    opt[f"logit_{param}"] = mi.Float(param_cfg["init_logit"])
    setattr(trainable_mat, param, to_value(opt[f"logit_{param}"]))

    if OUTPUT_PLOTS:
        history = []

    # Optimization loop
    loss = np.inf
    for _ in range(num_iterations):
        gain = gain_fn(trainable_scene)

        # Compute loss on total gain and the gradients
        loss = normalized_absolute_error(gain, ref_gain)
        dr.backward(loss)

        # Optimizer step
        opt.step()
        updated_value = to_value(opt[f"logit_{param}"])
        setattr(trainable_mat, param, updated_value)
        if OUTPUT_PLOTS:
            history.append(updated_value.numpy())

    assert dr.allclose(
        getattr(trainable_mat, param), param_cfg["ref_value"], rtol=rtol
    )
    assert loss.numpy().item() < loss_threshold

    if OUTPUT_PLOTS:
        plot_calibration(
            history,
            num_iterations,
            param_cfg["ref_value"],
            param_cfg["label"],
            plot_name,
        )


@pytest.mark.parametrize("param", ["conductivity", "relative_permittivity"])
def test01_paths(param):
    """
    Retrieves a material parameter from the coherent paths.

    Adapted from a script provided by @jeertmans.
    Source: https://github.com/NVlabs/sionna-rt/issues/66#issue-4204604024
    """
    num_iterations = 135 if param == "conductivity" else 75
    calibrate_material(
        param,
        gain_fn=paths_gain_fn(),
        num_iterations=num_iterations,
        rtol=0.002,
        loss_threshold=2e-4,
        plot_name=f"{param}_calibration_paths.png",
    )


@pytest.mark.parametrize("param", ["conductivity", "relative_permittivity"])
def test02_radio_map(param):
    """
    Retrieves a material parameter from a radio map.

    The radio map is a stochastic estimate of the received energy, so it
    exhibits a higher noise floor than the coherent paths and thus uses
    looser convergence tolerances.
    """
    num_iterations = 135 if param == "conductivity" else 75
    calibrate_material(
        param,
        gain_fn=radio_map_gain_fn(),
        num_iterations=num_iterations,
        rtol=0.01,
        loss_threshold=2e-3,
        plot_name=f"{param}_calibration_radio_map.png",
    )
