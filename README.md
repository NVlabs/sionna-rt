<!--
SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->
# Sionna RT: The Ray Tracing Package of Sionna&trade;

[Sionna RT](https://nvlabs.github.io/sionna-rt) is the stand-alone ray tracing package of the [Sionna&trade; Library for Research
on Communication Systems](https://github.com/NVlabs/sionna).
It is built on top of [Mitsuba 3](https://github.com/mitsuba-renderer/mitsuba3) and is interoperable with
[TensorFlow](https://www.tensorflow.org/), [PyTorch](https://pytorch.org/), and [JAX](https://jax.readthedocs.io/en/latest/index.html).

The official documentation can be found on the [Sionna
website](https://nvlabs.github.io/sionna/).


## Installation

The recommended way to install Sionna RT is via pip:

```
pip install sionna-rt
```

Sionna RT has the same requirements as Mitsuba 3 and we refer to its
[installation guide](https://mitsuba.readthedocs.io/en/stable/) for further information.

To run Sionna RT on CPU, [LLVM](https://llvm.org) is required by Dr.Jit.
Please check the [installation instructions for the LLVM backend](https://drjit.readthedocs.io/en/latest/what.html#backends).

### Installation from source
After to cloning the repository, you can install
``sionna-rt`` by running the following command from within the repository's root directory:

```
pip install .
```


## Testing
First, you need to install the test requirements by executing the
following command from the repository's root directory:

```
pip install '.[test]'
```

The unit tests can then be executed by running ``pytest`` from within the
``test`` folder.

## Building the Documentation
Install the requirements for building the documentation by running the following
command from the repository's root directory:

```
pip install '.[doc]'
```

You might need to install [pandoc](https://pandoc.org) manually.

You can then build the documentation by executing ``make html`` from within the ``doc`` folder.

The documentation can then be served by any web server, e.g.,

```
python -m http.server --dir build/html
```

## For Developers

The documentation of Sionna RT includes [developer guides](https://nvlabs.github.io/sionna/rt/developer/developer.html)
explaining how to extend it with custom antenna patterns, radio materials, etc.

Development requirements can be installed by executing from the repository's root directory:

```
pip install '.[dev]'
```

Linting of the code can be achieved by running ```pylint src/``` from the
repository's root directory.

## License and Citation

Sionna RT is Apache-2.0 licensed, as found in the [LICENSE](https://github.com/nvlabs/sionna-rt/blob/main/LICENSE) file.

If you use this software, please cite it as:
```bibtex
@software{sionna,
 title = {Sionna},
 author = {Hoydis, Jakob and Cammerer, Sebastian and {Ait Aoudia}, Fayçal and Nimier-David, Merlin and Maggi, Lorenzo and Marcus, Guillermo and Vem, Avinash and Keller, Alexander},
 note = {https://nvlabs.github.io/sionna/},
 year = {2022},
 version = {2.1.0}
}
```
