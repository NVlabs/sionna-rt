#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Stores the computed propagation paths"""

import mitsuba as mi
import drjit as dr

from typing import Literal

from .paths_buffer import PathsBuffer
from sionna.rt.constants import InteractionType, INVALID_SHAPE,\
    INVALID_PRIMITIVE
from sionna.rt import Scene
from sionna.rt.utils import cpx_mul, cpx_abs_square, sinc, cpx_convert,\
    map_angle_to_canonical_range, theta_phi_from_unit_vec

class Paths:
    # pylint: disable=line-too-long
    r"""
    Paths()

    Stores the simulated propagation paths

    Paths are generated for the loaded scene using a path solver, such as
    :class:`~sionna.rt.PathSolver`. Please refer to the documentation of this
    class for further details.
    """

    def __init__(self,
                 scene: Scene,
                 src_positions: mi.Point3f,
                 tgt_positions: mi.Point3f,
                 tx_velocities: mi.Vector3f,
                 rx_velocities: mi.Vector3f,
                 synthetic_array: bool,
                 paths_buffer: PathsBuffer,
                 rel_ant_positions_tx: mi.Point3f | None,
                 rel_ant_positions_rx: mi.Point3f | None):
        # Internal constructor. Users obtain instances from a path solver.

        self._paths_buffer = paths_buffer

        # Sources and targets, i.e., end-points of the  paths
        self._src_positions = src_positions
        self._tgt_positions = tgt_positions

        # Velocities of the radio devices
        self._tx_velocities = tx_velocities
        self._rx_velocities = rx_velocities

        # References to the transmitter and receiver arrays
        self._tx_array = scene.tx_array
        self._rx_array = scene.rx_array

        # Numbers of transmitters and receivers
        self._num_tx = len(scene.transmitters)
        self._num_rx = len(scene.receivers)

        # Flag indicating if synthetic arrays were used
        self._synthetic_array = synthetic_array

        # Wavelength for which the paths were computed
        self._wavelength = scene.wavelength

        # Frequency for which the paths were computed
        self._frequency = scene.frequency

        # Flag indicating if the tensors storing the paths components were built
        # For efficiency, these are only built if requested.
        self._paths_components_built = False

        # To avoid errors, we handle the corner case in which no paths were
        # found separately
        if paths_buffer.buffer_size == 0:
            self._build_empty_paths()
            return # Stop init

        # The following quantities are used in a few internal functions.
        # They are pre-computed here to avoid code duplication

        # Effective array size
        # If synthetic array is assumed, only a single source (target) was
        # used to model for the transmitter (receiver) array.
        # The synthetic phases shifts are applied afterwards.
        if self._synthetic_array:
            self._eff_tx_array_size = 1
            self._eff_rx_array_size = 1
        else:
            self._eff_tx_array_size = self._tx_array.array_size
            self._eff_rx_array_size = self._rx_array.array_size

        src_ind = paths_buffer.source_indices
        tgt_ind = paths_buffer.target_indices

        # Indices of transmitters, receivers, and antennas.
        # An antenna-first ordering is assumed.
        self._tx_ind = src_ind // self._eff_tx_array_size
        self._rx_ind = tgt_ind // self._eff_rx_array_size
        #
        self._tx_ant_ind = src_ind % self._eff_tx_array_size
        self._rx_ant_ind = tgt_ind % self._eff_rx_array_size

        # The following uses `dr.scatter_inc` to jointly:
        # - Compute the maximum number of paths over all (source, target)
        # couples, which is then used to allocate the memory for the tensors.
        # - Compute a path index such that paths sharing the same
        # (source, target) couple do not share the same index. This is later
        # used to scatter data in the allocated tensors.
        #
        # We must ensure that the path index used for scattering the path data
        # in the allocated tensors are unique for every (source, target) couple.
        # Map (src_ind, tgt_ind) to a unique integer identifying the path source
        # and target
        num_src = self._num_tx*self._eff_tx_array_size
        num_tgt = self._num_rx*self._eff_rx_array_size
        src_tgt_id = tgt_ind*num_src + src_ind
        #
        num_paths = dr.zeros(mi.UInt, num_src*num_tgt)
        self._path_ind = dr.scatter_inc(num_paths, src_tgt_id)
        dr.eval(self._path_ind)
        self._max_num_paths = dr.max(num_paths)[0]

        # Build the tensors

        # Builds tensors from the paths buffer
        self._build_from_buffer()

        # Apply synthetic array if required
        if self._synthetic_array:
            self._apply_synthetic_array(rel_ant_positions_tx,
                                        rel_ant_positions_rx)

        # Fuse the pattern and array dimensions
        self._fuse_pattern_array_dims()

    @property
    def sources(self):
        # pylint: disable=line-too-long
        r"""Positions of the paths sources. If synthetic
        arrays are not used (:attr:`~sionna.rt.Paths.synthetic_array` is `False`),
        then every transmit antenna is modeled as a source of paths.
        Otherwise, transmitters are modelled as if they had a single antenna
        located at their :attr:`~sionna.rt.RadioDevice.position`. The channel
        responses for each individual antenna of the arrays are then computed
        "synthetically" by applying appropriate phase shifts.

        :type: :py:class:`mi.Point3f`"""
        return self._src_positions

    @property
    def targets(self):
        # pylint: disable=line-too-long
        r"""Positions of the paths targets. If synthetic
        arrays are not used (:attr:`~sionna.rt.Paths.synthetic_array` is `False`),
        then every receiver antenna is modeled as a source of paths.
        Otherwise, receivers are modelled as if they had a single antenna
        located at their :attr:`~sionna.rt.RadioDevice.position`. The channel
        responses for each individual antenna of the arrays are then computed
        "synthetically" by applying appropriate phase shifts.

        :type: :py:class:`mi.Point3f`"""
        return self._tgt_positions

    @property
    def tx_array(self):
        """Antenna array used by transmitters

        :type: :class:`~sionna.rt.AntennaArray`
        """
        return self._tx_array

    @property
    def rx_array(self):
        """Antenna array used by receivers

        :type: :class:`~sionna.rt.AntennaArray`"""
        return self._rx_array

    @property
    def num_tx(self):
        """Number of transmitters

        :type: :py:class:`int`"""
        return self._num_tx

    @property
    def num_rx(self):
        """Number of receivers

        :type: :py:class:`int`
        """
        return self._num_rx

    @property
    def synthetic_array(self):
        """Flag indicating if synthetic arrays were used to trace the paths

        :type: :py:class:`bool`
        """
        return self._synthetic_array

    @property
    def valid(self):
        # pylint: disable=line-too-long
        """
        Flags indicating valid paths

        :type: ``mi.TensorXb [num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths]``
        """
        return self._valid

    @property
    def a(self):
        # pylint: disable=line-too-long
        r"""
        Real and imaginary components of the channel coefficients [unitless, linear scale]

        :type: ``Tuple[mi.TensorXf [num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths], mi.TensorXf [num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths]]``
        """
        return self._a_real, self._a_imag

    @property
    def tau(self):
        # pylint: disable=line-too-long
        """
        Paths delays [s]

        :type: ``mi.TensorXf [num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths] or [num_rx, num_tx, num_paths]``
        """
        return self._tau

    @property
    def theta_t(self):
        # pylint: disable=line-too-long
        """
        Zenith angles of departure [rad]

        :type: ``mi.TensorXf [num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths] or [num_rx, num_tx, num_paths]``
        """
        return self._theta_t

    @property
    def phi_t(self):
        # pylint: disable=line-too-long
        """
        Azimuth angles of departure [rad]

        :type: ``mi.TensorXf [num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths] or [num_rx, num_tx, num_paths]``
        """
        return self._phi_t

    @property
    def theta_r(self):
        # pylint: disable=line-too-long
        """
        Zenith angles of arrival [rad]

        :type: ``mi.TensorXf [num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths] or [num_rx, num_tx, num_paths]``
        """
        return self._theta_r

    @property
    def phi_r(self):
        # pylint: disable=line-too-long
        """
        Azimuth angles of arrival [rad]

        :type: ``mi.TensorXf [num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths] or [num_rx, num_tx, num_paths]``
        """
        return self._phi_r

    @property
    def interactions(self):
        # pylint: disable=line-too-long
        """
        Interaction type represented using
        :class:`~sionna.rt.constants.InteractionType`

        :type: ``mi.TensorXu [max_depth, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths] or [max_depth, num_rx, num_tx, num_paths]``
        """
        if not self._paths_components_built:
            self._build_paths_components()
        return self._interactions

    @property
    def objects(self):
        # pylint: disable=line-too-long
        """
        IDs of the intersected objects. Invalid objects are represented by
        :data:`~sionna.rt.constants.INVALID_SHAPE`.

        :type: ``mi.TensorXu [max_depth, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths] or [max_depth, num_rx, num_tx, num_paths]``
        """
        if not self._paths_components_built:
            self._build_paths_components()
        return self._shapes

    @property
    def primitives(self):
        # pylint: disable=line-too-long
        """
        Indices of the intersected primitives. Invalid primitives are
        represented by :data:`~sionna.rt.constants.INVALID_PRIMITIVE`.

        :type: ``mi.TensorXu [max_depth, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths] or [max_depth, num_rx, num_tx, num_paths]``
        """
        if not self._paths_components_built:
            self._build_paths_components()
        return self._primitives

    @property
    def vertices(self):
        # pylint: disable=line-too-long
        """
        Paths' vertices, i.e., the interaction points of the paths with the
        scene

        :type: ``mi.TensorXf [max_depth, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, 3] or [max_depth, num_rx, num_tx, num_paths, 3]``
        """
        if not self._paths_components_built:
            self._build_paths_components()
        return self._vertices

    @property
    def doppler(self):
        # pylint: disable=line-too-long
        r"""
        Doppler shift for each path

        To understand how Doppler shifts are computed, let us consider a single propagation path undergoing
        :math:`n` scattering processes, e.g., reflection, diffuse scattering,
        refraction, as shown in the figure below.

        .. figure:: ../figures/doppler.png
            :align: center

        The object on which lies the :math:`i\text{th}` scattering point has the velocity vector
        :math:`\hat{\mathbf{v}}_i` and the outgoing ray direction at this point is
        denoted :math:`\hat{\mathbf{k}}_i`. The first and last point correspond to the transmitter
        and receiver, respectively. We therefore have

        .. math::

            \hat{\mathbf{k}}_0 &= \hat{\mathbf{r}}(\theta_{\text{T}}, \varphi_{\text{T}})\\
            \hat{\mathbf{k}}_{n} &= -\hat{\mathbf{r}}(\theta_{\text{R}}, \varphi_{\text{R}})

        where :math:`(\theta_{\text{T}}, \varphi_{\text{T}})` are the AoDs,
        :math:`(\theta_{\text{R}}, \varphi_{\text{R}})` are the AoAs, and :math:`\hat{\mathbf{r}}(\theta,\varphi)` is defined in :eq:`spherical_vecs`.

        If the transmitter emits a signal with frequency :math:`f`, the receiver
        will observe the signal at frequency :math:`f'=f + f_\Delta`, where :math:`f_\Delta` is the Doppler
        shift, which can be computed as :cite:p:`Wiffen2018`

        .. math::

            f' = f \prod_{i=0}^n \frac{1 - \frac{\mathbf{v}_{i+1}^\mathsf{T}\hat{\mathbf{k}}_i}{c}}{1 - \frac{\mathbf{v}_{i}^\mathsf{T}\hat{\mathbf{k}}_i}{c}}.

        Under the assumption that :math:`\lVert \mathbf{v}_i \rVert\ll c`, we can apply the Taylor expansion :math:`(1-x)^{-1}\approx 1+x`, for :math:`x\ll 1`, to the previous equation
        to obtain

        .. math::

            f' &\approx f \prod_{i=0}^n \left(1 - \frac{\mathbf{v}_{i+1}^\mathsf{T}\hat{\mathbf{k}}_i}{c}\right)\left(1 + \frac{\mathbf{v}_{i}^\mathsf{T}\hat{\mathbf{k}}_i}{c}\right)\\
               &\approx f \left(1 + \sum_{i=0}^n \frac{\mathbf{v}_{i}^\mathsf{T}\hat{\mathbf{k}}_i -\mathbf{v}_{i+1}^\mathsf{T}\hat{\mathbf{k}}_i}{c} \right)

        where the second line results from ignoring terms in :math:`c^{-2}`. Solving for :math:`f_\Delta`, grouping terms with the same :math:`\mathbf{v}_i` together, and using :math:`f=c/\lambda`, we obtain

        .. math::

            f_\Delta = \frac{1}{\lambda}\left(\mathbf{v}_{0}^\mathsf{T}\hat{\mathbf{k}}_0 - \mathbf{v}_{n+1}^\mathsf{T}\hat{\mathbf{k}}_n + \sum_{i=1}^n \mathbf{v}_{i}^\mathsf{T}\left(\hat{\mathbf{k}}_i-\hat{\mathbf{k}}_{i-1} \right) \right) \qquad \text{[Hz]}.

        :type: ``mi.TensorXf [num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths] or [num_rx, num_tx, num_paths]``
        """
        return self._doppler

    def cir(self, *,
        sampling_frequency: float = 1.,
        num_time_steps: int = 1,
        normalize_delays: bool = True,
        reverse_direction: bool = False,
        out_type: Literal["drjit", "jax", "numpy", "tf", "torch"] = "drjit"
    ):
        # pylint: disable=line-too-long
        r"""
        Returns the baseband-equivalent channel impulse response :eq:`h_b`
        which can be used for link simulations by other Sionna components.
        Optionally, time evolution of the channel can be simulated based
        on the Doppler shifts of all paths.

        The baseband equivalent channel coefficient :math:`a^{\text{b}}_{i}(t)`
        at time :math:`t` is computed as :

        .. math::
            a^{\text{b}}_{i}(t) = \underbrace{a_{i} e^{-j2 \pi f \tau_{i}}}_{a^{\text{b}}_{i} } \underbrace{e^{j 2\pi f_{\Delta, i} t}}_{\text{Doppler phase shift}}

        where :math:`i` is the index of an arbitrary path, :math:`a_{i}`
        is the passband path coefficient (:attr:`~sionna.rt.Paths.a`),
        :math:`\tau_{i}` is the path delay (:attr:`~sionna.rt.Paths.tau`),
        :math:`f` is the carrier frequency, and :math:`f_{\Delta, i}` is the
        Doppler shift of the :math:`i\text{th}` path.

        :param sampling_frequency: Frequency [Hz] at which the channel impulse
            response is sampled

        :param num_time_steps: Number of time steps

        :param normalize_delays: If set to `True`, path delays are normalized
            such that the first path between any pair of antennas of a
            transmitter and receiver arrives at :math:`\tau = 0`

        :param reverse_direction: If set to True, swaps receivers and
            transmitters

        :param out_type: Name of the desired output type.
            Currently supported are
            `Dr.Jit <https://drjit.readthedocs.io/en/latest/reference.html>`_
            ("drjit"), `Numpy <https://numpy.org>`_ ("numpy"),
            `Jax <https://jax.readthedocs.io/en/latest/index.html>`_ ("jax"),
            `TensorFlow <https://www.tensorflow.org>`_ ("tf"),
            and `PyTorch <https://pytorch.org>`_ ("torch").

        :return:
            Real and imaginary components of the baseband equivalent channel
            coefficients :math:`a^{\text{b}}_{i}`
        :return type: Shape: [num_rx, num_rx_ant, num_tx,
            num_tx_ant, num_paths, num_time_steps],
            Type: ``Tuple[mi.TensorXf, mi.TensorXf]``
            | ``numpy.ndarray`` | ``jax.Array``
            | ``tf.Tensor`` | ``torch.Tensor``

        :return: Paths delays :math:`\tau_{i}` [s]
        :return type: Shape: [num_rx, num_rx_ant, num_tx,
            num_tx_ant, num_paths] or [num_rx, num_tx, num_paths],
            Type: ``mi.TensorXf``
            | ``numpy.ndarray`` | ``jax.Array``
            | ``tf.Tensor`` | ``torch.Tensor``
        """

        # Reverse direction if required
        if reverse_direction:
            a = self._reverse_direction(self.a)
            tau, doppler = self._reverse_direction((self.tau, self.doppler))
        else:
            a = dr.copy(self.a)
            tau = dr.copy(self.tau)
            doppler = dr.copy(self.doppler)

        # If no paths, then return immediately
        if tau.shape[-1] == 0:
            a = (dr.zeros(mi.TensorXf, list(a[0].shape)+[num_time_steps]),
                 dr.zeros(mi.TensorXf, list(a[1].shape)+[num_time_steps]))
            tau = dr.zeros(mi.TensorXf, tau.shape)
            if out_type == "drjit":
                return a, tau
            return cpx_convert(a, out_type), getattr(tau, out_type)()

        # Normalize delays if required
        if normalize_delays:
            tau = dr.select(tau<0, dr.inf, tau)
            if self.synthetic_array:
                min_tau = dr.min(tau, axis=-1)
                min_tau = dr.reshape(mi.TensorXf, min_tau,
                                    shape=list(min_tau.shape) + [1])
            else:
                min_tau = dr.min(tau, axis=(1,3,4))
                num_rx, num_tx = min_tau.shape
                min_tau = dr.reshape(mi.TensorXf, min_tau,
                                     shape=[num_rx, 1, num_tx, 1, 1])
            # Apply delay normalization
            tau -= min_tau

            # Set delays of invalid paths to -1
            tau = dr.select(dr.isnan(tau) | dr.isinf(tau), -1, tau)

        # Compute baseband-equivalent CIR
        if self.synthetic_array:
            num_rx, num_tx, num_paths = tau.shape
            reshape_to = [num_rx, 1, num_tx, 1, num_paths]
            tau_ = dr.reshape(mi.TensorXf, tau, reshape_to)
        else:
            tau_ = tau

        # Compute phase shifts and apply to paths coefficients
        tau_ = dr.select(tau_==-1, 0, tau_)
        phase = -dr.two_pi*self._frequency*tau_
        phase = map_angle_to_canonical_range(phase)
        sin_phase, cos_phase = dr.sincos(phase)
        exp = (cos_phase, sin_phase)
        a = cpx_mul(a, exp)

        # Add dummy time dimensions
        a = [dr.reshape(mi.TensorXf, a_, list(a_.shape) + [1]) for a_ in a]

        # Apply Doppler phase shifts
        if num_time_steps > 1:
            # Reshape the Doppler shift tensor to fit `a`
            if self.synthetic_array:
                doppler = dr.reshape(mi.TensorXf, doppler, reshape_to + [1])
            else:
                doppler = dr.reshape(mi.TensorXf, doppler,
                                     list(doppler.shape) + [1])
            time_steps = dr.arange(mi.Float, num_time_steps)/sampling_frequency
            time_steps = dr.reshape(mi.TensorXf, time_steps, [1, 1, 1, 1, 1,
                                                              num_time_steps])
            phase = dr.two_pi*doppler*time_steps
            phase = map_angle_to_canonical_range(phase)
            sin_phase, cos_phase = dr.sincos(phase)
            exp = (cos_phase, sin_phase)
            a = cpx_mul(a, exp)

        if out_type == "drjit":
            return a, tau
        return cpx_convert(a, out_type), getattr(tau, out_type)()

    def taps(self,
        bandwidth: float,
        l_min: int,
        l_max: int,
        sampling_frequency: float | None = None,
        num_time_steps: int = 1,
        normalize: bool = False,
        normalize_delays: bool = True,
        reverse_direction: bool = False,
        out_type: Literal["drjit", "jax", "numpy", "tf", "torch"] = "drjit"
    ):
        r"""
        Returns the channel taps forming the discrete complex
        baseband-equivalent channel impulse response

        This function assumes that a sinc filter is used for pulse shaping and
        receive filtering. Therefore, given a channel impulse response
        :math:`(a_{i}^\text{b}(t), \tau_{i}), 0 \leq i \leq M-1` (which can be
        computed by :meth:`~sionna.rt.Paths.cir`), the
        :math:`\ell\text{th}` channel tap at sample instance :math:`n`
        is computed as follows (see (Eq. 2.34) :cite:p:`Tse_RT`):

        .. math::
            \bar{h}_{n, \ell}
            = \sum_{i=0}^{M-1} a_{i}^\text{b}\left(\frac{n}{f_s}\right)
                \text{sinc}\left( \ell - W\tau_{i} \right)

        for :math:`\ell` ranging from ``l_min`` to ``l_max``, where :math:`W` is
        the ``bandwidth`` and :math:`f_s` is the ``sampling_frequency``.
        By default, ``sampling_frequency`` equals ``bandwidth``, in which case
        :math:`f_s = W`.

        :param bandwidth: Bandwidth [Hz] to which the channel impulse response
            will be limited

        :param l_min: Smallest time-lag for the discrete complex
            baseband-equivalent channel (:math:`L_{\text{min}}`)

        :param l_max: Largest time-lag for the discrete complex
            baseband-equivalent channel (:math:`L_{\text{max}}`)

        :param sampling_frequency: Frequency [Hz] at which the channel impulse
            response is sampled. If set to `None`, the ``bandwidth`` is used
            instead.

        :param num_time_steps: Number of time steps

        :param normalize: If set to `True`, the channel is normalized such that
            the average total energy of the channel taps is equal to one.

        :param normalize_delays: If set to `True`, path delays are normalized
            such that the first path between any pair of antennas of a
            transmitter and receiver arrives at :math:`\tau = 0`.

        :param reverse_direction: If set to True, swaps receivers and
            transmitters

        :param out_type: Name of the desired output type.
            Currently supported are
            `Dr.Jit <https://drjit.readthedocs.io/en/latest/reference.html>`_
            ("drjit"), `Numpy <https://numpy.org>`_ ("numpy"),
            `Jax <https://jax.readthedocs.io/en/latest/index.html>`_ ("jax"),
            `TensorFlow <https://www.tensorflow.org>`_ ("tf"),
            and `PyTorch <https://pytorch.org>`_ ("torch").

        :return: Channel tap coefficients
        :return type: Shape: [num_rx, num_rx_ant, num_tx, num_tx_ant,
            num_time_steps, l_max - l_min + 1],
            Type: ``Tuple[mi.TensorXf, mi.TensorXf]``
            | ``numpy.ndarray`` | ``jax.Array``
            | ``tf.Tensor`` | ``torch.Tensor``
        """

        # Get complex baseband equivalent CIR
        if sampling_frequency is None:
            sampling_frequency = bandwidth
        a, tau = self.cir(sampling_frequency=sampling_frequency,
                          num_time_steps=num_time_steps,
                          normalize_delays=normalize_delays,
                          reverse_direction=reverse_direction)

        # If no paths, then return immediately
        if tau.shape[-1] == 0:
            out_shape = list(a[0].shape)[:-2]+[num_time_steps,
                                                    l_max - l_min + 1]
            h = (dr.zeros(mi.TensorXf, out_shape),
                 dr.zeros(mi.TensorXf, out_shape))
            if out_type == "drjit":
                return h
            return cpx_convert(h, out_type)

        # Tap indices
        l = dr.arange(mi.Float, l_min, l_max+1)

        # Reshape for matching dimensions
        l = dr.reshape(mi.TensorXf, l, shape=[1]*6 + [-1])
        a = [dr.reshape(mi.TensorXf, a_, shape=list(a_.shape)+[1]) for a_ in a]
        if self.synthetic_array:
            num_rx, num_tx, num_paths = tau.shape
            reshape_to = [num_rx, 1, num_tx, 1, num_paths, 1, 1]
            tau = dr.reshape(mi.TensorXf, tau, reshape_to)
        else:
            tau = dr.reshape(mi.TensorXf, tau, shape=list(tau.shape) + [1,1])

        # Compute taps by low-pass filtering
        g = sinc(l - tau*bandwidth)
        h = [dr.sum(a_*g, axis=4) for a_ in a]

        # Normalize energy
        if normalize:
            # Normalization is performed such that for every link the energy
            # per block is one.
            # The total energy of a channel response is the sum of the squared
            # norm over the channel taps.

            # Total energy in all taps
            c = dr.sum(cpx_abs_square(h), axis=-1)

            # Average over time steps, RX antennas, and TX antennas
            c = dr.mean(c, axis=(1, 3, 4))

            # Reshaping before normalization
            num_rx, num_tx = c.shape
            c = dr.reshape(mi.TensorXf, c, [num_rx, 1, num_tx, 1, 1, 1])

            # Normalization
            h = [dr.select(c==0, 0, h_*dr.rsqrt(c)) for h_ in h]

        if out_type == "drjit":
            return h
        return cpx_convert(h, out_type)

    def cfr(self,
        frequencies: mi.Float,
        sampling_frequency: float = 1.,
        num_time_steps: int = 1,
        normalize_delays: bool = True,
        normalize: bool = False,
        reverse_direction: bool = False,
        out_type: Literal["drjit", "jax", "numpy", "tf", "torch"] = "drjit"
    ):
        r"""
        Compute the frequency response of the channel at ``frequencies``.
        Optionally, time evolution of the channel can be simulated based
        on the Doppler shifts of all paths.

        Given a channel impulse response
        :math:`(a_i^\text{b}(t), \tau_{i}), 0 \leq i \leq M-1`, as computed
        by :meth:`~sionna.rt.Paths.cir`,
        the channel frequency response for the frequency :math:`f`
        is computed as follows:

        .. math::
            \widehat{h}(f, t) = \sum_{i=0}^{M-1}a_i^\text{b}(t) e^{-j2\pi f \tau_{i}}

        The time evolution of the channel is simulated as described in
        the documentation of :meth:`~sionna.rt.Paths.cir`.

        :param frequencies: Frequencies [Hz] at which to compute the
            channel response

        :param sampling_frequency: Frequency [Hz] at which the channel impulse
            response is sampled

        :param num_time_steps: Number of time steps

        :param normalize_delays: If set to `True`, path delays are normalized
            such that the first path between any pair of antennas of a
            transmitter and receiver arrives at :math:`\tau = 0`.

        :param normalize: If set to `True`, the channel is normalized across
            time and frequencies to ensure unit average energy.

        :param reverse_direction: If set to True, swaps receivers and
            transmitters

        :param out_type: Name of the desired output type.
            Currently supported are
            `Dr.Jit <https://drjit.readthedocs.io/en/latest/reference.html>`_
            ("drjit"), `Numpy <https://numpy.org>`_ ("numpy"),
            `Jax <https://jax.readthedocs.io/en/latest/index.html>`_ ("jax"),
            `TensorFlow <https://www.tensorflow.org>`_ ("tf"),
            and `PyTorch <https://pytorch.org>`_ ("torch").

        :return: Real and imaginary components of the channel frequency
            response :math:`\widehat{h}(f, t)`

        :return type: Shape: [num_rx, num_rx_ant, num_tx, num_tx_ant,
            num_time_steps, num_frequencies],
            Type: ``Tuple[mi.TensorXf, mi.TensorXf]``
            | ``numpy.ndarray`` | ``jax.Array``
            | ``tf.Tensor`` | ``torch.Tensor``
        """

        frequencies = mi.Float(frequencies)

        # Get complex baseband equivalent CIR
        a, tau_ = self.cir(sampling_frequency=sampling_frequency,
                           num_time_steps=num_time_steps,
                           normalize_delays=normalize_delays,
                           reverse_direction=reverse_direction)

        # If no paths, then return immediately
        if tau_.shape[-1] == 0:
            out_shape = list(a[0].shape)[:-2]+[num_time_steps,
                                               frequencies.shape[0]]
            h = (dr.zeros(mi.TensorXf, out_shape),
                 dr.zeros(mi.TensorXf, out_shape))
            if out_type == "drjit":
                return h
            return cpx_convert(h, out_type)

        # Add dummy dimensions to account for synthetic arrays
        if self.synthetic_array:
            num_rx, num_tx, num_paths = tau_.shape
            tau_ = dr.reshape(mi.TensorXf, tau_,
                             [num_rx, 1, num_tx, 1, num_paths])

        # Add dimensions for frequencies and time
        tau_ = dr.reshape(mi.TensorXf, tau_, list(tau_.shape) + [1,1])
        a = [dr.reshape(mi.TensorXf, a_, list(a[0].shape) + [1]) for a_ in a]

        # Compute phase shifts
        phase = -dr.two_pi*frequencies*tau_
        phase = map_angle_to_canonical_range(phase)
        sin_phase, cos_phase = dr.sincos(phase)
        exp = (cos_phase, sin_phase)

        # Compute resulting Fourier coefficients
        a_f = cpx_mul(a, exp)

        # Compute CFR
        h_f = (dr.sum(a_f[0], axis=-3), dr.sum(a_f[1], axis=-3))

        # Normalize
        if normalize:
            # Per-link average energy across antennas, time, and frequencies.
            # Zero-energy links must stay zero (avoid 0 * inf = NaN).
            c = dr.mean(cpx_abs_square(h_f), axis=(1, 3, 4, 5))
            num_rx, num_tx = c.shape
            c = dr.reshape(mi.TensorXf, c, [num_rx, 1, num_tx, 1, 1, 1])
            h_f = [dr.select(c == 0, 0, h * dr.rsqrt(c)) for h in h_f]

        if out_type == "drjit":
            return h_f
        return cpx_convert(h_f, out_type)

    ###########################################
    # Internal methods
    ###########################################

    @staticmethod
    def _k_component(k: mi.TensorXf, component: int) -> mi.Float:
        r"""
        Extracts one component of a tensor of direction vectors

        ``k`` is assumed to store the (x,y,z) components along its last
        dimension, i.e., interleaved in its flat array.

        :param k: Tensor of direction vectors with a trailing dimension of 3
        :param component: Index of the component to extract (0, 1, or 2)

        :return: Requested component
        """

        num_items = dr.width(k.array) // 3
        ind = dr.arange(mi.UInt, num_items)*3 + component
        return dr.gather(mi.Float, k.array, ind)

    def _build_from_buffer(self) -> None:
        r"""
        Builds the multi-dimensional tensors storing the paths data from the
        path buffer object

        This function reorganizes the path data (coefficients, delays, angles
        of arrival and departure) into tensors.

        The vertices, intersected objects and primitives and other data only
        required for rendering or for more exhaustive study of the paths are not
        processed by this function and only computed on demand.
        """

        paths_buffer = self._paths_buffer
        num_tx = self._num_tx
        num_rx = self._num_rx
        tx_array_size = self._eff_tx_array_size
        rx_array_size = self._eff_rx_array_size
        tx_ind = self._tx_ind
        rx_ind = self._rx_ind
        tx_ant_ind = self._tx_ant_ind
        rx_ant_ind = self._rx_ant_ind
        num_rx_patterns = len(self._rx_array.antenna_pattern.patterns)
        num_tx_patterns = len(self._tx_array.antenna_pattern.patterns)
        # Path index such that paths sharing the same (source, target) couple
        # do not share the same index.
        path_ind = self._path_ind
        # Maximum number of paths over all (source, target) couples
        max_num_paths = self._max_num_paths

        # Allocate the tensors and fill them
        # `a`
        a_real = dr.zeros(mi.TensorXf, [num_rx, num_rx_patterns, rx_array_size,
                                        num_tx, num_tx_patterns, tx_array_size,
                                        max_num_paths])
        a_imag = dr.zeros(mi.TensorXf, [num_rx, num_rx_patterns, rx_array_size,
                                        num_tx, num_tx_patterns, tx_array_size,
                                        max_num_paths])
        # Scatter indices
        f1 = max_num_paths
        f2 = tx_array_size*f1
        f3 = num_tx_patterns*f2
        f4 = num_tx*f3
        f5 = rx_array_size*f4
        f6 = num_rx_patterns*f5
        a_scat_ind = path_ind + tx_ant_ind*f1 + tx_ind*f3 + rx_ant_ind*f4\
                        + rx_ind*f6
        # Fill the `a` tensor
        for n in range(num_rx_patterns):
            a_scat_ind_ = dr.copy(a_scat_ind) + n*f5
            for m in range(num_tx_patterns):
                a_scat_ind__ = a_scat_ind_ + m*f2
                dr.scatter(a_real.array, dr.real(paths_buffer.a[n][m]),
                        a_scat_ind__)
                dr.scatter(a_imag.array, dr.imag(paths_buffer.a[n][m]),
                        a_scat_ind__)


        # For the other tensors, if a synthetic arrays are used, then the
        # pattern and antenna dimension do not need to be created

        # Shapes and indices for scattering
        if self._synthetic_array:
            tensor_shape = [num_rx, num_tx, max_num_paths]
            num_rx_patterns = num_tx_patterns = 1
        else:
            tensor_shape = [num_rx, num_rx_patterns, rx_array_size, num_tx,
                            num_tx_patterns, tx_array_size, max_num_paths]

        # We need to recompute these factors as `num_rx_patterns`
        # or `num_rx_patterns` might have changed
        f3 = num_tx_patterns*f2
        f4 = num_tx*f3
        f5 = rx_array_size*f4
        f6 = num_rx_patterns*f5
        scat_ind = path_ind + tx_ant_ind*f1 + tx_ind*f3 + rx_ant_ind*f4\
                    + rx_ind*f6

        # Instantiate the tensors
        valid = dr.full(mi.TensorXb, False, tensor_shape)
        tau = dr.full(mi.TensorXf, -1., tensor_shape)
        theta_t = dr.zeros(mi.TensorXf, tensor_shape)
        phi_t = dr.zeros(mi.TensorXf, tensor_shape)
        theta_r = dr.zeros(mi.TensorXf, tensor_shape)
        phi_r = dr.zeros(mi.TensorXf, tensor_shape)
        doppler = dr.zeros(mi.TensorXf, tensor_shape)
        # `k_rx` follows the same convention as the angles of arrival, i.e., it
        # points opposite to the direction of propagation of the incident wave.
        # An extra dimension is required for the (x,y,z) components.
        k_tx = dr.zeros(mi.TensorXf, tensor_shape + [3])
        k_rx = dr.zeros(mi.TensorXf, tensor_shape + [3])

        # The angles of departure and arrival are derived from the directions
        # stored in the buffer, which are the primitive quantities
        k_tx_ = paths_buffer.k_tx
        k_rx_ = paths_buffer.k_rx
        theta_t_, phi_t_ = theta_phi_from_unit_vec(k_tx_)
        theta_r_, phi_r_ = theta_phi_from_unit_vec(k_rx_)

        # Finalize Doppler shift computation by applying the shift due to
        # transmitter and receiver mobility
        doppler_ = self._finalize_doppler_shift_compute()

        # Fill the tensor
        # Note that we cannot fuse this loop with the one used to fill `a` as
        # `num_rx_patterns` or `num_tx_patterns` might be different
        for n in range(num_rx_patterns):
            scat_ind_ = scat_ind + n*f5
            for m in range(num_tx_patterns):
                scat_ind__ = scat_ind_ + m*f2
                #
                dr.scatter(valid.array, True, scat_ind__)
                #
                dr.scatter(tau.array, paths_buffer.tau, scat_ind__)
                #
                dr.scatter(theta_t.array, theta_t_, scat_ind__)
                dr.scatter(phi_t.array, phi_t_, scat_ind__)
                #
                dr.scatter(theta_r.array, theta_r_, scat_ind__)
                dr.scatter(phi_r.array, phi_r_, scat_ind__)
                #
                dr.scatter(k_tx.array, k_tx_.x, scat_ind__*3)
                dr.scatter(k_tx.array, k_tx_.y, scat_ind__*3 + 1)
                dr.scatter(k_tx.array, k_tx_.z, scat_ind__*3 + 2)
                #
                dr.scatter(k_rx.array, k_rx_.x, scat_ind__*3)
                dr.scatter(k_rx.array, k_rx_.y, scat_ind__*3 + 1)
                dr.scatter(k_rx.array, k_rx_.z, scat_ind__*3 + 2)
                #
                dr.scatter(doppler.array, doppler_, scat_ind__)

        self._valid = valid
        self._a_real = a_real
        self._a_imag = a_imag
        self._tau = tau
        self._theta_t = theta_t
        self._phi_t = phi_t
        self._theta_r = theta_r
        self._phi_r = phi_r
        self._k_tx = k_tx
        self._k_rx = k_rx
        self._doppler = doppler

    def _apply_synthetic_array(self,
                               rel_ant_positions_tx: mi.Point3f | None,
                               rel_ant_positions_rx: mi.Point3f | None
                               ) -> None:
        r"""
        Applies the phase shifts to simulate the effect of a synthetic array
        on a planar wave

        :param rel_ant_positions_tx: Positions of the array elements with
            respect to the center of the transmitters. Only required if
            synthetic arrays are used.

        :param rel_ant_positions_rx: Positions of the array elements with
            respect to the center of the receivers. Only required if synthetic
            arrays are used.
        """

        num_tx = self._num_tx
        num_rx = self._num_rx
        tx_array_size = self._tx_array.array_size
        rx_array_size = self._rx_array.array_size
        max_num_paths = dr.shape(self._a_real)[-1]

        # [num_rx, num_rx_patterns, 1, num_tx, num_tx_patterns, 1,
        #   max_num_paths]
        a_real = self._a_real
        a_imag = self._a_imag

        # Directions of arrival and departures
        # The components are read directly from the stored direction vectors,
        # as reconstructing them from the angles is ill-conditioned near the
        # poles.
        # [num_rx, num_tx, max_num_paths]
        k_tx_x = self._k_component(self._k_tx, 0)
        k_tx_y = self._k_component(self._k_tx, 1)
        k_tx_z = self._k_component(self._k_tx, 2)
        # Expand for broadcasting
        k_tx_x = dr.reshape(mi.TensorXf, k_tx_x,
                            [num_rx, 1, 1, num_tx, 1, 1, max_num_paths])
        k_tx_y = dr.reshape(mi.TensorXf, k_tx_y,
                            [num_rx, 1, 1, num_tx, 1, 1, max_num_paths])
        k_tx_z = dr.reshape(mi.TensorXf, k_tx_z,
                            [num_rx, 1, 1, num_tx, 1, 1, max_num_paths])
        # [num_rx, num_tx, max_num_paths]
        k_rx_x = self._k_component(self._k_rx, 0)
        k_rx_y = self._k_component(self._k_rx, 1)
        k_rx_z = self._k_component(self._k_rx, 2)
        # Expand for broadcasting
        k_rx_x = dr.reshape(mi.TensorXf, k_rx_x,
                            [num_rx, 1, 1, num_tx, 1, 1, max_num_paths])
        k_rx_y = dr.reshape(mi.TensorXf, k_rx_y,
                            [num_rx, 1, 1, num_tx, 1, 1, max_num_paths])
        k_rx_z = dr.reshape(mi.TensorXf, k_rx_z,
                            [num_rx, 1, 1, num_tx, 1, 1, max_num_paths])

        # Relative positions of the antennas of the transmitters and receivers
        # [1, 1, 1, num_tx, 1, tx_array_size, 1]
        rel_ant_pos_tx_x = dr.reshape(mi.TensorXf, rel_ant_positions_tx.x,
                                      [1, 1, 1, num_tx, 1, tx_array_size, 1])
        rel_ant_pos_tx_y = dr.reshape(mi.TensorXf, rel_ant_positions_tx.y,
                                      [1, 1, 1, num_tx, 1, tx_array_size, 1])
        rel_ant_pos_tx_z = dr.reshape(mi.TensorXf, rel_ant_positions_tx.z,
                                      [1, 1, 1, num_tx, 1, tx_array_size, 1])

        # Relative positions of the antennas of the transmitters and receivers
        # [num_rx, 1, rx_array_size, 1, 1, 1, 1]
        rel_ant_pos_rx_x = dr.reshape(mi.TensorXf, rel_ant_positions_rx.x,
                                      [num_rx, 1, rx_array_size, 1, 1, 1, 1])
        rel_ant_pos_rx_y = dr.reshape(mi.TensorXf, rel_ant_positions_rx.y,
                                      [num_rx, 1, rx_array_size, 1, 1, 1, 1])
        rel_ant_pos_rx_z = dr.reshape(mi.TensorXf, rel_ant_positions_rx.z,
                                      [num_rx, 1, rx_array_size, 1, 1, 1, 1])

        # Compute the phase shifts by taking the dot products between directions
        # off departure (arrival) and the antenna array relative positions
        # TX
        # [num_rx, 1, 1, num_tx, 1, tx_array_size, max_num_paths]
        tx_phase_shifts = rel_ant_pos_tx_z*k_tx_z
        tx_phase_shifts = dr.fma(rel_ant_pos_tx_y, k_tx_y, tx_phase_shifts)
        tx_phase_shifts = dr.fma(rel_ant_pos_tx_x, k_tx_x, tx_phase_shifts)
        # RX
        # [num_rx, 1, rx_array_size, num_tx, 1, 1, max_num_paths]
        rx_phase_shifts = rel_ant_pos_rx_z*k_rx_z
        rx_phase_shifts = dr.fma(rel_ant_pos_rx_y, k_rx_y, rx_phase_shifts)
        rx_phase_shifts = dr.fma(rel_ant_pos_rx_x, k_rx_x, rx_phase_shifts)
        # Total phase shift
        # [num_rx, 1, rx_array_size, num_tx, 1, tx_array_size, max_num_paths]
        phase_shifts = rx_phase_shifts + tx_phase_shifts
        phase_shifts = dr.two_pi*phase_shifts/self._wavelength

        # Apply the phase shifts
        # [num_rx, num_rx_patterns, rx_array_size, num_tx, num_tx_patterns,
        # tx_array_size, max_num_paths]
        sin_phi, cos_phi = dr.sincos(phase_shifts)
        a_real_ = dr.fma(a_real, cos_phi, -a_imag*sin_phi)
        a_imag_ = dr.fma(a_real, sin_phi, a_imag*cos_phi)
        self._a_real = a_real_
        self._a_imag = a_imag_

    def _fuse_pattern_array_dims(self) -> None:
        r"""
        Merges the pattern and array dimensions of the tensors storing the
        channel coefficients
        """

        num_rx, num_rx_patterns, num_rx_ant, num_tx, num_tx_patterns,\
            num_tx_ant, max_num_paths = dr.shape(self._a_real)

        # [num_rx, num_rx_patterns*rx_array_size, num_tx,
        # num_tx_patterns*tx_array_size, max_num_paths]
        self._a_real = dr.reshape(mi.TensorXf, self._a_real,
                                  [num_rx, num_rx_patterns*num_rx_ant,
                                   num_tx, num_tx_patterns*num_tx_ant,
                                   max_num_paths])
        self._a_imag = dr.reshape(mi.TensorXf, self._a_imag,
                                  [num_rx, num_rx_patterns*num_rx_ant,
                                   num_tx, num_tx_patterns*num_tx_ant,
                                   max_num_paths])

        if not self._synthetic_array:
            self._valid = dr.reshape(mi.TensorXb, self._valid,
                                  [num_rx, num_rx_patterns*num_rx_ant,
                                   num_tx, num_tx_patterns*num_tx_ant,
                                   max_num_paths])
            self._tau = dr.reshape(mi.TensorXf, self._tau,
                                  [num_rx, num_rx_patterns*num_rx_ant,
                                   num_tx, num_tx_patterns*num_tx_ant,
                                   max_num_paths])
            self._theta_t = dr.reshape(mi.TensorXf, self._theta_t,
                                  [num_rx, num_rx_patterns*num_rx_ant,
                                   num_tx, num_tx_patterns*num_tx_ant,
                                   max_num_paths])
            self._phi_t = dr.reshape(mi.TensorXf, self._phi_t,
                                  [num_rx, num_rx_patterns*num_rx_ant,
                                   num_tx, num_tx_patterns*num_tx_ant,
                                   max_num_paths])
            self._theta_r = dr.reshape(mi.TensorXf, self._theta_r,
                                  [num_rx, num_rx_patterns*num_rx_ant,
                                   num_tx, num_tx_patterns*num_tx_ant,
                                   max_num_paths])
            self._phi_r = dr.reshape(mi.TensorXf, self._phi_r,
                                  [num_rx, num_rx_patterns*num_rx_ant,
                                   num_tx, num_tx_patterns*num_tx_ant,
                                   max_num_paths])
            self._k_tx = dr.reshape(mi.TensorXf, self._k_tx,
                                  [num_rx, num_rx_patterns*num_rx_ant,
                                   num_tx, num_tx_patterns*num_tx_ant,
                                   max_num_paths, 3])
            self._k_rx = dr.reshape(mi.TensorXf, self._k_rx,
                                  [num_rx, num_rx_patterns*num_rx_ant,
                                   num_tx, num_tx_patterns*num_tx_ant,
                                   max_num_paths, 3])
            self._doppler = dr.reshape(mi.TensorXf, self._doppler,
                                  [num_rx, num_rx_patterns*num_rx_ant,
                                   num_tx, num_tx_patterns*num_tx_ant,
                                   max_num_paths])

    def _build_empty_paths(self) -> None:
        r"""
        Builds empty tensors with dimensions fitting the setting of the paths
        buffer, i.e., same number of radio devices, antennas, etc
        """

        max_depth = self._paths_buffer.max_depth
        num_tx = self._num_tx
        num_rx = self._num_rx
        num_tx_ant = self._tx_array.num_ant
        num_rx_ant = self._rx_array.num_ant

        # Fused antenna dimensions (array size x polarization patterns)
        a_tensor_base_shape = [num_rx, num_rx_ant, num_tx, num_tx_ant, 0]
        if self._synthetic_array:
            other_tensor_base_shape = [num_rx, num_tx, 0]
        else:
            # Same fused antenna axes as ``a`` / non-empty non-synthetic paths
            other_tensor_base_shape = a_tensor_base_shape

        self._valid = dr.full(mi.TensorXb, False, other_tensor_base_shape)
        self._a_real = dr.zeros(mi.TensorXf, a_tensor_base_shape)
        self._a_imag = dr.zeros(mi.TensorXf, a_tensor_base_shape)
        self._tau = dr.zeros(mi.TensorXf, other_tensor_base_shape)
        self._theta_t = dr.zeros(mi.TensorXf, other_tensor_base_shape)
        self._phi_t = dr.zeros(mi.TensorXf, other_tensor_base_shape)
        self._theta_r = dr.zeros(mi.TensorXf, other_tensor_base_shape)
        self._phi_r = dr.zeros(mi.TensorXf, other_tensor_base_shape)
        self._k_tx = dr.zeros(mi.TensorXf, other_tensor_base_shape + [3])
        self._k_rx = dr.zeros(mi.TensorXf, other_tensor_base_shape + [3])
        self._doppler = dr.zeros(mi.TensorXf, other_tensor_base_shape)
        #
        self._interactions = dr.full(mi.TensorXu, InteractionType.NONE,
                                     [max_depth] + other_tensor_base_shape)
        self._shapes = dr.full(mi.TensorXu, INVALID_SHAPE,
                               [max_depth] + other_tensor_base_shape)
        self._primitives = dr.full(mi.TensorXu, INVALID_PRIMITIVE,
                                   [max_depth] + other_tensor_base_shape)
        self._vertices = dr.zeros(mi.TensorXf,
                                  [max_depth] + other_tensor_base_shape + [3])

        self._paths_components_built = True

    def _build_paths_components(self) -> None:
        r"""
        Builds and fills tensors for storing the additional paths data required
        for paths visualization or more exhaustive studies of the paths
        (vertices, interaction types, etc)
        """

        paths_buffer = self._paths_buffer
        depth_dim_size = paths_buffer.depth_dim_size
        total_paths_count = paths_buffer.buffer_size
        num_tx = self._num_tx
        num_rx = self._num_rx
        tx_array_size = self._eff_tx_array_size
        rx_array_size = self._eff_rx_array_size
        tx_ind = self._tx_ind
        rx_ind = self._rx_ind
        tx_ant_ind = self._tx_ant_ind
        rx_ant_ind = self._rx_ant_ind
        num_rx_patterns = len(self._rx_array.antenna_pattern.patterns)
        num_tx_patterns = len(self._tx_array.antenna_pattern.patterns)
        # Path index such that paths sharing the same (source, target) couple
        # do not share the same index.
        path_ind = self._path_ind
        # Maximum number of paths over all (source, target) couples
        max_num_paths = self._max_num_paths

        # Shape of the tensors
        if self._synthetic_array:
            tensor_shape = [depth_dim_size, num_rx, num_tx, max_num_paths]
            num_rx_patterns = num_tx_patterns = 1
        else:
            tensor_shape = [depth_dim_size, num_rx, num_rx_patterns,
                            rx_array_size, num_tx, num_tx_patterns,
                            tx_array_size, max_num_paths]

        # To build the indices that are used for scattering in the tensors, we
        # reshape the path, device, and antenna indices to tensors to leverage
        # the broadcasting feature of dr.jit
        # [depth_dim_size]
        depth_ind = dr.arange(mi.UInt, depth_dim_size)
        # [1, depth_dim_size]
        depth_ind = dr.reshape(mi.TensorXu, depth_ind, [1, depth_dim_size])
        # [total_paths_count, 1]
        path_ind = dr.reshape(mi.TensorXu, path_ind, [total_paths_count, 1])
        tx_ant_ind = dr.reshape(mi.TensorXu, tx_ant_ind, [total_paths_count, 1])
        tx_ind = dr.reshape(mi.TensorXu, tx_ind, [total_paths_count, 1])
        rx_ant_ind = dr.reshape(mi.TensorXu, rx_ant_ind, [total_paths_count, 1])
        rx_ind = dr.reshape(mi.TensorXu, rx_ind, [total_paths_count, 1])
        # Scatter indices
        f1 = max_num_paths
        f2 = tx_array_size*f1
        f3 = num_tx_patterns*f2
        f4 = num_tx*f3
        f5 = rx_array_size*f4
        f6 = num_rx_patterns*f5
        f7 = num_rx*f6
        # [total_paths_count, depth_dim_size]
        scat_ind = path_ind + tx_ant_ind*f1 + tx_ind*f3 + rx_ant_ind*f4\
                    + rx_ind*f6 + depth_ind*f7
        # [total_paths_count*depth_dim_size]
        scat_ind = scat_ind.array

        # Allocate and fill the tensors

        interactions = dr.full(mi.TensorXu, InteractionType.NONE, tensor_shape)
        shapes = dr.full(mi.TensorXu, INVALID_SHAPE, tensor_shape)
        primitives = dr.full(mi.TensorXu, INVALID_PRIMITIVE, tensor_shape)
        # `vertices` requires an extra dimension for storing the (x,y,z)
        # coordinates
        vertices = dr.zeros(mi.TensorXf, tensor_shape + [3])

        for n in range(num_rx_patterns):
            scat_ind_ = scat_ind + n*f5
            for m in range(num_tx_patterns):
                scat_ind__ = scat_ind_ + m*f2
                #
                dr.scatter(interactions.array,
                           paths_buffer.interaction_types.array,
                           scat_ind__)
                #
                dr.scatter(shapes.array, paths_buffer.shapes.array, scat_ind__)
                #
                dr.scatter(primitives.array, paths_buffer.primitives.array,
                           scat_ind__)
                #
                dr.scatter(vertices.array, paths_buffer.vertices_x.array,
                           scat_ind__*3)
                dr.scatter(vertices.array, paths_buffer.vertices_y.array,
                           scat_ind__*3 + 1)
                dr.scatter(vertices.array, paths_buffer.vertices_z.array,
                           scat_ind__*3 + 2)

        if not self._synthetic_array:
            interactions = dr.reshape(mi.TensorXu, interactions,
                                        [depth_dim_size,
                                        num_rx, num_rx_patterns*rx_array_size,
                                        num_tx, num_tx_patterns*tx_array_size,
                                        max_num_paths])
            shapes = dr.reshape(mi.TensorXu, shapes,
                                        [depth_dim_size,
                                        num_rx, num_rx_patterns*rx_array_size,
                                        num_tx, num_tx_patterns*tx_array_size,
                                        max_num_paths])
            primitives = dr.reshape(mi.TensorXu, primitives,
                                        [depth_dim_size,
                                        num_rx, num_rx_patterns*rx_array_size,
                                        num_tx, num_tx_patterns*tx_array_size,
                                        max_num_paths])
            vertices = dr.reshape(mi.TensorXf, vertices,
                                        [depth_dim_size,
                                        num_rx, num_rx_patterns*rx_array_size,
                                        num_tx, num_tx_patterns*tx_array_size,
                                        max_num_paths, 3])

        self._interactions = interactions
        self._shapes = shapes
        self._primitives = primitives
        self._vertices = vertices

        self._paths_components_built = True

    def _finalize_doppler_shift_compute(self) -> mi.Float:
        r"""
        Finalizes the computation of the Doppler shift of paths by applying the
        shift due to mobility of radio devices

        :return: Finalized Doppler shift [Hz]
        """

        paths_buffer = self._paths_buffer
        tx_ind = self._tx_ind
        rx_ind = self._rx_ind

        # Doppler shift due to transmitters mobility

        # Paths direction of departure
        k_tx = paths_buffer.k_tx
        # Transmitter velocities
        v_tx = dr.gather(mi.Vector3f, self._tx_velocities, tx_ind)
        # Doppler shift [Hz]
        tx_doppler = dr.dot(k_tx, v_tx)/self._wavelength

        # Doppler shift due to receivers mobility

        # Paths direction of arrival, flipped to point along the direction of
        # propagation of the incident wave
        k_rx = -paths_buffer.k_rx
        # Receiver velocities
        v_rx = dr.gather(mi.Vector3f, self._rx_velocities, rx_ind)
        # Doppler shift [Hz]
        rx_doppler = dr.dot(k_rx, v_rx)/self._wavelength

        doppler = paths_buffer.doppler + tx_doppler - rx_doppler
        return doppler

    def _reverse_direction(self,
                           t_list: list[mi.TensorXf]) -> list[mi.TensorXf]:
        r"""
        Reverses the direction of the paths

        Assumes that the tensors in `t_list` have the same shape.

        :param t_list: List of tensors to reverse the direction of the paths
        :return: List of tensors with the direction of the paths reversed
        """

        item = t_list[0]

        # Transpose the tensor using gather and scatter without for loops
        original_shape = item.shape
        num_items = len(original_shape)
        if num_items == 5:
            num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths = original_shape
            target_shape = [num_tx, num_tx_ant, num_rx, num_rx_ant, num_paths]
        elif num_items == 3:
            num_rx, num_tx, num_paths = original_shape
            target_shape = [num_tx, num_rx, num_paths]
            num_rx_ant = num_tx_ant = 1

        # Stop immediately if there are no paths
        if num_paths == 0:
            output_list = [dr.zeros(mi.TensorXf, target_shape) for _ in t_list]
            return output_list

        # Compute the indices for the gathering operation that implements
        # the transpose operation
        # Indices as we want them in the transposed tensor
        tx, txa, rx, rxa, p = dr.meshgrid(
            dr.arange(mi.UInt, num_tx),
            dr.arange(mi.UInt, num_tx_ant),
            dr.arange(mi.UInt, num_rx),
            dr.arange(mi.UInt, num_rx_ant),
            dr.arange(mi.UInt, num_paths),
            indexing='ij'
        )
        # Indices are computed to fit the tensor from which we are gathering
        gather_ind = rx*num_rx_ant*num_tx*num_tx_ant*num_paths + \
                        rxa*num_tx*num_tx_ant*num_paths + \
                        tx*num_tx_ant*num_paths + \
                        txa*num_paths + \
                        p

        # Gather the values from the original tensor using the new indices
        output_list = []
        for item in t_list:
            item_ = dr.gather(mi.Float, item.array, gather_ind)
            item_ = dr.reshape(mi.TensorXf, item_, target_shape)
            output_list.append(item_)
        return output_list
