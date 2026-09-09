#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Buffer to store paths during their computation"""

import mitsuba as mi
import drjit as dr
from typing import Union, Type, Tuple
import dataclasses

from sionna.rt.constants import InteractionType, INVALID_SHAPE,\
    INVALID_PRIMITIVE
from sionna.rt.utils import WedgeGeometry


class PathsBufferBase:
    """
    Base class for paths buffer, containing only the fields that need to be
    read from & written to in the same symbolic loop.
    When needed, we will create a copy of this object to avoid write-after-read.

    :param buffer_size: Size of the buffer
    :param max_depth: Maximum depth
    """

    def __init__(self, buffer_size: int, max_depth: int):
        # Size of the array
        # If `max_depth` is 0, we allocate one element to store paths data
        depth_dim_size = max(1, max_depth)

        self._max_depth = max_depth
        self._buffer_size = buffer_size
        self._depth_dim_size = depth_dim_size

        # Path vertices
        # The additional two entries along the depth dimension correspond to the
        # endpoints of the paths.
        self._vertices_x = dr.zeros(mi.TensorXf, [buffer_size, depth_dim_size])
        self._vertices_y = dr.zeros(mi.TensorXf, [buffer_size, depth_dim_size])
        self._vertices_z = dr.zeros(mi.TensorXf, [buffer_size, depth_dim_size])

        # Pointers to intersected shapes
        # Shapes pointers are stored as unsigned integers
        self._shapes = dr.full(mi.TensorXu, INVALID_SHAPE,
                                   [buffer_size, depth_dim_size])

        # Index of intersected primitive of the shape
        self._primitives = dr.full(mi.TensorXu, INVALID_PRIMITIVE,
                                   [buffer_size, depth_dim_size])

        # Used for a workaround required (for now) on the CUDA backend
        self._tensor_width_lit = dr.width(self._shapes.array)
        self._tensor_width = dr.opaque(mi.UInt32, self._tensor_width_lit)


    @staticmethod
    def from_paths_buffer(paths: "PathsBuffer"):
        # pylint: disable=protected-access
        """
        Creates a new basic paths buffer from an existing paths buffer by
        copying the relevant fields.
        This is useful to avoid reading & writing to the same arrays within
        a symbolic loop.
        """
        result = PathsBufferBase(paths.buffer_size, paths.max_depth)
        # Note: lazy copies
        result._vertices_x = dr.copy(paths._vertices_x)
        result._vertices_y = dr.copy(paths._vertices_y)
        result._vertices_z = dr.copy(paths._vertices_z)
        result._shapes = dr.copy(paths._shapes)
        result._primitives = dr.copy(paths._primitives)
        return result


    @property
    def max_depth(self):
        r"""Maximum depth

        :type: :py:class:`int`
        """
        return self._max_depth

    @property
    def depth_dim_size(self):
        r"""Size of the depth dimension of the arrays and tensors instantiated
        by this class. Equals to 1 if ``max_depth == 0`` or to ``max_depth``
        otherwise.

        :type: :py:class:`int`
        """
        return self._depth_dim_size

    @property
    def buffer_size(self):
        r"""Size of the buffer

        :type: :py:class:`int`
        """
        return self._buffer_size

    @property
    def vertices_x(self):
        r"""X coordinates of the paths' vertices

        :type: :py:class:`mi.TensorXf [buffer_size, depth_dim_size]`
        """
        return self._vertices_x

    @property
    def vertices_y(self):
        r"""Y coordinates of the paths' vertices

        :type: :py:class:`mi.TensorXf [buffer_size, depth_dim_size]`
        """
        return self._vertices_y

    @property
    def vertices_z(self):
        r"""Z coordinates of the paths' vertices

        :type: :py:class:`mi.TensorXf [buffer_size, depth_dim_size]`
        """
        return self._vertices_z

    @property
    def shapes(self):
        r"""Intersected shapes. Invalid shapes are represented by
        :data:`~sionna.rt.constants.INVALID_SHAPE`

        :type: :py:class:`mi.TensorXu [buffer_size, depth_dim_size]`
        """
        return self._shapes

    @property
    def primitives(self):
        r"""Intersected primitives. Invalid primitives are represented by
        :data:`~sionna.rt.constants.INVALID_PRIMITIVE`.

        :type: :py:class:`mi.TensorXu [buffer_size, depth_dim_size]`
        """
        return self._primitives

    def get_vertex(self, depth: mi.UInt, active: mi.Bool) -> mi.Point3f:
        r"""
        Gathers the coordinates of the vertices of the paths for the specified
        ``depth``

        :param depth: Depths
        :param active: Flags specifying active components

        :return: Coordinates of the paths vertices
        """

        vx = self._gather_depth(mi.Float, self._vertices_x, depth, active)
        vy = self._gather_depth(mi.Float, self._vertices_y, depth, active)
        vz = self._gather_depth(mi.Float, self._vertices_z, depth, active)

        return mi.Point3f(vx, vy, vz)

    def get_shape(self, depth: mi.UInt, active: mi.Bool) -> mi.UInt:
        r"""
        Gathers the indices to the intersected shapes for the specified
        ``depth``

        :param depth: Depths
        :param active: Flags specifying active components

        :return: Indices of the intersected shapes
        """

        return self._gather_depth(mi.UInt, self._shapes, depth, active)

    def get_primitive(self, depth: mi.UInt, active: mi.Bool) -> mi.UInt:
        r"""
        Gathers the indices to the intersected primitives for the specified
        ``depth``

        :param depth: Depths
        :param active: Flags specifying active components

        :return: Indices of the intersected primitives
        """

        return self._gather_depth(mi.UInt, self._primitives, depth, active)

    def get_primitive_props(self,
                            depth: mi.UInt,
                            return_normal: bool,
                            return_vertices: bool,
                            active: mi.Bool
        ) -> mi.Vector3f\
            | Tuple[mi.Point3f, mi.Point3f, mi.Point3f]\
            | Tuple[mi.Vector3f, mi.Point3f, mi.Point3f, mi.Point3f]:
        r"""
        Gathers primitive properties (normals and/or vertices) for the specified
        ``depth``

        :param depth: Depths
        :param return_normal: Whether to return face normals
        :param return_vertices: Whether to return vertex positions
        :param active: Flags specifying active components

        :return: Tuple containing requested primitive properties:
            - Face normal (if return_normal=True)
            - Vertex positions (if return_vertices=True)
        """

        valid = active & (depth > 0)

        # Gather shape pointer for the input depth and cast it to a mesh
        # pointer
        shape_int = self.get_shape(depth, valid)
        mesh_ptr = dr.reinterpret_array(mi.MeshPtr, shape_int)

        # Primitive index
        prim_ind = self.get_primitive(depth, valid)

        output = []
        # Normal
        if return_normal:
            normal = mesh_ptr.face_normal(prim_ind, valid)
            if not return_vertices:
                return normal
            output.append(normal)
        # Vertices
        if return_vertices:
            v_ind = mesh_ptr.face_indices(prim_ind, valid)
            output.append(mesh_ptr.vertex_position(v_ind.x, valid))
            output.append(mesh_ptr.vertex_position(v_ind.y, valid))
            output.append(mesh_ptr.vertex_position(v_ind.z, valid))
        return tuple(output)

    ###############################################
    # Internal methods
    ###############################################

    def _gather_depth(self,
                      dtype: Type[Union[mi.Float, mi.UInt, mi.Bool]],
                      tensor: mi.TensorXf | mi.TensorXu | mi.TensorXb,
                      depth: mi.UInt,
                      active: mi.Bool) -> mi.Float | mi.UInt | mi.Bool:
        r"""
        Gathers data from ``tensor`` for the specified ``depth``

        ``tensor`` is assumed to have shape `[buffer_size, depth_dim_size]`.

        :param dtype: Desired output Mitsuba type
        :param tensor: Tensor to gather from
        :param depth: Depths
        :param active: Flags specifying active components

        :return: Gathered values
        """

        valid = active & (depth > 0)

        ind = dr.arange(mi.UInt, self.buffer_size) * self.depth_dim_size \
              + depth - 1
        # There was a bug in the OptiX compiler stack which led to illegal
        # memory accesses. Although it was fixed in driver 580+, we keep this
        # workaround for backward compatibility with older drivers.
        assert dr.width(tensor.array) == self._tensor_width_lit
        ind &= (ind < self._tensor_width)

        return dr.gather(dtype, tensor.array, ind, valid)



class PathsBuffer(PathsBufferBase):
    r"""
    Class used to store the paths during their computation

    The output of a path solver is an instance of this class from which an
    instance of :class:`~sionna.rt.Paths` is built.

    :param buffer_size: Size of the buffer
    :param max_depth: Maximum depth
    :param diffraction: Whether diffraction is enabled
    """

    def __init__(self, buffer_size: int, max_depth: int, diffraction: bool):
        super().__init__(buffer_size, max_depth)

        self._diffraction = diffraction

        # Effective number of paths. This counter should be used to count the
        # number of paths effectively found by a solver.
        # Note that the buffer can be shrunk to this value using self.shrink()
        self._paths_counter = mi.UInt(0)

        # Set to True for if the path is valid
        self._valid = dr.full(dr.mask_t(mi.Float), False, buffer_size)

        # Index of the source from which the path originates
        self._src_indices = dr.zeros(mi.UInt, buffer_size)

        # Index of the target to which the path connects to
        self._tgt_indices = dr.zeros(mi.UInt, buffer_size)

        # Directions of departure and arrival as unit vectors.
        # As for the path vertices, the components are stored in separate
        # arrays, as Dr.Jit requires scatter targets to be flat arrays.
        self._k_tx_x = dr.zeros(mi.Float, buffer_size)
        self._k_tx_y = dr.zeros(mi.Float, buffer_size)
        self._k_tx_z = dr.zeros(mi.Float, buffer_size)
        #
        self._k_rx_x = dr.zeros(mi.Float, buffer_size)
        self._k_rx_y = dr.zeros(mi.Float, buffer_size)
        self._k_rx_z = dr.zeros(mi.Float, buffer_size)

        # Type of interaction (specular reflection, diffuse reflection, etc)
        self._interaction_types = dr.full(mi.TensorXu, InteractionType.NONE,
                                          [buffer_size, self.depth_dim_size])

        # Probabilities of the sampled interaction types
        # These are stored for field computation
        self._probs = dr.zeros(mi.TensorXf, [buffer_size, self.depth_dim_size])

        # Channel impulse response coefficients and delays are initialized to
        # `None`
        self._a = None
        self._tau = None
        # Doppler shifts of paths are initialized to `None`
        self._doppler = None

        # Diffracting edge properties
        # As diffraction is only supported for first order, we do not need to
        # store these properties for each depth.
        if diffraction:
            self._diffracting_wedges = \
                WedgeGeometry.build_with_size(buffer_size)
        else:
            self._diffracting_wedges = None

    @property
    def paths_counter(self):
        r"""Number of paths stored in the buffer

        :type: :py:class:`int`
        """
        return self._paths_counter

    def advance_paths_counter(self, add_n: mi.UInt32 | int) -> None:
        """Add the given number to the paths counter. Not atomic."""
        self._paths_counter += add_n

    @property
    def valid(self):
        r"""Flag indicating valid paths

        :type: :py:class:`mi.Bool`
        """
        return self._valid

    @valid.setter
    def valid(self, v):
        self._valid = v

    @property
    def source_indices(self):
        r"""Indices of the source from which the paths originates

        :type: :py:class:`mi.UInt`
        """
        return self._src_indices

    @property
    def target_indices(self):
        r"""Indices of the target to which the paths connect

        :type: :py:class:`mi.UInt`
        """
        return self._tgt_indices

    @property
    def k_tx(self):
        r"""Directions of departure as unit vectors

        :type: :py:class:`mi.Vector3f`
        """
        return mi.Vector3f(self._k_tx_x, self._k_tx_y, self._k_tx_z)

    @property
    def k_rx(self):
        r"""Directions of arrival as unit vectors, pointing from the target
        towards the direction from which the wave arrives, i.e., opposite to
        the direction of propagation of the incident wave

        :type: :py:class:`mi.Vector3f`
        """
        return mi.Vector3f(self._k_rx_x, self._k_rx_y, self._k_rx_z)

    @property
    def interaction_types(self):
        r"""Interactions types represented using
        :data:`~sionna.rt.constants.InteractionType`

        :type: :py:class:`mi.TensorXu [buffer_size, depth_dim_size]`
        """
        return self._interaction_types

    @property
    def diffracting_wedges(self):
        r"""Diffracting wedges

        :type: :py:class:`WedgeGeometry`
        """
        return self._diffracting_wedges

    @property
    def probs(self):
        r"""Probabilities of the sampled interaction types

        :type: :py:class:`mi.TensorXf [buffer_size, depth_dim_size]`
        """
        return self._probs

    @property
    def a(self):
        r"""Paths coefficients for every receive antenna pattern and transmit
        antenna pattern. ``a[n][m]`` stores the array of paths coefficients for
        the ``n`` th receive antenna pattern and the ``m`` th transmit antenna
        pattern

        :type: :py:class:`Tuple[mi.TensorXf, mi.TensorXf]`
        """
        return self._a

    @a.setter
    def a(self, v):
        self._a = v

    @property
    def tau(self):
        r"""Paths delays [s]

        :type: :py:class:`mi.Float`:
        """
        return self._tau

    @tau.setter
    def tau(self, v):
        self._tau = v

    @property
    def doppler(self):
        r""" Paths Doppler shifts [Hz]

        :type: :py:class:`mi.Float`
        """
        return self._doppler

    @doppler.setter
    def doppler(self, v):
        self._doppler = v

    def schedule(self) -> None:
        arrays = [
            self._max_depth,
            self._buffer_size,
            self._depth_dim_size,
            self._paths_counter,
            self._valid,
            self._src_indices,
            self._tgt_indices,
            self._k_tx_x,
            self._k_tx_y,
            self._k_tx_z,
            self._k_rx_x,
            self._k_rx_y,
            self._k_rx_z,
            self._interaction_types,
            self._vertices_x,
            self._vertices_y,
            self._vertices_z,
            self._shapes,
            self._primitives,
            self._probs,
            self._a,
            self._tau,
            self._doppler,
        ]
        if self._diffraction:
            arrays.append(self._diffracting_wedges)
        dr.schedule(*arrays)

    @dr.syntax
    def add_paths(self,
                  depth: mi.UInt,
                  indices: mi.UInt,
                  sample_data: int,
                  valid: mi.Bool,
                  tgt_index: mi.UInt,
                  k_tx: mi.Vector3f,
                  k_rx: mi.Vector3f,
                  active: mi.Bool) -> None:
        # pylint: disable=line-too-long
        r"""
        Adds paths to the buffer

        :param depth: Depths of the paths to add
        :param indices: Indices where to add paths in the buffer
        :param sample_data: Paths data to store
        :param valid: Flags indicating if the paths are valid, i.e., if their compute are finalized
        :param tgt_index: Targets to which the paths connect
        :param k_tx: Directions of departure of the paths as unit vectors
        :param k_rx: Directions of arrival of the paths as unit vectors,
            pointing opposite to the direction of propagation
        :param active: Flags specifying active paths. Inactive paths are not added.
        """

        depth_dim_size = self._depth_dim_size

        # Update the valid flag
        dr.scatter(self._valid, valid, indices, active=active)
        # Update the source index
        dr.scatter(self._src_indices, sample_data.src_indices, indices, active)
        # Update the target index
        dr.scatter(self._tgt_indices, tgt_index, indices, active)
        # Update directions of departure and arrival
        self._scatter_k(self._k_tx_x, self._k_tx_y, self._k_tx_z, k_tx,
                        indices, active)
        self._scatter_k(self._k_rx_x, self._k_rx_y, self._k_rx_z, k_rx,
                        indices, active)
        # Diffracting wedge
        if self._diffraction and (sample_data.diffracting_wedges is not None):
            dr.scatter(self._diffracting_wedges,
                       sample_data.diffracting_wedges,
                       indices, active)

        # Only add additional path data if `max_depth > 0`
        d = dr.ones(mi.UInt, dr.width(active))
        while d <= depth:
            interaction_types, shapes, primitives, vertices, probs \
                = sample_data.get(d, active=active)

            # Indices for updating the interaction type, shape, and primitive
            # arrays
            indices_t = indices*depth_dim_size + d - 1

            # Update interaction types
            dr.scatter(self._interaction_types.array, interaction_types,
                        indices_t, active)
            # Update shapes
            dr.scatter(self._shapes.array, shapes, indices_t, active)
            # Update primitives
            dr.scatter(self._primitives.array, primitives, indices_t, active)
            # Update vertices
            dr.scatter(self._vertices_x.array, vertices.x, indices_t,active)
            dr.scatter(self._vertices_y.array, vertices.y, indices_t,active)
            dr.scatter(self._vertices_z.array, vertices.z, indices_t,active)
            # Update probabilities
            dr.scatter(self._probs.array, probs, indices_t, active)

            d += 1

    def shrink(self) -> None:
        r"""
        Shrinks the buffer size to :attr:`~sionna.rt.PathsBuffer.paths_counter`

        Only the first :attr:`~sionna.rt.PathsBuffer.paths_counter` items are
        kept.
        """

        # Effective number of paths
        num_paths = dr.minimum(dr.max(self._paths_counter),
                               self._buffer_size)[0]

        depth_dim_size = self._depth_dim_size

        self._valid = dr.reshape(mi.Bool, self._valid, num_paths,
                                 shrink=True)
        self._src_indices = dr.reshape(mi.UInt, self._src_indices,
                                       num_paths, shrink=True)
        self._tgt_indices = dr.reshape(mi.UInt, self._tgt_indices,
                                       num_paths, shrink=True)
        if self._a is not None:
            a = []
            for a_ in self._a:
                a1 = []
                for a__ in a_:
                    a1.append(dr.reshape(mi.Complex2f, a__, num_paths,
                                         shrink=True))
                a.append(a1)
            self._a = a
        if self._tau is not None:
            self._tau = dr.reshape(mi.Float, self._tau, num_paths, shrink=True)
        if self._doppler is not None:
            self._doppler = dr.reshape(mi.Float, self._doppler, num_paths,
                                       shrink=True)
        self._k_tx_x = dr.reshape(mi.Float, self._k_tx_x,
                                  num_paths, shrink=True)
        self._k_tx_y = dr.reshape(mi.Float, self._k_tx_y,
                                  num_paths, shrink=True)
        self._k_tx_z = dr.reshape(mi.Float, self._k_tx_z,
                                  num_paths, shrink=True)
        self._k_rx_x = dr.reshape(mi.Float, self._k_rx_x,
                                  num_paths, shrink=True)
        self._k_rx_y = dr.reshape(mi.Float, self._k_rx_y,
                                  num_paths, shrink=True)
        self._k_rx_z = dr.reshape(mi.Float, self._k_rx_z,
                                  num_paths, shrink=True)
        self._interaction_types = mi.TensorXu(
            dr.reshape(mi.UInt, self._interaction_types.array,
                       num_paths*depth_dim_size, shrink=True),
            shape=(num_paths,depth_dim_size)
        )
        self._vertices_x = mi.TensorXf(
            dr.reshape(mi.Float, self._vertices_x.array,
                       num_paths*depth_dim_size, shrink=True),
            shape=(num_paths, depth_dim_size)
        )
        self._vertices_y = mi.TensorXf(
            dr.reshape(mi.Float, self._vertices_y.array,
                       num_paths*depth_dim_size, shrink=True),
            shape=(num_paths, depth_dim_size)
        )
        self._vertices_z = mi.TensorXf(
            dr.reshape(mi.Float, self._vertices_z.array,
                       num_paths*depth_dim_size, shrink=True),
            shape=(num_paths, depth_dim_size)
        )
        self._shapes = mi.TensorXu(
            dr.reshape(mi.UInt, self._shapes.array,
                       num_paths*depth_dim_size, shrink=True),
            shape=(num_paths, depth_dim_size)
        )
        self._primitives = mi.TensorXu(
            dr.reshape(mi.UInt, self._primitives.array,
                       num_paths*depth_dim_size, shrink=True),
            shape=(num_paths, depth_dim_size)
        )
        if self._diffraction:
            self._diffracting_wedges = WedgeGeometry(
            *[dr.reshape(type(getattr(self._diffracting_wedges, field.name)),
                          getattr(self._diffracting_wedges, field.name),
                          num_paths, shrink=True)
                for field in dataclasses.fields(self._diffracting_wedges)])
        self._probs = mi.TensorXf(
            dr.reshape(mi.Float, self._probs.array,
                       num_paths*depth_dim_size, shrink=True),
            shape=(num_paths, depth_dim_size)
        )

        self._buffer_size = num_paths
        self._paths_counter = num_paths
        self._tensor_width_lit = dr.width(self._shapes.array)
        self._tensor_width = dr.opaque(mi.UInt32, self._tensor_width_lit)

    def discard_invalid(self) -> None:
        r"""
        Discards invalid paths

        This function discards paths for which the entry in
        :attr:`~sionna.rt.PathsBuffer.valid` is set to `False`.
        """

        depth_dim_size = self._depth_dim_size
        self.schedule()

        # Indices of valid paths
        valid_ind = dr.compress(self._valid)

        # Number of valid paths
        num_valid_paths = dr.shape(valid_ind)[0]

        # Tensor indices for gathering the paths data
        depth_ind = dr.tile(dr.arange(mi.UInt, 0, depth_dim_size),
                            num_valid_paths)
        # Need to handle the case where no paths is found separately
        if num_valid_paths == 0:
            tensor_gind = []
        else:
            tensor_gind = dr.repeat(valid_ind, depth_dim_size)*depth_dim_size\
                            + depth_ind

        self._valid = dr.gather(mi.Bool, self._valid, valid_ind)
        self._src_indices = dr.gather(mi.UInt, self._src_indices, valid_ind)
        self._tgt_indices = dr.gather(mi.UInt, self._tgt_indices, valid_ind)
        if self._a is not None:
            a = []
            for a_ in self._a:
                a1 = []
                for a__ in a_:
                    a1.append(dr.gather(mi.Complex2f, a__, valid_ind))
                a.append(a1)
            self._a = a
        if self._tau is not None:
            self._tau = dr.gather(mi.Float, self._tau, valid_ind)
        if self._doppler is not None:
            self._doppler = dr.gather(mi.Float, self._doppler, valid_ind)
        self._k_tx_x = dr.gather(mi.Float, self._k_tx_x, valid_ind)
        self._k_tx_y = dr.gather(mi.Float, self._k_tx_y, valid_ind)
        self._k_tx_z = dr.gather(mi.Float, self._k_tx_z, valid_ind)
        self._k_rx_x = dr.gather(mi.Float, self._k_rx_x, valid_ind)
        self._k_rx_y = dr.gather(mi.Float, self._k_rx_y, valid_ind)
        self._k_rx_z = dr.gather(mi.Float, self._k_rx_z, valid_ind)
        self._interaction_types = mi.TensorXu(
            dr.gather(mi.UInt, self._interaction_types.array, tensor_gind),
            shape=(num_valid_paths, depth_dim_size)
        )
        self._vertices_x = mi.TensorXf(
            dr.gather(mi.Float, self._vertices_x.array, tensor_gind),
            shape=(num_valid_paths, depth_dim_size)
        )
        self._vertices_y = mi.TensorXf(
            dr.gather(mi.Float, self._vertices_y.array, tensor_gind),
            shape=(num_valid_paths, depth_dim_size)
        )
        self._vertices_z = mi.TensorXf(
            dr.gather(mi.Float, self._vertices_z.array, tensor_gind),
            shape=(num_valid_paths, depth_dim_size)
        )
        self._shapes = mi.TensorXu(
            dr.gather(mi.UInt, self._shapes.array, tensor_gind),
            shape=(num_valid_paths, depth_dim_size)
        )
        self._primitives = mi.TensorXu(
            dr.gather(mi.UInt, self._primitives.array, tensor_gind),
            shape=(num_valid_paths, depth_dim_size)
        )
        if self._diffraction:
            self._diffracting_wedges = dr.gather(
                WedgeGeometry,
                self._diffracting_wedges,
                valid_ind
            )
        self._probs = mi.TensorXf(
            dr.gather(mi.Float, self._probs.array, tensor_gind),
            shape=(num_valid_paths, depth_dim_size)
        )

        self._buffer_size = num_valid_paths
        self._paths_counter = num_valid_paths
        self._tensor_width_lit = dr.width(self._shapes.array)
        self._tensor_width = dr.opaque(mi.UInt32, self._tensor_width_lit)

    def get_interaction_type(self,
                             depth: mi.UInt,
                             active: mi.Bool) -> mi.UInt:
        r"""
        Gathers the interactions types for the specified ``depth``

        :param depth: Depths
        :param active: Flags specifying active components

        :return: Interactions types
        """

        return self._gather_depth(mi.UInt, self._interaction_types, depth,
                                  active)


    def get_prob(self, depth: mi.UInt, active: mi.Bool) -> mi.Float:
        r"""
        Gathers the probabilities of the sampled interaction types
        for the specified ``depth``

        :param depth: Depths
        :param active: Flags specifying active components
        """
        return self._gather_depth(mi.Float, self._probs, depth, active)

    def set_k_tx(self, k_tx: mi.Vector3f, active: mi.Bool) -> None:
        r"""
        Sets the directions of departure :attr:`~sionna.rt.PathsBuffer.k_tx`

        :param k_tx: Directions of departure as unit vectors
        :param active: Flags specifying active components
        """

        indices = dr.arange(mi.UInt, self.buffer_size)
        self._scatter_k(self._k_tx_x, self._k_tx_y, self._k_tx_z, k_tx,
                        indices, active)

    def set_k_rx(self, k_rx: mi.Vector3f, active: mi.Bool) -> None:
        r"""
        Sets the directions of arrival :attr:`~sionna.rt.PathsBuffer.k_rx`

        :param k_rx: Directions of arrival as unit vectors, pointing opposite
            to the direction of propagation of the incident wave
        :param active: Flags specifying active components
        """

        indices = dr.arange(mi.UInt, self.buffer_size)
        self._scatter_k(self._k_rx_x, self._k_rx_y, self._k_rx_z, k_rx,
                        indices, active)

    def set_interaction_type(self,
                             depth: mi.UInt,
                             value: mi.UInt,
                             active: mi.Bool) -> None:
        # pylint: disable=line-too-long
        r"""
        Sets the interactions types for the specified ``depth``

        :param depth: Depths
        :param value: Interactions types represented using :data:`~sionna.rt.constants.InteractionType`
        :param active: Flags specifying active components
        """

        self._scatter_depth(self._interaction_types, depth, value, active)

    def set_vertex(self,
                   depth: mi.UInt,
                   value: mi.Point3f,
                   active: mi.Bool) -> None:
        r"""
        Sets the coordinates of the vertices of the paths for the specified
        ``depth``

        :param depth: Depths
        :param value: Vertices
        :param active: Flags specifying active components
        """

        self._scatter_depth(self._vertices_x, depth, value.x, active)
        self._scatter_depth(self._vertices_y, depth, value.y, active)
        self._scatter_depth(self._vertices_z, depth, value.z, active)

    def set_shape(self,
                  depth: mi.UInt,
                  value: mi.UInt,
                  active: mi.Bool) -> None:
        r"""
        Sets the indices of the intersected shapes for the specified ``depth``

        Invalid intersections should be represented by
        :data:`~sionna.rt.constants.INVALID_SHAPE`

        :param depth: Depths
        :param value: Indices of the intersected shapes
        :param active: Flags specifying active components
        """

        self._scatter_depth(self._shapes, depth, value, active)

    def set_primitive(self,
                      depth: mi.UInt,
                      value: mi.UInt,
                      active: mi.Bool) -> None:
        r"""
        Sets the indices to the intersected primitives for the specified
        ``depth``

        :param depth: Depths
        :param value: Indices of intersected primitives
        :param active: Flags specifying active components
        """

        self._scatter_depth(self._primitives, depth, value, active)

    def set_prob(self,
                 depth: mi.UInt,
                 value: mi.Float,
                 active: mi.Bool) -> None:
        r"""
        Sets the probabilities of the sampled interaction types for
        the specified ``depth``

        :param depth: Depths
        :param value: Probabilities of the sampled interaction types
        :param active: Flags specifying active components
        """

        self._scatter_depth(self._probs, depth, value, active)

    def detach_geometry(self) -> None:
        r"""
        Detaches the arrays storing the paths geometries (i.e., vertices and
        directions of departure and arrival) from the automatic differentiation
        computational graph
        """

        self._vertices_x = dr.detach(self._vertices_x)
        self._vertices_y = dr.detach(self._vertices_y)
        self._vertices_z = dr.detach(self._vertices_z)
        self._k_tx_x = dr.detach(self._k_tx_x)
        self._k_tx_y = dr.detach(self._k_tx_y)
        self._k_tx_z = dr.detach(self._k_tx_z)
        self._k_rx_x = dr.detach(self._k_rx_x)
        self._k_rx_y = dr.detach(self._k_rx_y)
        self._k_rx_z = dr.detach(self._k_rx_z)
        if self._diffraction:
            self._diffracting_wedges = dr.detach(self._diffracting_wedges)

    ###############################################
    # Internal methods
    ###############################################

    @staticmethod
    def _scatter_k(k_x: mi.Float,
                   k_y: mi.Float,
                   k_z: mi.Float,
                   value: mi.Vector3f,
                   indices: mi.UInt,
                   active: mi.Bool) -> None:
        r"""
        Scatters the components of the direction vectors ``value`` into the
        arrays ``k_x``, ``k_y``, and ``k_z``

        :param k_x: Array storing the x components
        :param k_y: Array storing the y components
        :param k_z: Array storing the z components
        :param value: Direction vectors to insert
        :param indices: Indices at which to insert the direction vectors
        :param active: Flags specifying active components
        """

        dr.scatter(k_x, value.x, indices, active)
        dr.scatter(k_y, value.y, indices, active)
        dr.scatter(k_z, value.z, indices, active)

    def _scatter_depth(self,
                       tensor: mi.TensorXf | mi.TensorXu | mi.TensorXb,
                       depth: mi.UInt,
                       value: mi.Float | mi.UInt | mi.Bool,
                       active: mi.Bool) -> None:
        r"""
        Sets the items of ``tensor`` to ``value`` for the specified ``depth``

        ``tensor`` is assumed to have shape `[buffer_size, depth_dim_size]`.

        :param tensor: Tensor to update
        :param depth: Depths
        :param value: Value to insert in ``tensor``
        :param active: Flags specifying active components
        """

        valid = active & (depth > 0)
        ind = dr.arange(mi.UInt, self.buffer_size) * self.depth_dim_size \
              + depth - 1
        dr.scatter(tensor.array, value, ind, valid)
