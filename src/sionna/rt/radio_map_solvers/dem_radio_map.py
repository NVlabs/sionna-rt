"""Digital Elevation Model (DEM) radio map"""

import mitsuba as mi
import drjit as dr

from .radio_map import RadioMap
from sionna.rt.utils.geometry import triangulate_elevation
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from ..scene import Scene

class DemRadioMap(RadioMap):
    def __init__(self,
                 scene : "Scene",
                 elevation : mi.TensorXf,
                 cell_size : mi.Point2f,
                 center : mi.Point3f | None = None,
                 size : mi.Point2f | None = None):
  
        if center is None or size is None:
            # [min_x, min_y, min_z]
            scene_min = scene.mi_scene.bbox().min
            # In case of empty scene, bbox min is -inf
            scene_min = dr.select(dr.isinf(scene_min), -1.0, scene_min)
            # [max_x, max_y, max_z]
            scene_max = scene.mi_scene.bbox().max
            # In case of empty scene, bbox min is inf
            scene_max = dr.select(dr.isinf(scene_max), 1.0, scene_max)
            # Center and size
            if center is None:
                center = 0.5 * (scene_min + scene_max)
                center.z = 1.5
            if size is None:
                size = (scene_max - scene_min).xy
        size = mi.Point2f(size)
        center = mi.Point3f(center)
        cell_size = mi.Point2f(cell_size)

        if len(dr.shape(elevation)) != 2:
            raise ValueError("Elevation must be a 2D tensor.")
        else:
            elevation = mi.TensorXf(elevation)

        super().__init__(scene)
        # Number of cells
        self._size = size
        self._center = center
        self._cell_size = cell_size
        self._cells_per_dim = mi.Point2u(size / cell_size)

        # Builds the Mitsuba mesh modeling the measurement surface
        elevation_resolution = self._cells_per_dim + mi.Point2u(1, 1)
        self._elevation = self._resample_tensor(elevation, elevation_resolution)
        self._meas_surface = triangulate_elevation(self._elevation, center, size)

        cells_per_dim = self._cells_per_dim
        pathgain_shape = (self.num_tx, cells_per_dim.y[0], cells_per_dim.x[0])
        self._pathgain_map = dr.zeros(mi.TensorXf, pathgain_shape)

        # Copies to prevent data loss when repeatedly calling resample()
        self._original_pathgain_map = None
        self._original_elevation = self._elevation

    @property
    def measurement_surface(self):
        r"""Mitsuba Mesh corresponding to the
        radio map measurement surface

        :type: :py:class:`mi.Mesh`
        """
        return self._meas_surface

    @property
    def cells_count(self):
        r"""Total number of cells in the radio map

        :type: :py:class:`int`
        """
        cells_per_dim = self._cells_per_dim
        return cells_per_dim.x[0] * cells_per_dim.y[0]
    
    @property
    def cell_centers(self):
        r"""Positions of the centers of the cells in the global coordinate
        system.

        :type: :py:class:`mi.TensorXf [cells_per_dim_y, cells_per_dim_x, 3]`
        """
        cells_per_dim_x = self._cells_per_dim.x[0]
        cells_per_dim_y = self._cells_per_dim.y[0]
        u, v = dr.meshgrid(
            (dr.arange(mi.UInt, size=cells_per_dim_x) + 0.5) / cells_per_dim_x,
            (dr.arange(mi.UInt, size=cells_per_dim_y) + 0.5) / cells_per_dim_y
        )
        points = self._meas_surface.eval_parameterization(mi.Point2f(u, v))
        shape = (self._cells_per_dim.y[0], self._cells_per_dim.x[0], 3)
        return dr.reshape(mi.TensorXf, points, shape)
    
    @property
    def path_gain(self):
        r"""Path gains across the radio map from all transmitters

        :type: :py:class:`mi.TensorXf [num_tx, cells_per_dim_y, cells_per_dim_x]`
        """
        return self._pathgain_map

    def add(
        self,
        e_fields : mi.Vector4f,
        solid_angle : mi.Float,
        array_w : List[mi.Float],
        si_mp : mi.SurfaceInteraction3f,
        k_world : mi.Vector3f,
        tx_indices : mi.UInt,
        hit : mi.Bool
        ) -> None:
        r"""
        Adds the contribution of the rays that hit the measurement surface to 
        the radio maps

        The radio maps are updated in place.

        :param e_fields: Electric fields as real-valued vectors of dimension 4
        :param solid_angle: Ray tubes solid angles [sr]
        :param array_w: Weighting used to model the effect of the transmitter
            array
        :param si_mp: Informations about the interaction with the measurement
            surface
        :param k_world: Directions of propagation of the rays
        :param tx_indices: Indices of the transmitters from which the rays
            originate
        :param hit: Flags indicating if the rays hit the measurement surface
        """
        # Indices of the hit cells
        cell_ind = self._local_to_cell_ind(si_mp.uv)
        # Indices of the item in the tensor storing the radio maps
        tensor_ind = tx_indices * self.cells_count + cell_ind

        # Contribution to the path loss map
        a = dr.zeros(mi.Vector4f, 1)
        for e_field, aw in zip(e_fields, array_w):
            a += aw @ e_field
        a = dr.squared_norm(a)

        # Ray weight
        k_local = si_mp.to_local(k_world)
        cos_theta = dr.abs(k_local.z)
        w = solid_angle * dr.rcp(cos_theta)

        a *= w

        # Update the path loss map
        dr.scatter_add(target=self._pathgain_map.array,
                       value=a,
                       index=tensor_ind,
                       active=hit)

    def finalize(self) -> None:
        r"""Finalizes the computation of the radio map"""
        num_faces = 2 * self.cells_count
        cells_per_dim_x = self._cells_per_dim.x[0]
        cells_per_dim_y = self._cells_per_dim.y[0]
        # Note: this code is tightly coupled to triangulate_elevation()
        upper_index = dr.arange(mi.UInt, 0, num_faces // 2)
        lower_index = dr.arange(mi.UInt, num_faces // 2, num_faces)
        # Get the area of each cell as the sum of triangle areas
        upper_area = self._triangle_areas(upper_index)
        lower_area = self._triangle_areas(lower_index)
        cell_area = dr.reshape(dtype=mi.TensorXf,
                                value=upper_area + lower_area,
                                shape=(cells_per_dim_y, cells_per_dim_x))
        # Create and apply scaling factor
        scaling = dr.square(self._wavelength / (4 * dr.pi)) / cell_area
        self._pathgain_map *= scaling
        # Set original pathgain map for resampling later
        self._original_pathgain_map = self._pathgain_map

    @property
    def center(self):
        r"""Center of the radio map in the global coordinate system

        :type: :py:class:`mi.Point3f`
        """
        return self._center

    @property
    def size(self):
        r"""Size of the radio map [m]

        :type: :py:class:`mi.Point2f`
        """
        return self._size

    @property
    def elevation(self):
        r"""The elevation map used to create the measurement surface
        
        :type: :py:class:`mi.TensorXf` [cells_per_dim_y, cells_per_dim_x]
        """
        return self._elevation

    @property
    def cells_per_dim(self):
        r"""Number of cells per dimension

        :type: :py:class:`mi.Point2u`
        """
        return self._cells_per_dim

    def resample(self, cell_size: mi.Point2f):
        cell_size = mi.Point2f(cell_size)
        original_shape = mi.Point2f(self._original_pathgain_map.shape[1:])
        original_cell_size = self._size / original_shape
        # The new resolution must be at most as large as the provided elevation
        # data. Any more and the resampling leads to significant error. Resolve
        # this by increasing cell size or using a higher resolution DEM.
        if cell_size.x < original_cell_size.x:
            raise ValueError(f"`cell_size.x` must be greater than or equal to "
                             f"{original_cell_size.x[0]}.")
        if cell_size.y < original_cell_size.y:
            raise ValueError(f"`cell_size.y` must be greater than or equal to "
                             f"{original_cell_size.y[0]}.")

        res = mi.Point2u(dr.ceil(self._size / cell_size))
        if dr.all(res == original_shape):
            # Resampling to current size is a no-op
            return
        self._cells_per_dim = mi.Vector2u(res.x, res.y)

        # Resize the pathgain map
        pathgain_map = dr.zeros(mi.TensorXf, (self.num_tx, res.y[0], res.x[0]))
        for tx in range(self.num_tx):
            tmp = self._original_pathgain_map[tx, ...]
            pathgain_map[tx, ...] = self._resample_tensor(tmp, res).array
        self._pathgain_map = pathgain_map
        # Resize the stored elevation map
        self._elevation = self._resample_tensor(self._original_elevation, res)

    ###############################################
    # Internal methods
    ###############################################

    def _local_to_cell_ind(self, p_local : mi.Point2f) -> mi.Int:
        r"""
        Computes the indices of the hitted cells of the map from the local
        :math:`(x,y)` coordinates

        :param p_local: Coordinates of the intersected points in the
            measurement plane local frame

        :return: Cell indices in the flattened measurement plane
        """

        # Size of a cell in UV space
        cell_size_uv = mi.Vector2f(self._cells_per_dim)

        # Cell indices in the 2D measurement plane
        cell_ind = mi.Point2i(dr.floor(p_local * cell_size_uv))

        # Cell indices for the flattened measurement plane
        cell_ind = cell_ind[1] * self._cells_per_dim[0] + cell_ind[0]

        return cell_ind

    def _triangle_areas(self, tri_indices: mi.UInt):
        meas_surface = self._meas_surface
        prim_index = meas_surface.face_indices(tri_indices)
        v0 = meas_surface.vertex_position(prim_index[0])
        v1 = meas_surface.vertex_position(prim_index[1])
        v2 = meas_surface.vertex_position(prim_index[2])
        return dr.norm(dr.cross(v1 - v0, v2 - v0)) / 2


    def _resample_tensor(self, tensor: mi.TensorXf, res: mi.Point2u):
        if tensor.ndim != 2:
            raise ValueError("_resample_tensor requires tensor to be 2D.")
        # The resample() function needs this to be the scalar type
        res = mi.ScalarPoint2u(res.x[0], res.y[0])
        bitmap = mi.Bitmap(tensor, mi.Bitmap.PixelFormat.Y)
        resampled = mi.TensorXf(bitmap.resample(res))
        return dr.reshape(mi.TensorXf, resampled, res)
