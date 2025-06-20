"""Digital Elevation Model (DEM) radio map"""

import mitsuba as mi
import drjit as dr

from .mesh_radio_map import MeshRadioMap
from sionna.rt.utils.geometry import triangulate_elevation
from typing import override, TYPE_CHECKING

if TYPE_CHECKING:
    from ..scene import Scene

class DemRadioMap(MeshRadioMap):
    def __init__(self,
                 scene : "Scene",
                 elevation: mi.TensorXf,
                 center : mi.Point3f | None = None,
                 orientation : mi.Point3f | None = None,
                 size : mi.Point2f | None = None):

        # Check the properties of the rectangle defining the radio map's base
        if (center is None) and (size is None) and (orientation is None):
            # Default value for center: Center of the scene
            # Default value for the scale: Just enough to cover all the scene
            # with axis-aligned edges of the rectangle
            # [min_x, min_y, min_z]
            scene_min = scene.mi_scene.bbox().min
            # In case of empty scene, bbox min is -inf
            scene_min = dr.select(dr.isinf(scene_min), -1.0, scene_min)
            # [max_x, max_y, max_z]
            scene_max = scene.mi_scene.bbox().max
            # In case of empty scene, bbox min is inf
            scene_max = dr.select(dr.isinf(scene_max), 1.0, scene_max)
            # Center and size
            center = 0.5 * (scene_min + scene_max)
            center.z = 1.5
            size = (scene_max - scene_min).xy
            # Set the orientation to default value
            orientation = dr.zeros(mi.Point3f, 1)
        elif ((center is None) or (size is None) or (orientation is None)):
            raise ValueError("If one of `center`, `orientation`," \
                             " or `size` is not None, then all of them" \
                             " must not be None.")
        else:
            center = mi.Point3f(center)
            orientation = mi.Point3f(orientation)
            size = mi.Point2f(size)

        if len(dr.shape(elevation)) != 2:
            raise ValueError("Elevation must be a 2D tensor.")
        else:
            elevation = mi.TensorXf(elevation)

        # Number of cells
        cells_per_dim = mi.Point2u(elevation.shape[1], elevation.shape[0])

        # Builds the Mitsuba mesh modeling the measurement surface
        meas_surface = triangulate_elevation(elevation)
        center = mi.Point3f(center)
        orientation = mi.Point3f(orientation)
        orientation_deg = orientation * 180. / dr.pi
        size = mi.Point2f(size)
        to_world = mi.Transform4f().translate(center) \
                                .rotate([0., 0., 1.], orientation_deg.x) \
                                .rotate([0., 1., 0.], orientation_deg.y) \
                                .rotate([1., 0., 0.], orientation_deg.z) \
                                .scale([0.5 * size.x, 0.5 * size.y, 1])
        params = mi.traverse(meas_surface)
        # Apply the to_world transformation
        vertices = dr.reshape(mi.Point3f, params["vertex_positions"], (3, -1))
        vertices = to_world @ vertices
        params["vertex_positions"] = dr.ravel(vertices)

        params.update()
        super().__init__(scene, meas_surface=meas_surface)

        self._elevation = elevation
        self._cells_per_dim = cells_per_dim
        self._center = center
        self._orientation = orientation
        self._size = size

    @property
    def center(self):
        r"""Center of the radio map in the global coordinate system

        :type: :py:class:`mi.Point3f`
        """
        return self._center

    @property
    def orientation(self):
        r"""Orientation of the radio map :math:`(\alpha, \beta, \gamma)`
        specified through three angles corresponding to a 3D rotation as defined
        in :eq:`rotation`. An orientation of :math:`(0,0,0)` corresponds to a
        radio map that is parallel to the XY plane.

        :type: :py:class:`mi.Point3f`
        """
        return self._orientation

    @property
    def size(self):
        r"""Size of the radio map [m]

        :type: :py:class:`mi.Point2f`
        """
        return self._size

    @property
    def elevation(self):
        r"""The elevation map used to create the measurement surface
        
        :type: :py:class:`mi.TensorXf` [samples_per_dim_y, samples_per_dim_x]
        """
        return self._elevation

    @property
    @override
    def cells_count(self):
        r"""Total number of cells in the radio map

        :type: :py:class:`int`
        """
        cells_per_dim = self._cells_per_dim
        return cells_per_dim.x[0] * cells_per_dim.y[0]

    @property
    def cells_per_dim(self):
        r"""Number of cells per dimension

        :type: :py:class:`mi.Point2u`
        """
        return self._cells_per_dim

    @property
    @override
    def cell_centers(self):
        r"""Positions of the centers of the cells in the global coordinate
        system.

        :type: :py:class:`mi.TensorXf [cells_per_dim_y, cells_per_dim_x, 3]`
        """
        # TODO create UV coordinates in a unit grid with the correct number of cells
        grid = ...
        points = self._meas_surface.eval_parameterization(grid)
        shape = (self._cells_per_dim.y[0], self._cells_per_dim.x[0], 3)
        return dr.reshape(mi.TensorXf, points, shape)


    @property
    @override
    def path_gain(self):
        r"""Path gains across the radio map from all transmitters

        :type: :py:class:`mi.TensorXf [num_tx, cells_per_dim_y, cells_per_dim_x]`
        """
        return self._pathgain_map

    def finalize(self) -> None:
        r"""Finalizes the computation of the radio map"""
        super().finalize()
        # [num_tx, 2 * cells_per_dim_y * cells_per_dim_x]
        pathgain_map = self._pathgain_map
        cells_per_dim_x = self._cells_per_dim.x[0]
        cells_per_dim_y = self._cells_per_dim.y[0]
        grid_shape = (self.num_tx, cells_per_dim_y, cells_per_dim_x)
        pathgain_map_grid = dr.zeros(mi.TensorXf, shape=grid_shape)

        num_faces = len(pathgain_map.array)
        upper_indices = dr.arange(mi.UInt, start=0, stop=num_faces // 2)
        lower_indices = dr.arange(mi.UInt, start=num_faces // 2, stop=num_faces)
        # [num_tx * cells_per_dim_y * cells_per_dim_x]
        all_grid_indices = dr.arange(mi.UInt, size=len(pathgain_map_grid.array))
        # [num_tx * cells_per_dim_y * cells_per_dim_x]
        upper_face_values = dr.gather(mi.Float, pathgain_map.array, upper_indices)
        # [num_tx * cells_per_dim_y * cells_per_dim_x]
        lower_face_values = dr.gather(mi.Float, pathgain_map.array, lower_indices)
        # TODO make this work for multiple transmitters
        dr.scatter_add(target=pathgain_map_grid.array,
                       value=upper_face_values,
                       index=all_grid_indices,
                       active=~dr.isnan(upper_face_values))
        dr.scatter_add(target=pathgain_map_grid.array,
                       value=lower_face_values,
                       index=all_grid_indices,
                       active=~dr.isnan(lower_face_values))
        # [num_tx * cells_per_dim_y * cells_per_dim_x]
        count = dr.zeros(mi.TensorXu, grid_shape)
        dr.scatter_add(count.array, 1, all_grid_indices, ~dr.isnan(upper_face_values))
        dr.scatter_add(count.array, 1, all_grid_indices, ~dr.isnan(lower_face_values))
        pathgain_map_grid /= count
        self._pathgain_map = pathgain_map_grid

    def resample(self, resolution: mi.Point2u):
        resolution = mi.Point2u(resolution)
        self._cells_per_dim = resolution
        shape = (resolution.y[0], resolution.x[0])
        # Resize the pathgain map
        bitmap = mi.Bitmap(self._pathgain_map, mi.Bitmap.PixelFormat.Y)
        resampled = mi.TensorXf(bitmap.resample(resolution))
        self._pathgain_map = dr.reshape(mi.TensorXf, resampled, shape)
        # Resize the stored elevation map
        bitmap = mi.Bitmap(self._elevation, mi.Bitmap.PixelFormat.Y)
        resampled = mi.TensorXf(bitmap.resample(resolution))
        self._elevation = dr.reshape(mi.TensorXf, resampled, shape)