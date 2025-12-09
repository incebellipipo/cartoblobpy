"""
Graph representation module for path planning applications.

This module provides functionality for loading, manipulating and representing
graphs from images for use in path planning algorithms.
"""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import networkx as nx
from itertools import product
import scipy
import yaml
import os


class Graph:
    """
    A graph representation for path planning.

    This class handles the creation and manipulation of a graph from an image,
    where obstacles, start points, and goal points are represented by different colors.
    """

    START_COLOR = np.array([0, 255, 0, 255])
    GOAL_COLOR = np.array([255, 0, 0, 255])
    OBSTRACLE_COLOR = np.array([0, 0, 0, 255])

    def __init__(self):
        """
        Initialize a new Graph instance.

        Creates an empty graph with default parameters for obstacle inflation
        and obstacle threshold.
        """

        self.__start = None
        self.__goal = None

        self.__grid = None
        self.__nodes = nx.Graph()

        self.__treshold = 0.6  # treshold for obstacles

        # Resolution of the grid in world coordinates (meters per pixel)
        self.__resolution = 1.0
        # Origin of the grid in world coordinates, (x, y, yaw)
        self.__origin = np.zeros(3)

        # Coordinate frame for interpreting world coordinates:
        # "ENU": world = [x_east, y_north]
        # "NED": world = [x_north, y_east]
        self.__frame = "ENU"

    def inflate_obstacles(self, radius, use_world_units=True):
        """
        Inflate obstacles in the grid to account for agent size.

        This uses a distance transform and creates a "soft" inflation
        where cost decays with distance to obstacles.

        :param radius: If ``use_world_units`` is ``True``: radius in meters; otherwise in pixels.
        :type radius: float
        :param use_world_units: Interpret ``radius`` as meters when ``True``; pixels when ``False``.
        :type use_world_units: bool
        :returns: The inflated occupancy grid with values in ``[0, 1]``.
        :rtype: numpy.ndarray
        :raises RuntimeError: If the grid is not initialized yet.
        """
        if self.__grid is None:
            raise RuntimeError(
                "Grid not initialized; load an image or YAML first.")

        if radius <= 0:
            # No inflation, just return the current grid
            return self.__grid

        # Binary obstacle mask: 1 = obstacle, 0 = free
        obstacle_mask = (self.__grid > self.__treshold).astype(np.uint8)

        # Distance transform from free space to nearest obstacle (in pixels)
        distance_pixels = scipy.ndimage.distance_transform_edt(
            1 - obstacle_mask)

        if use_world_units:
            # Convert distance and radius to meters
            distance = distance_pixels * self.__resolution
            radius_scale = radius  # meters
        else:
            distance = distance_pixels
            radius_scale = float(radius)  # pixels

        # Soft inflation: cost decays with distance.
        # For example: cost = exp(-d / R), with cost=1 at obstacle.
        # You can tune this if you want a different profile.
        with np.errstate(over="ignore"):
            inflated = np.exp(-distance / radius_scale)

        # Make sure actual obstacles stay at 1
        inflated = np.where(obstacle_mask, 1.0, inflated)

        # Clamp to [0, 1]
        inflated = np.clip(inflated, 0.0, 1.0)

        self.__grid = inflated
        return self.__grid

    def load_from_yaml(self, yaml_file):
        """
        Load a graph representation from a YAML file.

        The YAML file should contain:

        - ``image``: Path to the image file (absolute or relative to the YAML file).
        - ``resolution``: Resolution of the grid in meters per pixel.
        - ``origin``: Origin of the grid in world coordinates ``[x, y, yaw]``.

        Optional keys:

        - ``start``: Grid coordinates ``(row, column)`` for the start point.
        - ``goal``: Grid coordinates ``(row, column)`` for the goal point.

        :param yaml_file: Path to the YAML file to load.
        :type yaml_file: str
        :raises ValueError: If required keys are missing or invalid.
        """
        with open(yaml_file, 'r') as file:
            config = yaml.safe_load(file)

        if 'image' not in config:
            raise ValueError("YAML file must contain 'image' key.")

        image_file = config['image']

        # If the image path is not absolute, make it relative to the YAML file
        if not os.path.isabs(image_file):
            yaml_dir = os.path.dirname(os.path.abspath(yaml_file))
            image_file = os.path.join(yaml_dir, image_file)

        if 'resolution' in config:
            self.__resolution = config['resolution']

        if 'origin' in config:
            self.__origin = np.array(config['origin'])

        if 'start' in config:
            self.__start = tuple(config['start'])

        if 'goal' in config:
            self.__goal = tuple(config['goal'])

        self.load_from_image(image_file)

    def load_from_image(self, image_file):
        """
        Load a graph representation from an image file.

        Color coding in the image:

        - Green pixels: start location
        - Red pixels: goal location
        - Black pixels: obstacles
        - Transparent/white pixels: free space

        :param image_file: Path to the image file to load.
        :type image_file: str
        """
        # Load png and convert transparent pixels to white
        img = Image.open(image_file).convert(
            "RGBA").transpose(Image.FLIP_TOP_BOTTOM)

        # Convert to NumPy array
        grid = np.array(img)

        self.__grid = grid[:, :, 3] / 255.0  # Alpha channel

        # Find all pixels for start (green) and goal (red)
        start_mask = np.all(grid[:, :, :3] == [0, 255, 0], axis=-1)
        goal_mask = np.all(grid[:, :, :3] == [255, 0, 0], axis=-1)

        # Mark start and goal areas as free space in occupancy map
        self.__grid[start_mask] = 0
        self.__grid[goal_mask] = 0

        # Compute center of mass for start point if exists
        start_points = np.where(start_mask)
        if len(start_points[0]) > 0:
            start_r = np.mean(start_points[0])
            start_c = np.mean(start_points[1])
            self.__start = self.grid_to_world((start_r, start_c))

        # Compute center of mass for goal point if exists
        goal_points = np.where(goal_mask)
        if len(goal_points[0]) > 0:
            goal_r = np.mean(goal_points[0])
            goal_c = np.mean(goal_points[1])
            self.__goal = self.grid_to_world((goal_r, goal_c))

    def build_graph(self):
        """
        Build a NetworkX graph from the grid representation.

        Creates nodes for all non-obstacle cells and adds edges between
        adjacent cells. Edge weights are the Euclidean step distance scaled by
        local occupancy cost.

        :returns: A populated graph of free cells with weighted edges.
        :rtype: networkx.Graph
        """

        self.__nodes.clear()

        rows, cols = self.__grid.shape

        for r, c in product(range(rows), range(cols)):

            if self.__grid[r, c] > self.__treshold:
                continue

            self.__nodes.add_node((r, c))

            for dr, dc in (p for p in product([1, 0, -1], repeat=2) if p != (0, 0)):

                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    self.__nodes.add_edge(
                        (r, c), (nr, nc), weight=(
                            np.sqrt(dr**2 + dc**2) * (1 + self.__grid[nr, nc]))
                    )

    @property
    def nodes(self):
        """
        Get the graph nodes.

        :returns: The graph representation of the environment.
        :rtype: networkx.Graph
        """
        return self.__nodes

    @property
    def start(self):
        """
        Get the start point coordinates.

        :returns: Grid coordinates ``(row, column)`` of the start point, or ``None`` if not set.
        :rtype: tuple[int, int] | None
        """
        return self.__start

    @property
    def grid(self):
        """
        Get the grid representation of the environment.

        :returns: 2D array where values represent obstacle probability (``0``=free, ``1``=obstacle).
        :rtype: numpy.ndarray
        """
        return self.__grid

    @property
    def goal(self):
        """
        Get the goal point coordinates in world coordinates.

        :returns: World coordinates ``[x, y]`` of the goal point in the selected frame, or ``None`` if not set.
        :rtype: numpy.ndarray | None
        """
        return self.__goal

    @goal.setter
    def goal(self, goal):
        """
        Set the goal point coordinates.

        :param goal: World coordinates ``(x, y)`` for the goal point.
        :type goal: array_like
        """
        self.__goal = goal

    @start.setter
    def start(self, start):
        """
        Set the start point coordinates.

        :param start: Grid coordinates ``(row, column)`` for the start point.
        :type start: tuple[int, int]
        """
        self.__start = start

    @property
    def resolution(self):
        """
        Get the resolution of the grid in world coordinates (meters per pixel).

        :returns: Resolution in meters per pixel.
        :rtype: float
        """
        return self.__resolution

    @resolution.setter
    def resolution(self, resolution):
        """
        Set the resolution of the grid in world coordinates (meters per pixel).

        :param resolution: Resolution in meters per pixel.
        :type resolution: float
        """
        self.__resolution = resolution

    @property
    def origin(self):
        """
        Get the origin of the grid in world coordinates (x, y, yaw).

        :returns: Origin as a numpy array ``[x, y, yaw]``.
        :rtype: numpy.ndarray
        """
        return self.__origin

    @origin.setter
    def origin(self, origin):
        """
        Set the origin of the grid in world coordinates (x, y, yaw).

        :param origin: Origin as a numpy array ``[x, y, yaw]``.
        :type origin: array_like
        """
        self.__origin = np.array(origin)

    @property
    def width(self):
        """
        Get the real width of the grid in world coordinates.

        :returns: Width in meters.
        :rtype: float
        """
        return self.__grid.shape[1] * self.__resolution

    @property
    def height(self):
        """
        Get the real height of the grid in world coordinates.

        :returns: Height in meters.
        :rtype: float
        """
        return self.__grid.shape[0] * self.__resolution

    @property
    def shape(self):
        """
        Get the real size of the grid in world coordinates.

        :returns: Size as a tuple ``(width, height)`` in meters.
        :rtype: tuple[float, float]
        """
        return (self.width, self.height)

    @property
    def occupancy_threshold(self):
        """
        Get the occupancy threshold for obstacle detection.

        :returns: Occupancy threshold value.
        :rtype: float
        """
        return self.__treshold

    @property
    def coordinate_frame(self):
        """
        Get the coordinate frame.

        :returns: "ENU" or "NED".
        :rtype: str
        """
        return self.__frame

    @coordinate_frame.setter
    def coordinate_frame(self, frame):
        """
        Set the coordinate frame.

        :param frame: "ENU" or "NED".
        :type frame: str
        :raises ValueError: If the frame is not one of "ENU" or "NED".
        """
        frame = frame.upper()
        if frame not in ("ENU", "NED"):
            raise ValueError("coordinate_frame must be 'ENU' or 'NED'")
        self.__frame = frame

    def world_to_grid(self, world_coords):
        """
        Transform world coordinates to grid coordinates.

        In ENU frame: ``world_coords = [x_east, y_north]``.
        In NED frame: ``world_coords = [x_north, y_east]``.

        Internally, the map plane is treated as ``[x_east, y_north]``, and grid indices are:
        ``row ≈ y_north / resolution`` and ``col ≈ x_east / resolution``.

        :param world_coords: World coordinates ``[x, y]`` in the selected ``coordinate_frame``.
        :type world_coords: array_like
        :returns: Grid coordinates ``[row, column]``.
        :rtype: numpy.ndarray
        :raises ValueError: If ``world_coords`` is not a 2-element vector.
        """
        world_coords = np.asarray(world_coords, dtype=float)
        if world_coords.shape[0] != 2:
            raise ValueError(
                "world_coords must be a 2-element array-like [x, y].")

        ox, oy, oyaw = self.__origin

        # Convert world (frame) -> internal EN ([x_east, y_north])
        if self.__frame == "ENU":
            wx, wy = world_coords[0], world_coords[1]
            ox_en, oy_en = ox, oy
        else:  # NED: world = [x_north, y_east]
            wx, wy = world_coords[1], world_coords[0]  # [x_east, y_north]
            ox_en, oy_en = oy, ox

        translated = np.array([wx - ox_en, wy - oy_en])

        # Rotate into map frame (counter-clockwise rotation)
        c, s = np.cos(-oyaw), np.sin(-oyaw)
        rotation_matrix = np.array([[c, -s], [s, c]])
        rotated = rotation_matrix @ translated

        # Map plane coordinates in meters
        x_east_m, y_north_m = rotated[0], rotated[1]

        # Convert to grid indices (row, col)
        row = y_north_m / self.__resolution
        col = x_east_m / self.__resolution

        return np.array([row, col])

    def grid_to_world(self, grid_coords):
        """
        Transform grid coordinates to world coordinates.

        In ENU frame: returns ``[x_east, y_north]``.
        In NED frame: returns ``[x_north, y_east]``.

        :param grid_coords: Grid coordinates ``[row, column]``.
        :type grid_coords: array_like
        :returns: World coordinates ``[x, y]`` in the selected ``coordinate_frame``.
        :rtype: numpy.ndarray
        :raises ValueError: If ``grid_coords`` is not a 2-element vector.
        """
        grid_coords = np.asarray(grid_coords, dtype=float)
        if grid_coords.shape[0] != 2:
            raise ValueError(
                "grid_coords must be a 2-element array-like [row, col].")

        row, col = grid_coords[0], grid_coords[1]
        ox, oy, oyaw = self.__origin

        # Grid indices -> internal EN coordinates in meters
        x_east_m = col * self.__resolution
        y_north_m = row * self.__resolution

        # Rotate back to world EN frame
        c, s = np.cos(oyaw), np.sin(oyaw)
        rotation_matrix = np.array([[c, -s], [s, c]])
        rotated = rotation_matrix @ np.array([x_east_m, y_north_m])

        # Add origin in EN
        if self.__frame == "ENU":
            wx_en = rotated[0] + ox
            wy_en = rotated[1] + oy
            # ENU world is just EN
            world_x, world_y = wx_en, wy_en
        else:
            wx_en = rotated[0] + oy  # note swap
            wy_en = rotated[1] + ox
            # NED world = [x_north, y_east] = [y_north_en, x_east_en]
            world_x = wy_en  # north
            world_y = wx_en  # east

        return np.array([world_x, world_y])

    def is_free_path(self, point1, point2):
        """
        Check if the straight line path between two world points is obstacle-free.

        :param point1: World coordinates of the first point ``[x, y]``.
        :type point1: array_like
        :param point2: World coordinates of the second point ``[x, y]``.
        :type point2: array_like
        :returns: ``True`` if the path is free of obstacles, ``False`` otherwise.
        :rtype: bool
        """
        # Convery world to grid
        pg1 = self.world_to_grid(point1).astype(int)
        pg2 = self.world_to_grid(point2).astype(int)

        # Check out of bounds
        rows, cols = self.__grid.shape
        if (pg1[0] < 0 or pg1[0] >= rows or pg1[1] < 0 or pg1[1] >= cols or
                pg2[0] < 0 or pg2[0] >= rows or pg2[1] < 0 or pg2[1] >= cols):
            return False

        # Get all points in the line using Bresenham's algorithm, so implement it here
        from .utils import bresenham

        line_points = bresenham(pg1[0], pg1[1], pg2[0], pg2[1])
        for r, c in line_points:
            if self.__grid[r, c] > self.__treshold:
                return False

        return True

    def distance_to_closest_obstacle(self, world_point):
        """
        Calculate the distance from a world point to the closest obstacle.

        :param world_point: World coordinates of the point ``[x, y]``.
        :type world_point: array_like
        :returns: Distance to the closest obstacle in meters.
        :rtype: float
        """
        # Convert world to grid
        pg = self.world_to_grid(world_point).astype(int)

        # Compute distance transform (distance to nearest obstacle)
        obstacle_mask = (self.__grid > self.__treshold).astype(np.uint8)
        distance_map = scipy.ndimage.distance_transform_edt(1 - obstacle_mask)

        # Get distance in pixels and convert to meters
        distance_pixels = distance_map[pg[0], pg[1]]
        distance_meters = distance_pixels * self.__resolution

        return distance_meters

    def plot(self, ax=None, show_start_goal=True, show_colorbar=False, **imshow_kwargs) -> plt.Axes:
        """
        Plot the occupancy grid in real-world coordinates.

        The horizontal axis is East [m], the vertical axis is North [m].
        This convention is fixed; the ``coordinate_frame`` only changes how
        world coordinates map into this plot.

        :param ax: Axes to draw on. If ``None``, a new figure and axes are created.
        :type ax: matplotlib.axes.Axes | None
        :param show_start_goal: Plot start and goal points if available.
        :type show_start_goal: bool
        :param show_colorbar: Add a colorbar for occupancy values.
        :type show_colorbar: bool
        :param imshow_kwargs: Extra keyword arguments passed to ``plt.imshow`` (e.g., ``cmap="gray"``).
        :type imshow_kwargs: dict
        :returns: The axes with the plot.
        :rtype: matplotlib.axes.Axes
        :raises RuntimeError: If the grid is not initialized yet.
        """
        if self.__grid is None:
            raise RuntimeError(
                "Grid not initialized; load an image or YAML first.")

        if ax is None:
            fig, ax = plt.subplots()

        height, width = self.__grid.shape

        # Real-world extents in meters (map plane)
        extent = [0.0, self.width, 0.0, self.height]

        imshow_defaults = dict(origin="lower", extent=extent)
        imshow_defaults.update(imshow_kwargs)

        # Create a masked array where values of 0 are invalid (transparent)
        grid_to_plot = np.ma.masked_where(self.__grid == 0, self.__grid)

        # Plot the masked array instead of the raw grid
        im = ax.imshow(grid_to_plot, **imshow_defaults)

        start_grid = self.world_to_grid(self.__start) if self.__start is not None else None
        goal_grid = self.world_to_grid(self.__goal) if self.__goal is not None else None

        # Start and goal markers (converted to world EN plane)
        if show_start_goal:
            if start_grid is not None:
                r, c = start_grid
                # grid_to_world expects [row, col]
                w_start = self.grid_to_world(np.array([r, c]))
                # Convert world to internal EN for plotting
                if self.__frame == "ENU":
                    x_plot, y_plot = w_start[0], w_start[1]
                else:  # NED
                    # world = [x_north, y_east] -> EN = [x_east, y_north]
                    x_plot, y_plot = w_start[1], w_start[0]
                ax.plot(x_plot, y_plot, "go", label="start")

            if goal_grid is not None:
                r, c = goal_grid
                w_goal = self.grid_to_world(np.array([r, c]))
                if self.__frame == "ENU":
                    x_plot, y_plot = w_goal[0], w_goal[1]
                else:
                    x_plot, y_plot = w_goal[1], w_goal[0]
                ax.plot(x_plot, y_plot, "ro", label="goal")

        if self.__frame == "NED":
            ax.set_xlabel(r"$y$, East [m]")
            ax.set_ylabel(r"$x$, North [m]")
        else:
            ax.set_xlabel(r"$x$, East [m]")
            ax.set_ylabel(r"$y$, North [m]")

        if show_start_goal and ((start_grid is not None) or (goal_grid is not None)):
            ax.legend()

        if show_colorbar:
            plt.colorbar(im, ax=ax, label="Occupancy / cost")

        return ax

    def is_free(self, world_point):
        """
        Check if a world point is in free space.

        :param world_point: World coordinates of the point ``[x, y]``.
        :type world_point: array_like
        :returns: ``True`` if the point is free; ``False`` if it is an obstacle or out of bounds.
        :rtype: bool
        """
        # Convert world to grid
        pg = self.world_to_grid(world_point).astype(int)

        # check out of bounds
        if pg[0] < 0 or pg[0] >= self.__grid.shape[0] or pg[1] < 0 or pg[1] >= self.__grid.shape[1]:
            return False

        if self.__grid[pg[0], pg[1]] > self.__treshold:
            return False
        return True
