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


VERMILLION = (213, 94, 0, 255)
BLUE_GREEN = (0, 158, 115, 255)
SKY_BLUE = (86, 180, 233, 255)
REDDISH_PURPLE = (204, 121, 167, 255)


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

        self.__infilation = 2 # pixels
        self.__treshold = 0.6 # treshold for obstacles

        # Resolution of the grid in world coordinates (meters per pixel)
        self.__resolution = 1.0
        # Origin of the grid in world coordinates, (x, y, yaw)
        self.__origin = np.zeros(3)

    def inflate_obstacles(self, radius=1):
        """
        Inflate obstacles in the grid to account for agent size.

        :param radius: The inflation radius in pixels. Zero disables inflation.
        :returns: The grid with inflated obstacles if radius=0, otherwise None.
        """
        if radius == 0:
            return self.__grid

        # Convert to binary mask: Obstacles = 1, Free space = 0
        obstacle_mask = ( self.__grid > 0.5).astype(np.uint8)

        # Compute distance transform (distance to nearest obstacle)
        distance_map = scipy.ndimage.distance_transform_edt(1 - obstacle_mask)

        # Apply a gradient effect using an exponential or linear decay
        inflated_map = np.exp(-distance_map / radius)

        # apply obstacle mask to inflated map to make sure obstacles are not inflated
        inflated_map = np.where(obstacle_mask, 1, inflated_map)

        self.__grid = inflated_map

    def load_from_yaml(self, yaml_file):
        """
        Load a graph representation from a YAML file.
        The YAML file should contain:
        - image: Path to the image file.
        - resolution: Resolution of the grid in meters per pixel.
        - origin: Origin of the grid in world coordinates (x, y, yaw).

        The YAML file can also contain:
        - start: (row, column) coordinates for the start point.
        - goal: (row, column) coordinates for the goal point.

        :param yaml_file: Path to the YAML file to load.
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

        The image should use specific colors to represent:
        - Green pixels: start location
        - Red pixels: goal location
        - Black pixels: obstacles
        - Transparent/white pixels: free space

        :param image_file: Path to the image file to load.
        """
        # Load png and convert transparent pixels to white
        img = Image.open(image_file).convert("RGBA").transpose(Image.FLIP_TOP_BOTTOM)

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
            start_r = int(np.mean(start_points[0]))
            start_c = int(np.mean(start_points[1]))
            self.__start = (start_r, start_c)

        # Compute center of mass for goal point if exists
        goal_points = np.where(goal_mask)
        if len(goal_points[0]) > 0:
            goal_r = int(np.mean(goal_points[0]))
            goal_c = int(np.mean(goal_points[1]))
            self.__goal = (goal_r, goal_c)

    def build_graph(self):
        """
        Build a networkx graph from the grid representation.

        Creates nodes for all non-obstacle cells and adds edges between adjacent cells.
        Edge weights are calculated based on distance and proximity to obstacles.
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
                        (r, c), (nr, nc), weight=(np.sqrt(dr**2 + dc**2) * (1 + self.__grid[nr, nc]))
                    )



    @property
    def nodes(self):
        """
        Get the graph nodes.

        :returns: The graph representation of the environment.
        """
        return self.__nodes

    @property
    def start(self):
        """
        Get the start point coordinates.

        :returns: (row, column) coordinates of the start point, or None if not set.
        """
        return self.__start

    @property
    def grid(self):
        """
        Get the grid representation of the environment.

        :returns: 2D array where values represent obstacle probability (0=free, 1=obstacle).
        """
        return self.__grid

    @property
    def goal(self):
        """
        Get the goal point coordinates.

        :returns: (row, column) coordinates of the goal point, or None if not set.
        """
        return self.__goal

    @goal.setter
    def goal(self, goal):
        """
        Set the goal point coordinates.

        :param goal: (row, column) coordinates for the goal point.
        """
        self.__goal = goal

    @start.setter
    def start(self, start):
        """
        Set the start point coordinates.

        :param start: (row, column) coordinates for the start point.
        """
        self.__start = start

    @property
    def resolution(self):
        """
        Get the resolution of the grid in world coordinates (meters per pixel).

        :returns: Resolution in meters per pixel.
        """
        return self.__resolution

    @resolution.setter
    def resolution(self, resolution):
        """
        Set the resolution of the grid in world coordinates (meters per pixel).

        :param resolution: Resolution in meters per pixel.
        """
        self.__resolution = resolution

    @property
    def origin(self):
        """
        Get the origin of the grid in world coordinates (x, y, yaw).

        :returns: Origin as a numpy array [x, y, yaw].
        """
        return self.__origin

    @origin.setter
    def origin(self, origin):
        """
        Set the origin of the grid in world coordinates (x, y, yaw).

        :param origin: Origin as a numpy array [x, y, yaw].
        """
        self.__origin = np.array(origin)

    @property
    def real_width(self):
        """
        Get the real width of the grid in world coordinates.

        :returns: Width in meters.
        """
        return self.__grid.shape[1] * self.__resolution

    @property
    def real_height(self):
        """
        Get the real height of the grid in world coordinates.

        :returns: Height in meters.
        """
        return self.__grid.shape[0] * self.__resolution

    @property
    def real_size(self):
        """
        Get the real size of the grid in world coordinates.

        :returns: Size as a tuple (width, height) in meters.
        """
        return (self.real_width, self.real_height)

    @property
    def occupancy_threshold(self):
        """
        Get the occupancy threshold for obstacle detection.

        :returns: Occupancy threshold value.
        """
        return self.__treshold


    def world_to_grid(self, world_coords):
        """
        Transform world coordinates to grid coordinates

        :param world_coords: World coordinates as a numpy array [x, y].
        :returns: Grid coordinates as a numpy array [row, column].
        """
        # Extract origin components
        ox, oy, oyaw = self.__origin

        # Translate
        translated = np.array([world_coords[0] - ox, world_coords[1] - oy])

        # Rotate (counter-clockwise rotation matrix)
        c, s = np.cos(-oyaw), np.sin(-oyaw)
        rotation_matrix = np.array([[c, -s], [s, c]])
        rotated = rotation_matrix @ translated

        # Scale to grid coordinates
        grid_coords = rotated / self.__resolution

        return grid_coords

    def grid_to_world(self, grid_coords):
        """
        Transform grid coordinates to world coordinates

        :param grid_coords: Grid coordinates as a numpy array [row, column].
        :returns: World coordinates as a numpy array [x, y].
        """
        # Extract origin components
        ox, oy, oyaw = self.__origin

        # Scale to meters
        scaled = np.array(grid_coords) * self.__resolution

        # Rotate (clockwise rotation matrix - inverse of counter-clockwise)
        c, s = np.cos(oyaw), np.sin(oyaw)
        rotation_matrix = np.array([[c, -s], [s, c]])
        rotated = rotation_matrix @ scaled

        # Translate
        world_coords = np.array([rotated[0] + ox, rotated[1] + oy])

        return world_coords

    def is_free_path(self, point1, point2):
        """
        Check if the straight line path between two points is free of obstacles.

        :param point1: World coordinates of the first point as a numpy array [x, y].
        :param point2: World coordinates of the second point as a numpy array [x, y].
        :returns: True if the path is free of obstacles, False otherwise.
        """
        # Convery world to grid
        pg1 = self.world_to_grid(point1).astype(int)
        pg2 = self.world_to_grid(point2).astype(int)

        # Get all points in the line using Bresenham's algorithm, so implement it here
        from cartoblobpy.utils import bresenham

        line_points = bresenham(pg1[0], pg1[1], pg2[0], pg2[1])
        for r, c in line_points:
            if self.__grid[r, c] > self.__treshold:
                return False

        return True