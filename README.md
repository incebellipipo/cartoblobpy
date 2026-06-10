# cartoblobpy

cartoblobpy is a small Python library for working with 2D occupancy grids,
map images, and weighted graph representations for path planning. It can load
maps from PNG images or YAML map descriptions, convert between world and grid
coordinates, inspect occupancy values, plot maps, inflate obstacles, and build
NetworkX graphs from free space.

## Features

- Load maps from image files or YAML map descriptions.
- Work with start/goal markers embedded in map images.
- Convert between world coordinates and grid coordinates in ENU or NED frames.
- Query occupancy, free space, path validity, and distance to obstacles.
- Plot occupancy grids and named layers with Matplotlib.
- Build weighted graphs with NetworkX for path-planning workflows.
- Access bundled example maps and persisted results from `cartoblobpy.assets`.

## Installation

Install the package into your environment from the project root:

```powershell
pip install -e .
```

If you prefer a regular install:

```powershell
pip install .
```

The runtime dependencies are listed in `requirements.txt`.

## Quick Start

Load a bundled example map, inspect it, and plot it:

```python
from cartoblobpy.assets import get_map_path
from cartoblobpy.graph import Graph

graph = Graph()
graph.load_from_yaml(get_map_path("map012.yaml"))

print("grid shape:", graph.grid.shape)
print("resolution:", graph.resolution)
print("start:", graph.start)
print("goal:", graph.goal)

ax = graph.plot(show_start_goal=True, show_colorbar=True, cmap="gray_r")
```

If you already have a map image, load it directly:

```python
from cartoblobpy.graph import Graph

graph = Graph()
graph.load_from_image("path/to/map.png")
```

## Working With Maps

YAML map files are expected to define at least:

- `image`: path to the map image, absolute or relative to the YAML file.
- `resolution`: meters per pixel.
- `origin`: map origin as `[x, y, yaw]`.

Optional keys include:

- `start`: start position in grid coordinates.
- `goal`: goal position in grid coordinates.
- `layers`: named layer images, optionally with value mappings.

Typical operations after loading a map include:

```python
point = [2.5, 4.0]

graph.is_free(point)
graph.is_free_path([0.0, 0.0], [5.0, 5.0])
graph.value_at(point)
graph.distance_to_closest_obstacle(point)
graph.inflate_obstacles(radius=1.5)
graph.build_graph()
```

The graph object supports `ENU` and `NED` coordinate frames through the
`coordinate_frame` property.

## Bundled Assets

The `cartoblobpy.assets` module provides helpers for example data and saved
results:

- `list_example_maps()` lists the packaged YAML maps.
- `get_map_path(name)` returns the absolute path to a bundled map.
- `save_result(result, filename)` stores a pickle under `assets/results`.
- `load_result(filename)` loads a saved result.
- `list_available_results()` lists saved `.pkl` files.

## Running Tests

Run the unittest suite with Python on Windows (PowerShell):

```powershell
py -m unittest discover -s tests -p "test_*.py" -v
```

If `py` is unavailable, use your environment's Python:

```powershell
python -m unittest discover -s tests -p "test_*.py" -v
```

## Documentation

Sphinx documentation lives under `docs/`. Build it from the project root if you
want the generated HTML docs.

## License

cartoblobpy is distributed under the European Union Public Licence 1.2 (EUPL 1.2).
