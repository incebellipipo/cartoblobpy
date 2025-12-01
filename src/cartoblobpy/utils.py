

def bresenham(x0, y0, x1, y1):
    """Bresenham's Line Algorithm
    Produces a list of tuples from start and end.

    :param x0: Starting x coordinate.
    :type x0: int
    :param y0: Starting y coordinate.
    :type y0: int
    :param x1: Ending x coordinate.
    :type x1: int
    :param y1: Ending y coordinate.
    :type y1: int
    :returns: List of points in the line from ``(x0, y0)`` to ``(x1, y1)``.
    :rtype: list[tuple[int, int]]
    """
    points = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    while True:
        points.append((x0, y0))
        if x0 == x1 and y0 == y1:
            break
        err2 = err * 2
        if err2 > -dy:
            err -= dy
            x0 += sx
        if err2 < dx:
            err += dx
            y0 += sy

    return points