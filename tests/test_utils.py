import unittest

from cartoblobpy.utils import bresenham


class TestBresenham(unittest.TestCase):
    def test_same_point(self):
        self.assertEqual(bresenham(0, 0, 0, 0), [(0, 0)])

    def test_horizontal_line(self):
        pts = bresenham(0, 0, 3, 0)
        self.assertEqual(pts, [(0, 0), (1, 0), (2, 0), (3, 0)])

    def test_vertical_line(self):
        pts = bresenham(0, 0, 0, 3)
        self.assertEqual(pts, [(0, 0), (0, 1), (0, 2), (0, 3)])

    def test_diagonal_line(self):
        pts = bresenham(0, 0, 3, 3)
        self.assertEqual(pts, [(0, 0), (1, 1), (2, 2), (3, 3)])

    def test_steep_line(self):
        pts = bresenham(0, 0, 2, 5)
        # Ensure start/end and monotonicity
        self.assertEqual(pts[0], (0, 0))
        self.assertEqual(pts[-1], (2, 5))
        # Coordinates should be within bounding box
        for x, y in pts:
            self.assertGreaterEqual(x, 0)
            self.assertLessEqual(x, 2)
            self.assertGreaterEqual(y, 0)
            self.assertLessEqual(y, 5)


if __name__ == "__main__":
    unittest.main()
