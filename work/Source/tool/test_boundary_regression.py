import unittest
from boundary_regression import errors, sample_boundary, topology_issues


def polygon(vertices):
    points = [dict(zip(('x', 'y'), p)) for p in vertices]
    return [{'start': p, 'end': points[(i+1) % len(points)]} for i, p in enumerate(points)]


class BoundaryRegressionTest(unittest.TestCase):
    def test_long_line_middle_is_checked(self):
        straight = polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        indented = polygon([(0, 0), (.5, .2), (1, 0), (1, 1), (0, 1)])
        score = errors(sample_boundary(straight, 1000, 1000), sample_boundary(indented, 1000, 1000))
        self.assertGreater(score['p95_px'], 100)

    def test_inserting_collinear_points_does_not_change_score(self):
        a = polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        b = polygon([(0, 0), (.2, 0), (.5, 0), (1, 0), (1, 1), (0, 1)])
        self.assertLess(errors(sample_boundary(a, 1000, 1000), sample_boundary(b, 1000, 1000))['max_px'], 1e-8)

    def test_crossing_is_blocked(self):
        self.assertIn('self_intersection', topology_issues(polygon([(0, 0), (1, 1), (0, 1), (1, 0)]), 400, 400))

    def test_normal_rectangle_has_no_topology_issues(self):
        self.assertEqual(topology_issues(polygon([(0, 0), (1, 0), (1, 1), (0, 1)]), 400, 400), [])


if __name__ == '__main__':
    unittest.main()
