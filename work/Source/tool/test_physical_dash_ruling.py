import unittest

import numpy as np

from complete_stroke_guard import bend_change
from physical_dash_ruling import register_observed_dashes, audit_dash_rows


class PhysicalDashRulingTests(unittest.TestCase):
    def test_rigid_dashes_can_form_a_straight_row_without_filling_gaps(self):
        height, width = 80, 160
        basis = np.array([width-1, height-1])
        centers_x = [20, 50, 80, 110]
        centers_y = [20, 25, 30, 25]
        observed_jitter = np.array([
            0, 0, 1, 1, -1, -1, 1,
            1, -1, -1, 1, 0, -1, 1,
            -1, 0, 0, 0, -1, -1, 1,
            1, 1, 0, 1, -1, 0, 1,
        ])
        contrast = np.zeros((height, width), np.float32)
        samples = []
        for dash, (cx, cy) in enumerate(zip(centers_x, centers_y)):
            contrast[cy-1:cy+2, cx-3:cx+4] = 20
            samples.extend((x, cy+observed_jitter[dash*7+x-(cx-3)])
                           for x in range(cx-3, cx+4))
        stroke = dict(axis=0, source_points=(np.asarray(samples)/basis).tolist())
        owners, registration = register_observed_dashes(
            contrast, [stroke], [width, height],
        )
        self.assertEqual(registration['material_owner_count'], 4)
        self.assertEqual(registration['original_dash_pixels'], 4*3*7)
        self.assertEqual(registration['rows'][0]['directly_matched_trace_samples'], len(samples))
        self.assertEqual(registration['rows'][0]['observed_dash_bodies'], 4)
        self.assertEqual(np.count_nonzero(contrast[:, 30:40]), 0)

        # One continuous backward field gives each observed dash a constant
        # vertical translation; the field interpolates only in blank stock.
        knots_x = [0, 16, 24, 46, 54, 76, 84, 106, 114, 159]
        knots_shift = [0, 5, 5, 0, 0, -5, -5, 0, 0, 0]
        x = np.arange(width)
        shift = np.interp(x, knots_x, knots_shift)
        y = np.arange(height)
        up = np.clip((y-10)/5, 0, 1)
        down = np.clip((45-y)/5, 0, 1)
        vertical = up*up*(3-2*up)*down*down*(3-2*down)
        xx, yy = np.meshgrid(x, y)
        field = np.stack([xx/basis[0], (yy-shift[None, :]*vertical[:, None])/basis[1]], -1)
        mesh = dict(rows=height, columns=width, grid=field.reshape(-1, 2).tolist())
        row = audit_dash_rows(mesh, owners, registration)[0]
        self.assertEqual(row['original_shape_pass'], 4)
        self.assertLess(row['original_shape_worst_rigid_error_px'], .03)
        self.assertGreater(row['before']['p90_px'], 2)
        self.assertLess(row['after']['p90_px'], 1e-7)
        # The legacy comparison rejects this beneficial inter-dash movement,
        # demonstrating why it cannot be the only line-quality gate.
        original = np.asarray(samples, float)
        moved = original.copy()
        moved[:, 1] += np.interp(original[:, 0], knots_x, knots_shift)
        self.assertFalse(bend_change(original, moved)['accepted'])


if __name__ == '__main__':
    unittest.main()
