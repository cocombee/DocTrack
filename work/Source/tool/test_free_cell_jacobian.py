import unittest

import numpy as np

from free_cell_jacobian import repair_free_cells


class FreeCellJacobianTest(unittest.TestCase):
    def setUp(self):
        u, v = np.meshgrid(np.linspace(0, 1, 5), np.linspace(0, 1, 5))
        self.field = np.stack([u, v], axis=-1)
        self.field[2, 2, 0] = .7
        self.fixed = np.ones((5, 5), bool)
        self.fixed[2, 2] = False

    def test_near_threshold_repair_moves_only_free_stock(self):
        input_field = self.field.copy()
        result, audit = repair_free_cells(self.field, self.fixed, [21, 21])
        self.assertTrue(audit['accepted'], audit)
        self.assertGreaterEqual(audit['final_minimum'], .204)
        self.assertLessEqual(audit['maximum_free_node_move_px'], .05)
        self.assertEqual(audit['maximum_fixed_node_move_px'], 0)
        np.testing.assert_array_equal(result[self.fixed], input_field[self.fixed])
        np.testing.assert_array_equal(self.field, input_field)

    def test_no_free_stock_refuses_without_changing_field(self):
        fixed = np.ones((5, 5), bool)
        result, audit = repair_free_cells(self.field, fixed, [21, 21])
        self.assertFalse(audit['accepted'])
        np.testing.assert_array_equal(result, self.field)

    def test_far_below_threshold_refuses_without_changing_field(self):
        folded = self.field.copy()
        folded[2, 2, 0] = .8
        result, audit = repair_free_cells(folded, self.fixed, [21, 21])
        self.assertFalse(audit['accepted'])
        self.assertEqual(audit['steps'], 0)
        np.testing.assert_array_equal(result, folded)


if __name__ == '__main__':
    unittest.main()
