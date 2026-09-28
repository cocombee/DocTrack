import copy
import unittest

import numpy as np

from composite_mesh_inverse import inverse_retained
from composite_mesh_sampling import MeshSampler
from material_pixel_footprints import audit, capture, paired_points
from physical_character_material import (
    IDENTITY, partition_initial_material_by_components, relative_guard_owners,
)


class PhysicalCharacterMaterialTests(unittest.TestCase):
    def test_one_cross_line_ocr_owner_splits_without_losing_weak_original_ink(self):
        height, width = 80, 120
        basis = np.array([width-1, height-1])
        mask = np.zeros((height, width), np.uint8)
        mask[20:23, 20:24] = 1
        mask[40:43, 22:26] = 1
        upper = np.argwhere(mask[20:23, 20:24]) + [20, 20]
        lower = np.argwhere(mask[40:43, 22:26]) + [40, 22]
        centers = np.c_[np.r_[upper[:, 1], lower[:, 1], 24, 70],
                        np.r_[upper[:, 0], lower[:, 0], 21, 70]]
        # One OCR owner incorrectly contains both physical rows; the isolated
        # original pixel remains registered even though it has no mask label.
        original = capture(IDENTITY, [centers/basis], [width, height])
        saved = copy.deepcopy(original)
        parts, report = partition_initial_material_by_components(
            original, mask, first_owner=500,
        )
        self.assertEqual(report['physical_component_owners'], 3)
        self.assertEqual(report['weak_centers_assigned_to_observed_components'], 1)
        self.assertEqual(report['isolated_original_centers'], 1)
        self.assertEqual(report['unique_original_pixel_centers'], len(centers))
        grouped = [np.rint(paired_points(owner)[0][::5]*basis).astype(int)
                   for owner in parts]
        self.assertEqual(sorted(len(group) for group in grouped), [1, 12, 13])
        np.testing.assert_array_equal(np.unique(np.vstack(grouped), axis=0),
                                      np.unique(centers, axis=0))
        self.assertEqual(original, saved)

    def test_partition_rejects_late_registration_and_relative_guard_keeps_origin(self):
        height, width = 80, 120
        basis = np.array([width-1, height-1])
        mask = np.zeros((height, width), np.uint8)
        mask[30:34, 40:45] = 1
        y, x = np.where(mask)
        original = capture(IDENTITY, [np.c_[x, y]/basis], [width, height])
        shifted = dict(rows=2, columns=2,
                       grid=[[.01, 0], [1, 0], [.01, 1], [1, 1]])
        late = capture(shifted, [np.c_[x, y]/basis], [width, height])
        with self.assertRaisesRegex(ValueError, 'initial, unmodified'):
            partition_initial_material_by_components(late, mask)
        components, _ = partition_initial_material_by_components(original, mask)
        saved = copy.deepcopy(components)
        relative = relative_guard_owners(shifted, components)
        source, _ = paired_points(components[0])
        prior, reference = paired_points(relative[0])
        np.testing.assert_allclose(prior, inverse_retained(MeshSampler(shifted), source), atol=1e-10)
        np.testing.assert_array_equal(prior, reference)
        self.assertTrue(audit(IDENTITY, relative)[0]['accepted'])
        self.assertEqual(components, saved)


if __name__ == '__main__':
    unittest.main()
