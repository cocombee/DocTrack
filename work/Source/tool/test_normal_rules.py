import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

import normal_rule_candidate as normal
from printed_rule_sections import measure_sections


class NormalRuleTests(unittest.TestCase):
    def test_contradictory_fixed_ink_cannot_be_overwritten(self):
        self.assertIsNone(normal.ordered_knots([
            (0, 0, 0, 0), (1, 1, 1, 1),
            (.2, .3, .25, .35), (.27, .4, .27, .4),
        ]))

    def test_compatible_fixed_bands_merge_without_warping_ink(self):
        dest, source = normal.ordered_knots([
            (0, 0, 0, 0), (1, 1, 1, 1),
            (.2, .3, .2, .3), (.27, .4, .27, .4),
        ])
        np.testing.assert_array_equal(dest, source)

    def test_word_mask_must_match_measurement_coordinates(self):
        with self.assertRaises(ValueError):
            measure_sections(np.ones((100, 200), np.uint8),
                             word_mask=np.zeros((200, 100), np.uint8))

    def test_vertical_correction_preserves_words_and_established_horizontal_rules(self):
        for position in (40, 700):
            with self.subTest(position=position), tempfile.TemporaryDirectory() as folder:
                gray = np.full((400, 800), 235, np.uint8)
                for x in (250, 500, 790):
                    cv2.line(gray, (x, 20), (x, 380), 70, 1)
                horizontal = []
                for y in (20, 120, 250, 380):
                    cv2.line(gray, (20, y), (790, y), 70, 1)
                    horizontal.append(np.c_[np.linspace(20 / 799, 790 / 799, 101),
                                            np.full(101, y / 399)])
                cv2.putText(gray, 'LOCKED WORDS', (100, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, .6, 20, 2)
                cv2.putText(gray, '12345', (330, 220),
                            cv2.FONT_HERSHEY_SIMPLEX, .6, 20, 2)
                y = np.linspace(50, 350, 301)
                x = position + 6 * np.sin(y / 400 * 2 * np.pi)
                guide = np.c_[x / 799, y / 399]
                cv2.polylines(gray, [np.rint(np.c_[x, y]).astype(np.int32)], False, 70, 1)
                path = Path(folder) / 'page.png'
                cv2.imwrite(str(path), cv2.cvtColor(gray, cv2.COLOR_GRAY2BGRA))
                u, v = np.meshgrid(np.linspace(0, 1, 33), np.linspace(0, 1, 33))
                base = {'rows': 33, 'columns': 33,
                        'grid': np.stack([u, v], -1).reshape(-1, 2).tolist(),
                        'method': 'Accepted prior surface', 'established_rulers':
                        [{'normal_axis': 1, 'points': horizontal[1].tolist()}]}
                audit = []
                with patch.object(normal.rows.accepted.accepted, 'circular_landmarks', return_value=[]), \
                        patch.object(normal.p.reference, 'barcode_rulers', return_value=[]):
                    result = normal.refine(path, base, selected_guide=guide, audit_log=audit)
                self.assertTrue(result is not base, str(audit))
                self.assertIs(result['preserved_mesh'], base)
                evidence = result['normal_rule_audit']
                self.assertLess(evidence['full_rule_error_px'], .1)
                self.assertLess(evidence['fixed_ink_shift_px'], .01)
                self.assertLess(evidence['horizontal_normal_shift_px'], .01)
                self.assertLess(evidence['established_normal_shift_px'], .01)
                grid = np.asarray(result['grid']).reshape(result['rows'], result['columns'], 2)
                np.testing.assert_allclose(grid[:, 0, 0], 0, atol=1e-12)
                np.testing.assert_allclose(grid[:, -1, 0], 1, atol=1e-9)


if __name__ == '__main__':
    unittest.main()
