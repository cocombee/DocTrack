"""Printed table rules must not acquire unrelated content outside their span."""
import unittest
from pathlib import Path
import cv2
import numpy as np
from printed_line_mesh import extract_rules

class RuleSpanTests(unittest.TestCase):
    def test_partial_parallel_rules_keep_their_supported_span(self):
        gray = np.full((400, 640), 230, np.uint8)
        for y in (200, 300):
            cv2.line(gray, (300, y), (620, y), 20, 1)
        rules = extract_rules(gray)
        self.assertEqual(len(rules), 2)
        for p in rules:
            self.assertGreater(p[:, 0].min(), .45)
            self.assertGreater(p[:, 0].max(), .94)
            self.assertLess(np.ptp(p[:, 1]), .002)

    def test_fingerprint_is_not_an_extension_of_table_rule(self):
        path = Path(__file__).resolve().parents[3] / 'Scanner Integration/Stable Measurement Candidate 49/Image 4 Preliminary.png'
        rgba = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        self.assertIsNotNone(rgba)
        alpha = rgba[:, :, 3:4] / 255
        bgr = (rgba[:, :, :3] * alpha + 255 * (1 - alpha)).astype(np.uint8)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        self.assertIsNotNone(gray)
        rules = extract_rules(gray)
        self.assertTrue(rules)
        for p in rules:
            # The middle rule exists only right of the vertical label column.
            if .46 < np.mean(p[:, 1]) < .55:
                self.assertGreater(p[:, 0].min(), .38)

if __name__ == '__main__':
    unittest.main()
