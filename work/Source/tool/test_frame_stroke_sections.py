import unittest
import cv2
import numpy as np
from frame_stroke_sections import frame_sections


class FrameStrokeTests(unittest.TestCase):
    def page(self):
        gray = np.full((240, 400), 230, np.uint8)
        x = np.arange(40, 351)
        y = (80 + 9 * ((x - 200) / 160) ** 2).round().astype(int)
        cv2.polylines(gray, [np.c_[x, y].astype(np.int32)], False, 70, 1)
        for xx in (40, 350):
            cv2.line(gray, (xx, 30), (xx, 160), 60, 2)
        guide = np.c_[x[80:230] / 399, y[80:230] / 239]
        perpendicular = [np.array([[xx / 399, 30 / 239], [xx / 399, 160 / 239]])
                         for xx in (40, 350)]
        return gray, np.full_like(gray, 255), guide, perpendicular

    def test_complete_curved_frame_has_real_centers_and_both_crossings(self):
        gray, safe, guide, crossing = self.page()
        found = frame_sections(gray, safe, [guide], crossing, np.zeros_like(gray))
        self.assertEqual(len(found), 1)
        self.assertGreater(len(found[0]['points']), len(guide))
        x, y = np.rint(found[0]['points'] * [399, 239]).astype(int).T
        self.assertTrue((gray[y, x] < 230).all())
        self.assertNotEqual(found[0]['crossing_indices'][0], found[0]['crossing_indices'][1])

    def test_word_crossing_cannot_supply_missing_rule_ink(self):
        gray, safe, guide, crossing = self.page()
        mask = np.zeros_like(gray);mask[:, 180:230] = 1
        self.assertEqual(frame_sections(gray, safe, [guide], crossing, mask), [])

    def test_single_crossing_or_wrong_existing_section_cannot_qualify(self):
        gray, safe, guide, crossing = self.page()
        self.assertEqual(frame_sections(gray, safe, [guide], crossing[:1], np.zeros_like(gray)), [])
        self.assertEqual(frame_sections(gray, safe, [guide + [0, .04]], crossing, np.zeros_like(gray)), [])



if __name__ == '__main__':
    unittest.main()
