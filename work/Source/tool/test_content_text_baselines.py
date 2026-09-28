import unittest
import cv2
import numpy as np

from content_text_baselines import capture_text_baselines
from content_plane_evidence import _measure


class ContentTextBaselineTests(unittest.TestCase):
    def row(self, *, curved=False, blank=False):
        gray = np.full((640, 800), 240, np.uint8)
        words = []
        for start in (160, 240, 320):
            bottom = 120 + .02 * start
            quad = np.array([[start-2, bottom-18], [start+48, bottom-17],
                             [start+48, bottom+2], [start-2, bottom+1]])
            if curved:
                quad[:, 1] = [102, 102, 145, 145]
            words.append(dict(text='AB', confidence=.99,
                              points=(quad / [800, 640]).tolist()))
            for x in (start+4, start+30):
                y = round(120 + .02*x)
                if curved:
                    y = round(118 + 12*((x-260)/100)**2)
                if not blank:
                    gray[y-13:y+1, x:x+7] = 45
        return gray, np.full_like(gray, 255), words

    def test_native_boxes_only_associate_actual_sparse_glyph_baseline(self):
        gray, safe, words = self.row()
        before = gray.copy()
        traces = capture_text_baselines(gray, safe, words)
        self.assertEqual(len(traces), 1)
        trace = traces[0]
        self.assertEqual(len(trace['points']), 6)
        self.assertEqual(trace['glyph_baseline_support']['native_word_count'], 3)
        measured, reason = _measure(trace, np.array([800, 640]), 0)
        self.assertIsNone(reason)
        self.assertIsNotNone(measured)
        self.assertLess(abs(trace['measured_angle_radians'] - np.arctan(.02)), .003)
        np.testing.assert_array_equal(gray, before)

    def test_boxes_with_no_ink_or_unsafe_stock_cannot_supply_geometry(self):
        gray, safe, words = self.row(blank=True)
        self.assertEqual(capture_text_baselines(gray, safe, words), [])
        gray, safe, words = self.row()
        self.assertEqual(capture_text_baselines(gray, np.zeros_like(safe), words), [])

    def test_straight_native_boxes_cannot_hide_curved_measured_glyphs(self):
        gray, safe, words = self.row(curved=True)
        traces = capture_text_baselines(gray, safe, words)
        self.assertEqual(len(traces), 1)
        measured, reason = _measure(traces[0], np.array([800, 640]), 0)
        self.assertIsNone(measured)
        self.assertEqual(reason, 'curved trace')

    def test_ambiguous_duplicate_words_do_not_supply_independent_row(self):
        gray, safe, words = self.row()
        words.append(words[1].copy())
        self.assertEqual(capture_text_baselines(gray, safe, words), [])

    def test_invalid_raster_pair_is_rejected(self):
        gray, safe, words = self.row()
        with self.assertRaises(ValueError):
            capture_text_baselines(gray, safe[:10], words)


if __name__ == '__main__':
    unittest.main()
