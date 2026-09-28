import unittest

import cv2
import numpy as np

from content_plane_evidence import estimate_printed_plane
from content_stroke_ownership import connected_rule_traces


class StrokeOwnershipTest(unittest.TestCase):
    def scene(self):
        return np.full((400, 400), 240, np.uint8), np.full((400, 400), 255, np.uint8), np.zeros((400, 400), np.uint8)

    def trace(self, x, y):
        return dict(kind='rule', points=np.c_[x, np.broadcast_to(y, np.shape(x))] / 399)

    def test_observed_gap_merges_fragments_without_changing_original_samples(self):
        gray, safe, words = self.scene()
        cv2.line(gray, (40, 100), (340, 100), 80, 2)
        traces = [self.trace(np.arange(40, 150), 100), self.trace(np.arange(162, 341), 100)]
        joined, links = connected_rule_traces(traces, gray, safe, words, 0)
        self.assertEqual(len(joined), 1)
        self.assertEqual(joined[0]['source_indices'], [0, 1])
        self.assertEqual(len(links), 1)
        for trace in traces:
            self.assertTrue(all(any(np.array_equal(p, q) for q in joined[0]['points']) for p in trace['points']))

    def test_nearby_endpoints_do_not_paint_missing_ink(self):
        gray, safe, words = self.scene()
        cv2.line(gray, (40, 100), (149, 100), 80, 2)
        cv2.line(gray, (162, 100), (340, 100), 80, 2)
        traces = [self.trace(np.arange(40, 150), 100), self.trace(np.arange(162, 341), 100)]
        joined, links = connected_rule_traces(traces, gray, safe, words, 0)
        self.assertEqual(len(joined), 2)
        self.assertEqual(links, [])

    def test_word_or_unsafe_gap_cannot_connect_owners(self):
        for blocked in ('words', 'safe'):
            gray, safe, words = self.scene()
            cv2.line(gray, (40, 100), (340, 100), 80, 2)
            if blocked == 'words':
                words[95:106, 155:158] = 1
            else:
                safe[95:106, 155:158] = 0
            traces = [self.trace(np.arange(40, 150), 100), self.trace(np.arange(162, 341), 100)]
            self.assertEqual(len(connected_rule_traces(traces, gray, safe, words, 0)[0]), 2)

    def test_orthogonal_network_does_not_join_separate_rules(self):
        gray, safe, words = self.scene()
        cv2.line(gray, (40, 100), (190, 100), 80, 2)
        cv2.line(gray, (200, 120), (340, 120), 80, 2)
        cv2.line(gray, (190, 100), (200, 120), 80, 2)
        traces = [self.trace(np.arange(40, 191), 100), self.trace(np.arange(200, 341), 120)]
        self.assertEqual(len(connected_rule_traces(traces, gray, safe, words, 0)[0]), 2)

    def test_connected_curve_is_one_residual_owner_not_several_plane_votes(self):
        gray, safe, words = self.scene()
        x = np.arange(40, 341)
        y = 100 + .0008 * (x - 190)**2
        cv2.polylines(gray, [np.rint(np.c_[x, y]).astype(np.int32)], False, 80, 1)
        traces = [self.trace(x[lo:hi], y[lo:hi]) for lo, hi in ((0, 95), (103, 195), (203, 301))]
        joined, _ = connected_rule_traces(traces, gray, safe, words, 0)
        self.assertEqual(len(joined), 1)
        v = [dict(kind='rule', points=self.trace(x, col)['points'][:, ::-1]) for col in (40, 190, 340)]
        result = estimate_printed_plane(traces, v, [400, 400], gray=gray, safe=safe, word_mask=words)
        self.assertFalse(result['accepted'])
        self.assertIsNone(result['matrix'])
        self.assertEqual(result['ownership'][0]['groups'], [[0, 1, 2]])
        self.assertIn('curved', result['rejected'][0]['reason'])

    def test_transposition_retains_physical_owner(self):
        gray, safe, words = self.scene()
        cv2.line(gray, (40, 100), (340, 100), 80, 2)
        h = [self.trace(np.arange(40, 150), 100), self.trace(np.arange(162, 341), 100)]
        v = [dict(t, points=t['points'][:, ::-1]) for t in h]
        horizontal, _ = connected_rule_traces(h, gray, safe, words, 0)
        vertical, _ = connected_rule_traces(v, gray.T, safe.T, words.T, 1)
        np.testing.assert_array_equal(horizontal[0]['points'][:, ::-1], vertical[0]['points'])

    def test_owner_cannot_be_an_unordered_ink_cloud(self):
        gray, safe, words = self.scene()
        cv2.line(gray, (40, 100), (340, 100), 80, 2)
        trace = self.trace(np.arange(40, 150), 100)
        trace['points'] = trace['points'][np.random.default_rng(0).permutation(len(trace['points']))]
        joined, links = connected_rule_traces([trace, self.trace(np.arange(162, 341), 100)], gray, safe, words, 0)
        self.assertEqual(len(joined), 2)
        self.assertEqual(links, [])

    def test_measurement_rasters_must_be_paired(self):
        gray, safe, words = self.scene()
        with self.assertRaises(ValueError):
            connected_rule_traces([], gray, safe[:-1], words, 0)
        with self.assertRaises(ValueError):
            estimate_printed_plane([], [], [400, 400], gray=gray, safe=safe)


if __name__ == '__main__':
    unittest.main()
