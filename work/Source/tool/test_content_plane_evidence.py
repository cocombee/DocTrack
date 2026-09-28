import unittest
import numpy as np
from content_plane_evidence import estimate_printed_plane, transform_points, guard_printed_plane


class PrintedPlaneTest(unittest.TestCase):
    size = [1200, 1600]

    def grid(self, matrix=None, kind='rule'):
        matrix = np.eye(3) if matrix is None else matrix
        t = np.linspace(.08, .92, 151)
        h = [dict(kind=kind, points=transform_points(matrix, np.c_[t, np.full(len(t), y)]))
             for y in (.12, .3, .5, .7, .88)]
        v = [dict(kind=kind, points=transform_points(matrix, np.c_[np.full(len(t), x), t]))
             for x in (.1, .3, .5, .7, .9)]
        return h, v

    def test_recovers_printed_axes_from_independent_projective_traces(self):
        photo = np.array([[.98, .065, .012], [-.035, 1.01, .021], [.05, -.025, 1]])
        h, v = self.grid(photo)
        result = estimate_printed_plane(h, v, self.size)
        self.assertTrue(result['accepted'], result)
        corrected = np.asarray(result['matrix']) @ photo
        # Previously unseen positions, outside the lines used for fitting,
        # must also retain straight lines under this projective estimate.
        t = np.linspace(.05, .95, 117)
        for y in (.21, .64, .94):
            q = transform_points(corrected, np.c_[t, np.full(len(t), y)])
            self.assertLess(np.ptp(q[:, 1]), 1e-10)
        for x in (.17, .61, .96):
            q = transform_points(corrected, np.c_[np.full(len(t), x), t])
            self.assertLess(np.ptp(q[:, 0]), 1e-10)

    def test_torn_or_curved_silhouette_has_no_vote(self):
        h, v = self.grid()
        before = estimate_printed_plane(h, v, self.size)
        t = np.linspace(.01, .99, 151)
        torn = dict(kind='paper_edge', points=np.c_[t, .02 + .04 * np.sin(t * 16)])
        after = estimate_printed_plane([torn, *h], [*v, torn], self.size)
        np.testing.assert_array_equal(before['matrix'], after['matrix'])
        self.assertEqual(sum(r['reason'] == 'not printed evidence' for r in after['rejected']), 2)

    def test_outline_alone_cannot_estimate_interior(self):
        h, v = self.grid(kind='paper_edge')
        result = estimate_printed_plane(h, v, self.size)
        self.assertFalse(result['accepted'])
        self.assertIsNone(result['matrix'])

    def test_one_direction_does_not_invent_perspective(self):
        h, _ = self.grid(kind='text_baseline')
        self.assertFalse(estimate_printed_plane(h, [], self.size)['accepted'])

    def test_local_bend_is_residual_not_plane_evidence(self):
        h, v = self.grid()
        before = estimate_printed_plane(h, v, self.size)
        t = np.linspace(.08, .4, 151)
        bent = dict(kind='rule', points=np.c_[t, .06 + .06 * np.sin(np.pi * (t-.08)/.32)])
        after = estimate_printed_plane([bent, *h], v, self.size)
        self.assertTrue(after['accepted'])
        self.assertTrue(any(r['reason'] == 'curved trace' for r in after['rejected']))
        np.testing.assert_allclose(before['matrix'], after['matrix'], atol=1e-12)

    def test_small_bent_tail_cannot_hide_in_percentile_plane_fit(self):
        h, v = self.grid()
        before = estimate_printed_plane(h, v, self.size)
        t = np.linspace(.08, .92, 501)
        y = np.full(len(t), .05)
        y[-8:] += np.linspace(0, .004, 8)
        bent = dict(kind='rule', points=np.c_[t, y])
        after = estimate_printed_plane([bent, *h], v, self.size)
        self.assertTrue(after['accepted'], after)
        self.assertTrue(any(r['reason'] == 'curved trace' for r in after['rejected']))
        np.testing.assert_allclose(before['matrix'], after['matrix'], atol=1e-12)

    def test_sparse_text_requires_measured_glyph_support(self):
        h, v = self.grid()
        sparse = [dict(kind='text_baseline', points=t['points'][::30]) for t in h]
        self.assertFalse(estimate_printed_plane(sparse, v, self.size)['accepted'])
        for trace in sparse:
            trace['glyph_baseline_support'] = dict(
                basis='Native row with measured raster glyph bottoms',
                native_word_count=3, glyph_count=len(trace['points']))
        self.assertTrue(estimate_printed_plane(sparse, v, self.size)['accepted'])
        for trace in sparse:
            trace['glyph_baseline_support']['native_word_count'] = 1
        self.assertFalse(estimate_printed_plane(sparse, v, self.size)['accepted'])

    def test_disjoint_collinear_fragments_are_one_vote(self):
        t = np.linspace(.08, .38, 51)
        fragments = [dict(kind='rule', points=np.c_[t + dx, np.full(len(t), .4)])
                     for dx in (0, .2, .4)]
        _, v = self.grid()
        result = estimate_printed_plane(fragments, v, self.size)
        self.assertFalse(result['accepted'])
        self.assertIn('independent', result['reason'])

    def test_converging_straight_fragments_are_valid_perspective(self):
        h, v = self.grid()
        for i, trace in enumerate(v):
            p = trace['points'].copy()
            p[:, 0] += (p[:, 1] - .5) * (i-2) * .045
            trace['points'] = p
        result = estimate_printed_plane(h, v, self.size)
        self.assertTrue(result['accepted'], result)
        for trace in v:
            mapped = transform_points(result['matrix'], trace['points'])
            self.assertLess(np.ptp(mapped[:, 0]), 1e-10)

    def test_conflicting_straight_fragments_abstain(self):
        h, v = self.grid()
        # Nonlinear changes of slope across columns have no common vanishing
        # point. Linear slope changes instead describe real convergence.
        for trace, slope in zip(v, (-.09, .06, -.025, .08, -.055)):
            p = trace['points'].copy()
            p[:, 0] += (p[:, 1] - .5) * slope
            trace['points'] = p
        result = estimate_printed_plane(h, v, self.size)
        self.assertFalse(result['accepted'])
        self.assertIsNone(result['matrix'])

    def test_invalid_measurement_is_rejected(self):
        for size in ([1, 100], [100, float('nan')], [100]):
            with self.assertRaises(ValueError):
                estimate_printed_plane([], [], size)

    def test_verified_shape_lines_can_supply_plane(self):
        h, v = self.grid(kind='shape_stroke')
        self.assertTrue(estimate_printed_plane(h, v, self.size)['accepted'])

    def test_plane_consensus_does_not_authorize_word_shape_damage(self):
        h, v = self.grid()
        proposal = estimate_printed_plane(h, v, self.size)
        proposal['matrix'][0][1] = .02
        owner = np.array([[.2, .2], [.3, .2], [.2, .24], [.3, .24]])
        original = owner.copy()
        result = guard_printed_plane(proposal, [owner], self.size)
        self.assertFalse(result['accepted'])
        self.assertIsNone(result['matrix'])
        self.assertTrue(proposal['accepted'])
        np.testing.assert_array_equal(owner, original)

    def test_rigid_plane_and_strict_unknown_limit(self):
        owner = np.array([[.2, .2], [.3, .2], [.2, .24], [.3, .24]])
        proposal = dict(accepted=True, matrix=np.eye(3).tolist())
        self.assertTrue(guard_printed_plane(proposal, [owner], self.size, unknown_owners=[owner])['accepted'])
        proposal['matrix'][0][0] += .0006
        # 0.072px pair expansion passes words, but fails 0.03px missed print.
        self.assertTrue(guard_printed_plane(proposal, [owner], self.size)['accepted'])
        result = guard_printed_plane(proposal, [owner], self.size, unknown_owners=[owner])
        self.assertFalse(result['accepted'])

    def test_plane_without_ink_owners_cannot_be_qualified(self):
        proposal = dict(accepted=True, matrix=np.eye(3).tolist())
        self.assertFalse(guard_printed_plane(proposal, [], self.size)['accepted'])


if __name__ == '__main__':
    unittest.main()
