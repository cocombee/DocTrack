import unittest
import numpy as np
from word_flanked_rule_rows import word_flanked_rows,supported_registered_sections


class WordFlankedRowTests(unittest.TestCase):
    def fixture(self):
        left = np.c_[np.linspace(.05, .3, 101), np.full(101, .2)]
        right = np.c_[np.linspace(.65, .95, 121), np.full(121, .215)]
        quads = [np.array([[x, .197], [x + .16, .197],
                           [x + .16, .224], [x, .224]]) for x in (.305, .475)]
        groups = []
        for quad in quads:
            xx, yy = np.meshgrid(np.linspace(quad[0, 0] + .002, quad[1, 0] - .002, 8),
                                 np.linspace(.2, .22, 5))
            groups.append(np.c_[xx.ravel(), yy.ravel()])
        return [left, right], quads, groups, [right.copy()]

    def detect(self, guides, quads, groups, anchors):
        return word_flanked_rows(guides, quads, groups, len(groups), anchors, 1281, 1281)

    def test_complete_words_support_a_shared_row_without_inventing_gap_ink(self):
        guides, quads, groups, anchors = self.fixture()
        result = self.detect(guides, quads, groups, anchors)
        self.assertEqual(len(result), 1)
        row = result[0]
        self.assertEqual(row['anchor_side'], 1)
        self.assertAlmostEqual(row['target'], .215)
        self.assertEqual(row['word_owners'], [0, 1])
        np.testing.assert_array_equal(row['guides'][0], guides[0])
        np.testing.assert_array_equal(row['guides'][1], guides[1])
        self.assertGreater(row['word_coverage'], .9)

    def test_unsupported_gap_missing_word_and_ambiguous_ownership_decline(self):
        guides, quads, groups, anchors = self.fixture()
        self.assertEqual(self.detect(guides, quads[:1], groups[:1], anchors), [])
        self.assertEqual(self.detect(guides, quads, groups + [groups[0]], anchors), [])
        moved = [quads[0], quads[1] + [.08, 0]]
        self.assertEqual(self.detect(guides, moved, groups, anchors), [])

    def test_different_word_rows_and_fonts_cannot_supply_one_baseline(self):
        guides, quads, groups, anchors = self.fixture()
        self.assertEqual(self.detect(guides, [quads[0], quads[1] + [0, .003]], groups, anchors), [])
        tall = quads[1].copy(); tall[2:, 1] += .015
        self.assertEqual(self.detect(guides, [quads[0], tall], groups, anchors), [])

    def test_no_registered_anchor_or_a_curved_anchor_cannot_supply_a_target(self):
        guides, quads, groups, anchors = self.fixture()
        self.assertEqual(self.detect(guides, quads, groups, []), [])
        curved = anchors[0].copy(); curved[:, 1] += np.linspace(0, .001, len(curved))
        self.assertEqual(self.detect(guides, quads, groups, [curved]), [])
        self.assertEqual(self.detect(guides, quads, groups, [anchors[0] + [0, .005]]), [])

    def test_overlapping_sections_or_a_word_cut_by_a_section_decline(self):
        guides, quads, groups, anchors = self.fixture()
        left = guides[0].copy(); left[:, 0] += .015
        self.assertEqual(self.detect([left, guides[1]], quads, groups, anchors), [])
        self.assertEqual(self.detect([guides[0], guides[0]], quads, groups, anchors), [])

    def test_already_aligned_sections_and_two_possible_baseline_targets_decline(self):
        guides, quads, groups, anchors = self.fixture()
        aligned = guides[0].copy(); aligned[:, 1] = .215
        self.assertEqual(self.detect([aligned, guides[1]], quads, groups, anchors), [])
        ambiguous = guides[0].copy(); ambiguous[:, 1] = .213
        self.assertEqual(self.detect([ambiguous, guides[1]], quads, groups, anchors), [])

    def test_invalid_native_owner_boundaries_cannot_use_carried_words_as_new_evidence(self):
        guides, quads, groups, anchors = self.fixture()
        for count in (-1, 3, True, 1.5):
            with self.assertRaises(ValueError):
                word_flanked_rows(guides, quads, groups, count, anchors, 1281, 1281)

    def test_complete_registration_requires_two_real_sections_and_no_word_in_a_weak_gap(self):
        width=height=1281;mask=np.zeros((height,width),np.uint8);safe=np.full_like(mask,255)
        trace=np.c_[np.arange(64,449)/1280,np.full(385,.2)]
        guides=[trace[:240],trace[245:]]
        recovered=supported_registered_sections(guides,[trace],mask,safe,width,height)
        self.assertEqual(len(recovered),1)
        np.testing.assert_array_equal(recovered[0],trace)
        self.assertEqual(supported_registered_sections([trace[:240]],[trace],mask,safe,width,height),[])
        self.assertEqual(supported_registered_sections([trace[:120],trace[245:]],[trace],mask,safe,width,height),[])
        mask[256,306]=1
        self.assertEqual(supported_registered_sections(guides,[trace],mask,safe,width,height),[])


if __name__ == '__main__':
    unittest.main()
