import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

import printed_rule_sections as detector
import section_row_candidate as candidate
import frame_section_candidate as frame
import shared_row_candidate as shared


class RuleSectionTests(unittest.TestCase):
    def test_same_rule_is_measured_at_different_page_heights_without_crossing_words(self):
        for row in (100, 200, 300):
            with self.subTest(row=row):
                gray = np.full((400, 800), 235, np.uint8)
                x = np.arange(40, 760)
                y = row + 5 * np.sin(x / 800 * 2 * np.pi)
                cv2.polylines(gray, [np.rint(np.c_[x, y]).astype(np.int32)], False, 90, 1)
                cv2.putText(gray, 'WORDS', (330, row + 6),
                            cv2.FONT_HERSHEY_SIMPLEX, .8, 20, 2)
                sections = detector.measure_sections(gray)
                self.assertGreaterEqual(len(sections), 2)
                for section in sections:
                    pixels = section * [799, 399]
                    self.assertFalse(pixels[:, 0].min() < 330 and pixels[:, 0].max() > 410)
                    expected = row + 5 * np.sin(pixels[:, 0] / 800 * 2 * np.pi)
                    self.assertLess(np.max(abs(pixels[:, 1] - expected)), 1)

    def test_a_close_word_stops_only_a_stroke_that_touches_its_protected_ink(self):
        gray=np.full((400,800),235,np.uint8);words=np.zeros_like(gray)
        cv2.line(gray,(40,150),(760,150),90,1)
        for x in (340,350,360):
            cv2.line(gray,(x,154),(x,164),30,1)
            words[153:166,x-1:x+2]=1
        legacy=detector.measure_sections(gray,word_mask=words)
        precise=detector.measure_sections(gray,word_mask=words,word_safe_ridge=True)
        self.assertFalse(any(g[0,0]*799<100 and g[-1,0]*799>700 for g in legacy))
        self.assertTrue(any(g[0,0]*799<100 and g[-1,0]*799>700 for g in precise))
        words[148:153,340:361]=1
        blocked=detector.measure_sections(gray,word_mask=words,word_safe_ridge=True)
        self.assertFalse(any(g[0,0]*799<340 and g[-1,0]*799>361 for g in blocked))

    def test_blank_stock_does_not_supply_rule_sections(self):
        gray = np.full((400, 800), 235, np.uint8)
        self.assertEqual(detector.measure_sections(gray), [])

    def test_separate_rows_cannot_be_combined_into_one_supported_row(self):
        first = np.c_[np.linspace(.02, .45, 100), np.linspace(.25, .26, 100)]
        second = np.c_[np.linspace(.47, .98, 100), np.linspace(.30, .31, 100)]
        self.assertEqual(candidate.associate_rows([first, second], 800, 400), [])

    def test_existing_row_matching_excludes_other_printed_rows(self):
        x = np.linspace(.02, .98, 101)
        measured = np.c_[x, .3 + .005 * np.sin(x * 2 * np.pi)]
        same = np.c_[x, np.full(101, .3)]
        other = np.c_[x, np.full(101, .4)]
        self.assertEqual(candidate.matching_rules(measured, [other, same], 400), [1])

    def test_neighbour_guards_exclude_only_the_selected_visible_rule(self):
        u, v = np.meshgrid(np.linspace(0, 1, 33), np.linspace(0, 1, 33))
        grid = np.stack([u, v], -1)
        x = np.linspace(.05, .95, 101)
        selected = np.c_[x, .3 + .005 * np.sin(x * 2 * np.pi)]
        neighbour = selected + [0, .015]
        vertical = np.c_[x, np.full(101, .2)]
        guards = candidate.neighbouring_rule_guards(
            grid, [selected, neighbour], [vertical], [selected], 800, 400)
        self.assertEqual([g['normal_axis'] for g in guards], [1, 0])
        np.testing.assert_allclose(guards[0]['points'], neighbour, atol=1e-15, rtol=0)
        np.testing.assert_allclose(guards[1]['points'], vertical[:, ::-1], atol=1e-15, rtol=0)

    def test_fixed_rule_guard_covers_its_close_crossing_without_reaching_distant_print(self):
        trace = np.c_[np.linspace(.103, .897, 101), np.full(101, .3)]
        left = np.c_[np.full(101, .1), np.linspace(.2, .4, 101)]
        right = np.c_[np.full(101, .9), np.linspace(.2, .4, 101)]
        distant = np.c_[np.full(101, .05), np.linspace(.2, .4, 101)]
        unrelated = left + [0, .3]
        self.assertEqual(candidate.joined_rule_extent(trace, [left, right, distant], 800, 1280), (.1, .9))
        self.assertEqual(candidate.joined_rule_extent(trace, [distant, unrelated], 800, 1280), (.103, .897))

    def test_shared_row_step_declines_documents_without_a_supported_grid(self):
        base = {'method': 'Accepted receipt correction'}
        gray = np.full((400, 800), 235, np.uint8)
        cv2.putText(gray, 'KEEP THIS RECEIPT', (100, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, .7, 20, 2)
        with patch.object(candidate, 'current_measurement', return_value=(gray, np.full_like(gray,255), gray, None)):
            self.assertIs(shared.refine('unused.png', base), base)

    def test_shared_row_keeps_faint_word_edges_and_the_next_rule_fixed(self):
        gray = np.full((400, 800), 235, np.uint8)
        cv2.putText(gray, 'KEEP FAINT FRINGES', (90, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, .7, 155, 2)
        cv2.putText(gray, 'WORKING WORDS', (470, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, .7, 155, 2)
        x = np.linspace(50, 750, 701)
        y = np.where(x < 350, 180 + 8 * np.sin((x - 50) / 300 * np.pi), 180)
        guide = np.c_[x / 799, y / 399]
        cv2.polylines(gray, [np.rint(np.c_[x, y]).astype(np.int32)], False, 90, 1)
        cv2.line(gray, (50, 198), (750, 198), 90, 1)
        primary = np.c_[np.linspace(400 / 799, 750 / 799, 101), np.full(101, 180 / 399)]
        other = np.c_[np.linspace(50 / 799, 750 / 799, 101), np.full(101, 198 / 399)]
        u, v = np.meshgrid(np.linspace(0, 1, 33), np.linspace(0, 1, 33))
        base = {'rows': 33, 'columns': 33, 'grid': np.stack([u, v], -1).reshape(-1, 2).tolist(), 'method': 'Accepted'}
        wy, wx = np.where(gray[140:165] < 220)
        whole_word_ink = np.c_[wx / 799, (wy + 140) / 399]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'page.png'
            cv2.imwrite(str(path), cv2.cvtColor(gray, cv2.COLOR_GRAY2BGRA))
            def rules(image):
                return [primary, other] if image.shape == gray.shape else []
            with patch.object(candidate.p.reference, 'extract_rules', side_effect=rules), \
                    patch.object(candidate.p.reference, 'barcode_rulers', return_value=[]), \
                    patch.object(candidate.accepted.accepted, 'circular_landmarks', return_value=[]):
                result = candidate.refine(path, base, guides=[guide], align_to_existing=True,
                                          include_wide_ink=True, minimum_resolution=(4097, 513),
                                          word_ink_groups=[whole_word_ink])
        self.assertTrue(result is not base)
        self.assertEqual(result['joint_row_audit']['matched_primary_rules'], 1)
        self.assertLess(result['joint_row_audit']['full_rule_error_px'], .1)
        self.assertLess(result['joint_row_audit']['other_print_shift_px'], .01)
        self.assertLess(result['joint_row_audit']['whole_word_ink_rigidity_px'], .01)
        grid = np.asarray(result['grid']).reshape(result['rows'], result['columns'], 2)
        yy, xx = np.where(gray[140:165] < 220)
        pixels = np.c_[xx / 799, (yy + 140) / 399]
        xx, yy = np.meshgrid(np.arange(50, 751), np.arange(197, 200))
        pixels = np.vstack([pixels, np.c_[xx.ravel() / 799, yy.ravel() / 399]])
        error = abs(candidate.p.inverse(grid, pixels) - pixels) * [800, 400]
        self.assertLess(error.max(), .01)

    def test_unsupported_step_returns_the_entire_original_chain(self):
        u, v = np.meshgrid(np.linspace(0, 1, 33), np.linspace(0, 1, 33))
        base = {'rows': 33, 'columns': 33, 'grid': np.stack([u, v], -1).reshape(-1, 2).tolist(),
                'method': 'accepted', 'preserved_mesh': {'method': 'earlier accepted stage'},
                'refinement_regions': [[.1, .2, .8, .9]]}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'page.png'
            cv2.imwrite(str(path), np.full((400, 800, 4), 255, np.uint8))
            self.assertIs(candidate.refine(path, base), base)
        with patch.object(candidate.accepted, 'solve', return_value=base), \
                patch.object(candidate, 'refine', side_effect=ValueError('unsupported proposal')):
            self.assertIs(candidate.solve('unused.png'), base)

    def test_later_patch_levels_new_curve_while_the_prior_row_stays_fixed(self):
        gray = np.full((400, 800), 235, np.uint8)
        cv2.putText(gray, 'PRESERVED WORKING WORDS', (80, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, .7, 20, 2)
        x = np.linspace(50, 750, 701)
        y = 180 + 8 * np.sin(x / 800 * 2 * np.pi)
        guide = np.c_[x / 799, y / 399]
        cv2.polylines(gray, [np.rint(np.c_[x, y]).astype(np.int32)], False, 90, 1)
        cv2.line(gray, (50, 158), (750, 158), 90, 1)
        frozen = np.c_[np.linspace(50 / 799, 750 / 799, 101), np.full(101, 158 / 399)]
        u, v = np.meshgrid(np.linspace(0, 1, 33), np.linspace(0, 1, 33))
        base = {'rows': 33, 'columns': 33, 'grid': np.stack([u, v], -1).reshape(-1, 2).tolist(),
                'method': 'accepted row', 'established_rulers':
                [{'normal_axis': 1, 'points': frozen.tolist()}]}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'page.png'
            cv2.imwrite(str(path), cv2.cvtColor(gray, cv2.COLOR_GRAY2BGRA))
            with patch.object(candidate.p.reference, 'extract_rules', return_value=[]), \
                    patch.object(candidate.p.reference, 'barcode_rulers', return_value=[]), \
                    patch.object(candidate.accepted.accepted, 'circular_landmarks', return_value=[]):
                result = candidate.refine(path, base, guides=[guide])
        self.assertIsNot(result, base)
        self.assertIs(result['preserved_mesh'], base)
        self.assertLess(result['joint_row_audit']['full_rule_error_px'], .1)
        self.assertLess(result['joint_row_audit']['established_normal_shift_px'], .01)
        self.assertEqual(result['established_rulers'][0], base['established_rulers'][0])

    def test_broad_extension_requires_agreement_at_all_page_heights(self):
        x = np.linspace(.05, .48, 301)
        for row in (.25, .5, .75):
            broad = np.c_[x, row + .02 * np.sin((x - .05) / .43 * 2 * np.pi)]
            narrow = broad[:160]
            with self.subTest(row=row):
                self.assertEqual(len(frame.supported_extensions([narrow], [broad], 800, 400)), 1)
                unrelated = narrow + [0, .02]
                self.assertEqual(frame.supported_extensions([unrelated], [broad], 800, 400), [])

    def test_temporary_neighbour_rule_guard_preserves_its_curve_without_freezing_a_future_target(self):
        gray = np.full((400, 800), 235, np.uint8)
        cv2.putText(gray, 'PRESERVE WORKING WORDS', (80, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, .7, 20, 2)
        x = np.arange(50, 751)
        y = 180 + 8 * np.sin(x / 800 * 2 * np.pi)
        guide = np.c_[x / 799, y / 399]
        neighbour = np.c_[x / 799, (205 + 2 * np.sin(x / 100)) / 399]
        for points in (guide, neighbour):
            cv2.polylines(gray, [np.rint(points * [799, 399]).astype(np.int32)], False, 90, 1)
        u, v = np.meshgrid(np.linspace(0, 1, 33), np.linspace(0, 1, 33))
        base = {'rows': 33, 'columns': 33,
                'grid': np.stack([u, v], -1).reshape(-1, 2).tolist()}
        temporary = {'normal_axis': 1, 'points': neighbour.tolist()}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'page.png'
            cv2.imwrite(str(path), cv2.cvtColor(gray, cv2.COLOR_GRAY2BGRA))
            with patch.object(candidate.p.reference, 'extract_rules', return_value=[]), \
                    patch.object(candidate.p.reference, 'barcode_rulers', return_value=[]), \
                    patch.object(candidate.accepted.accepted, 'circular_landmarks', return_value=[]):
                result = candidate.refine(path, base, guides=[guide],
                                          additional_rulers=[temporary],
                                          precise_geometry=True)
        self.assertIsNot(result, base)
        grid = np.asarray(result['grid']).reshape(result['rows'], result['columns'], 2)
        actual = candidate.precise_inverse(grid, neighbour)
        self.assertLess(abs(actual[:, 1] - neighbour[:, 1]).max() * 400, .1)
        self.assertNotIn(temporary, result['established_rulers'])
        corrected = candidate.precise_inverse(grid, guide)
        self.assertLess(np.ptp(corrected[:, 1]) * 400, .1)


if __name__ == '__main__':
    unittest.main()
