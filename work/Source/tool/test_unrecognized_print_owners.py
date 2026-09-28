import unittest
import cv2
import numpy as np
from unrecognized_print_owners import unrecognized_groups


class UnrecognizedPrintTests(unittest.TestCase):
    def page(self):
        gray = np.full((400, 500), 230, np.uint8)
        gray[150:159, 200:210] = 70
        gray[150:160, 212:221] = 80
        gray[150:159, 250:260] = 90
        return gray, np.full_like(gray, 255)

    def test_missed_adjacent_bodies_keep_exact_ink_and_separate_word_gap(self):
        gray, safe = self.page()
        groups, audit = unrecognized_groups(gray, [], safe, [], [])
        self.assertEqual(len(groups), 2)
        self.assertEqual(len(audit['measured_owners'][0]['components']), 2)
        for group in groups:
            x, y = np.rint(group * [499, 399]).astype(int).T
            self.assertTrue((gray[y, x] < 230).all())
        self.assertFalse(audit['recognition_inferred'])

    def test_partial_native_owner_cannot_be_reassigned(self):
        gray, safe = self.page()
        native = np.array([[203 / 499, 153 / 399]])
        before = native.copy()
        groups, audit = unrecognized_groups(gray, [native], safe, [], [])
        self.assertFalse(any((np.rint(g * [499, 399]).astype(int)[:, 0] < 210).any()
                             for g in groups))
        np.testing.assert_array_equal(native, before)
        self.assertFalse(audit['existing_owners_changed'])

    def test_rule_crossing_and_unsafe_body_are_not_new_owners(self):
        gray, safe = self.page()
        horizontal = [np.array([[190 / 499, 154 / 399], [225 / 499, 154 / 399]])]
        safe[151, 253] = 0
        groups, _ = unrecognized_groups(gray, [], safe, horizontal, [])
        self.assertEqual(groups, [])

    def test_long_printed_rules_are_not_compact_glyph_owners(self):
        gray = np.full((400, 500), 230, np.uint8)
        cv2.line(gray, (30, 35), (440, 35), 60, 2)
        cv2.line(gray, (25, 50), (25, 370), 70, 2)
        groups, _ = unrecognized_groups(gray, [], np.full_like(gray, 255), [], [])
        self.assertEqual(groups, [])


if __name__ == '__main__':
    unittest.main()
