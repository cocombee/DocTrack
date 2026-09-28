import unittest
import numpy as np
from word_flanked_rule_rows import observed_registered_sections


class ObservedRuleTailTests(unittest.TestCase):
    def evidence(self):
        trace = np.c_[np.arange(40, 401)/799, np.full(361, 200/399)]
        x = np.arange(36, 201)
        a = np.c_[x/799, (200 + np.maximum(0, 40-x)*.4)/399]
        b = np.c_[np.arange(206, 401)/799, np.full(195, 200/399)]
        safe = np.full((400, 800), 255, np.uint8)
        mask = np.zeros_like(safe)
        return trace, a, b, mask, safe

    def test_observed_tail_extends_without_rewriting_old_material(self):
        trace, a, b, mask, safe = self.evidence()
        result = observed_registered_sections([a, b], [trace], mask, safe, 800, 400)
        self.assertEqual(len(result), 1)
        np.testing.assert_array_equal(result[0][:4], a[:4])
        np.testing.assert_array_equal(result[0][4:], trace)
        self.assertTrue((np.diff(result[0][:, 0]) > 0).all())

    def test_words_and_unsafe_stock_cannot_supply_tail(self):
        for kind in ('word', 'stock'):
            trace, a, b, mask, safe = self.evidence()
            if kind == 'word': mask[197:205, 34:40] = 1
            else: safe[197:205, 34:40] = 0
            result = observed_registered_sections([a, b], [trace], mask, safe, 800, 400)
            np.testing.assert_array_equal(result[0], trace)

    def test_conflicting_tail_measurements_are_not_chosen(self):
        trace, a, b, mask, safe = self.evidence()
        conflict = a.copy();conflict[:4, 1] += .25/399
        result = observed_registered_sections([a, conflict, b], [trace], mask, safe, 800, 400)
        np.testing.assert_array_equal(result[0], trace)

    def test_long_or_unmatched_sections_cannot_extend_registration(self):
        trace, a, b, mask, safe = self.evidence()
        a[:4, 0] -= 20/799
        result = observed_registered_sections([a, b], [trace], mask, safe, 800, 400)
        np.testing.assert_array_equal(result[0], trace)
        wrong = a.copy();wrong[:, 1] += 3/399
        self.assertEqual(observed_registered_sections([wrong, b], [trace], mask, safe, 800, 400), [])


if __name__ == '__main__': unittest.main()
