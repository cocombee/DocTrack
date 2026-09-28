import unittest
import numpy as np
from word_rule_membership import separate_rule_ink


class WordRuleMembershipTests(unittest.TestCase):
    def test_disjoint_divider_dashes_do_not_become_word_ink(self):
        for row in (100, 200, 300):
            yy, xx = np.meshgrid(np.arange(row-20,row-8),np.arange(200,205))
            glyph = np.c_[xx.ravel()+.5, yy.ravel()+.5]/[800,400]
            dash = np.c_[np.arange(200,210)+.5,np.full(10,row+.5)]/[800,400]
            rule = np.array([[.1,(row+.5)/400],[.9,(row+.5)/400]])
            clean, removed = separate_rule_ink([np.vstack([glyph,dash])],rule,800,400)
            np.testing.assert_array_equal(clean[0],glyph)
            np.testing.assert_array_equal(removed[0],dash)

    def test_contact_or_a_letter_near_the_rule_keeps_its_protection(self):
        yy,xx=np.meshgrid(np.arange(180,200),np.arange(200,205))
        glyph=np.c_[xx.ravel()+.5,yy.ravel()+.5]/[800,400]
        dash=np.c_[np.arange(200,210)+.5,np.full(10,200.5)]/[800,400]
        group=np.vstack([glyph,dash])
        rule=np.array([[.1,200.5/400],[.9,200.5/400]])
        clean,removed=separate_rule_ink([group],rule,800,400)
        np.testing.assert_array_equal(clean[0],group)
        self.assertEqual(len(removed[0]),0)


if __name__ == '__main__':
    unittest.main()
