import unittest
import cv2
import numpy as np
from repeated_rule_mask import recover
from word_rule_partition import partition_measured_repeated_rulings


class RepeatedRulingWordPartitionTests(unittest.TestCase):
    def fixture(self):
        gray=np.full((400,800),240,np.uint8)
        for x in range(60,760,17):gray[198:202,x:x+14]=45
        yy,xx=np.mgrid[230:248,179:187];glyphs=np.c_[xx.ravel(),yy.ravel()]
        yy,xx=np.mgrid[198:202,179:193];dash=np.c_[xx.ravel(),yy.ravel()]
        group=np.vstack([glyphs,dash])/[799,399]
        xs=np.flatnonzero(gray[200]<240);guide=np.c_[xs/799,np.full(len(xs),199.5/399)]
        _,evidence=recover(gray,np.full_like(gray,255),np.zeros_like(gray),
                           np.zeros_like(gray),[guide],whole_rulings=True)
        return group,glyphs/[799,399],dash/[799,399],evidence

    def test_one_accidentally_included_dash_has_exact_glyph_and_rule_roles(self):
        group,glyphs,dash,evidence=self.fixture()
        core,roles,audit=partition_measured_repeated_rulings([group],evidence,800,400)
        np.testing.assert_array_equal(core[0],glyphs)
        np.testing.assert_array_equal(roles[0]['points'],dash)
        self.assertEqual(sorted(audit[0]['glyph_point_indices']+roles[0]['point_indices']),list(range(len(group))))

    def test_missing_second_profile_or_ambiguous_ruling_keeps_word_ownership(self):
        group,_,__,evidence=self.fixture()
        single=[{**evidence[0],'thresholds':[6]}]
        for proof in ([],single,evidence+evidence):
            core,roles,_=partition_measured_repeated_rulings([group],proof,800,400)
            np.testing.assert_array_equal(core[0],group)
            self.assertFalse(roles)

    def test_touching_glyph_or_distant_dash_cannot_transfer(self):
        group,_,__,evidence=self.fixture()
        bridge=np.c_[np.full(29,179),np.arange(201,230)]/[799,399]
        for actual in (np.vstack([group,bridge]),group+[0,.03]):
            core,roles,_=partition_measured_repeated_rulings([actual],evidence,800,400)
            np.testing.assert_array_equal(core[0],actual)
            self.assertFalse(roles)

    def test_pairing_is_validated_before_any_role_change(self):
        with self.assertRaises(ValueError):
            partition_measured_repeated_rulings([np.array([[np.nan,0]])],[],800,400)


if __name__=='__main__':unittest.main()
