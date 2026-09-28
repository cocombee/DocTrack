import unittest
import cv2
import numpy as np
from word_component_ownership import partition_shared_ink


class ComponentOwnershipTests(unittest.TestCase):
    def fixture(self):
        gray=np.full((400,500),235,np.uint8)
        gray[90:102,100:108]=70;gray[112:124,100:108]=70
        yy,xx=np.where(gray<200);points=np.c_[xx/499,yy/399]
        upper=points[yy<102];lower=points[yy>=112]
        quads=[np.array([[98,88],[110,88],[110,104],[98,104]])/[499,399],
               np.array([[98,110],[110,110],[110,126],[98,126]])/[499,399]]
        return gray,upper,lower,quads

    def test_borrowed_cap_pixels_keep_their_complete_lower_glyph_owner(self):
        gray,upper,lower,quads=self.fixture()
        borrowed=lower[:8]
        clean,audit=partition_shared_ink(gray,[np.vstack([upper,borrowed]),lower],quads)
        np.testing.assert_array_equal(clean[0],np.unique(upper,axis=0))
        np.testing.assert_array_equal(clean[1],np.unique(lower,axis=0))
        self.assertTrue(audit['all_physical_points_retained'])
        self.assertEqual(audit['roles'][0]['owner'],1)

    def test_two_full_independent_observations_stay_ambiguous(self):
        gray,upper,lower,quads=self.fixture()
        groups=[np.vstack([upper,lower]),lower]
        clean,audit=partition_shared_ink(gray,groups,quads)
        self.assertEqual(audit['roles'],[])
        for a,z in zip(groups,clean):np.testing.assert_array_equal(np.unique(a,axis=0),z)

    def test_wider_halo_needs_every_borrowed_point_outside_competing_quad(self):
        gray,upper,lower,quads=self.fixture();borrowed=lower[:24]
        groups=[np.vstack([upper,borrowed]),lower]
        clean,audit=partition_shared_ink(gray,groups,quads)
        np.testing.assert_array_equal(clean[0],np.unique(upper,axis=0))
        self.assertEqual(audit['roles'][0]['owner'],1)
        enlarged=[quads[0].copy(),quads[1]]
        enlarged[0][2:,1]=125/399
        clean,audit=partition_shared_ink(gray,groups,enlarged)
        self.assertEqual(audit['roles'],[])
        np.testing.assert_array_equal(clean[0],np.unique(groups[0],axis=0))

    def test_single_point_punctuation_needs_a_unique_undilated_envelope(self):
        gray,upper,lower,quads=self.fixture();gray[103,105]=100
        dot=np.array([[105/499,103/399]])
        groups=[np.vstack([upper,dot]),np.vstack([lower,dot])]
        clean,audit=partition_shared_ink(gray,groups,quads)
        self.assertEqual(len(clean[0]),len(upper)+1)
        self.assertEqual(len(clean[1]),len(lower))
        ambiguous=[quads[0],quads[0]]
        _,audit=partition_shared_ink(gray,groups,ambiguous)
        self.assertEqual(audit['roles'],[])

    def test_invalid_pairing_and_empty_owner_are_not_accepted(self):
        gray,upper,lower,quads=self.fixture()
        with self.assertRaises(ValueError):partition_shared_ink(gray,[upper],quads)
        with self.assertRaises(ValueError):partition_shared_ink(gray,[np.empty((0,2))],[quads[0]])


if __name__=='__main__':unittest.main()
