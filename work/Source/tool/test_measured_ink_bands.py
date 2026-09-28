import unittest
import numpy as np
from measured_ink_bands import pixel_footprint, column_bands


class MeasuredInkBandTests(unittest.TestCase):
    def test_all_pixel_corners_and_centers_remain_accounted_for(self):
        points=np.array([[.2,.3],[.4,.5]])
        cloud=pixel_footprint(points,101,201)
        self.assertEqual(len(cloud),10)
        for point in points:
            self.assertTrue(np.any(np.all(cloud==point,axis=1)))
        np.testing.assert_allclose(cloud.min(0),[.195,.2975])
        np.testing.assert_allclose(cloud.max(0),[.405,.5025])

    def test_blank_corners_do_not_couple_separate_word_owners(self):
        label=np.array([[.2,.2],[.2,.23],[.3,.2],[.3,.3]])
        number=np.array([[.2,.28],[.2,.31]])
        low,high=column_bands([label,number],101,.005,.005)
        self.assertLess(high[0,20],low[1,20])
        self.assertEqual(high[0,30],.305)
        self.assertFalse(np.isfinite(low[0,25]))
        self.assertTrue(np.isfinite(low[1,20]))

    def test_off_lattice_points_and_padding_cover_every_nearby_knot(self):
        cloud=np.array([[.204,.3],[.205,.4],[.199,.2]])
        low,high=column_bands([cloud],101,.012,.003)
        for col in range(101):
            active=abs(cloud[:,0]-col/100)<=.012
            if active.any():
                self.assertLessEqual(low[0,col],cloud[active,1].min())
                self.assertGreaterEqual(high[0,col],cloud[active,1].max())
            else:
                self.assertFalse(np.isfinite(low[0,col]))

    def test_invalid_footprints_are_declined(self):
        for points in ([],[[float('nan'),.3]],[[.3,.2,.1]],[[1.01,.3]]):
            with self.assertRaises(ValueError):pixel_footprint(points,101,201)
        with self.assertRaises(ValueError):column_bands([],101,-1,0)
        with self.assertRaises(ValueError):column_bands([],101,float('nan'),0)


if __name__=='__main__':unittest.main()
