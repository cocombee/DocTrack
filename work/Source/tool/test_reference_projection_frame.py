import unittest
import numpy as np

from reference_projection_frame import second_projection_frame
from content_plane_evidence import transform_points


class ReferenceProjectionFrameTests(unittest.TestCase):
    def fixture(self):
        corners=np.array([[.15,.16],[.79,.11],[.91,.81],[.22,.9]])
        t=np.linspace(0,1,181)[:,None]
        sides=[corners[i]*(1-t)+corners[(i+1)%4]*t for i in range(4)]
        return corners,sides

    def test_rounded_first_pin_does_not_determine_virtual_sharp_corner(self):
        corners,sides=self.fixture()
        pins=corners.copy();pins[0]+=[.014,.018]
        old_pins=pins.copy();old_sides=[p.copy() for p in sides]
        result=second_projection_frame(pins,sides,[1000,800])
        np.testing.assert_allclose(result['second_reference_corners'],corners,atol=1e-12)
        np.testing.assert_array_equal(pins,old_pins)
        for a,b in zip(sides,old_sides):np.testing.assert_array_equal(a,b)
        np.testing.assert_array_equal(result['first_pinpoints'],pins)
        self.assertFalse(result['source_boundary_changed'])
        self.assertFalse(result['virtual_lines_are_printed_ink'])

    def test_folded_edge_samples_remain_in_crop_extent(self):
        corners,sides=self.fixture()
        sides[0]=sides[0].copy()
        sides[0][:25,1]-=np.linspace(.04,.01,25)
        material=np.array([[.3,.4],[.55,.6]])
        result=second_projection_frame(corners,sides,[1000,800],material_points=material)
        np.testing.assert_allclose(result['second_reference_corners'],corners,atol=1e-12)
        matrix=np.asarray(result['source_to_reference_matrix'])
        source=np.vstack([corners,*sides,material])
        mapped=transform_points(matrix,source)
        bounds=np.asarray(result['observed_source_bounds_in_reference'])
        self.assertTrue((mapped>=bounds[:2]-1e-12).all())
        self.assertTrue((mapped<=bounds[2:]+1e-12).all())
        self.assertLess(bounds[1],-.02)
        np.testing.assert_allclose(transform_points(np.linalg.inv(matrix),mapped),source,atol=1e-12)
        self.assertFalse(result['residual_dewarp_applied'])

    def test_all_four_side_angles_define_projective_square_without_card_ratio(self):
        corners,sides=self.fixture()
        result=second_projection_frame(corners,sides,[1000,800])
        np.testing.assert_allclose(transform_points(result['source_to_reference_matrix'],corners),
            [[0,0],[1,0],[1,1],[0,1]],atol=1e-12)
        self.assertEqual(len(result['side_angle_evidence']),4)
        self.assertNotIn('card_ratio',result)

    def test_missing_or_unreliable_side_cannot_be_invented(self):
        corners,sides=self.fixture()
        with self.assertRaises(ValueError):second_projection_frame(corners,sides[:3],[1000,800])
        sides[0]=np.repeat(corners[0][None],40,axis=0)
        with self.assertRaises(ValueError):second_projection_frame(corners,sides,[1000,800])

    def test_invalid_perimeter_or_frame_order_declines(self):
        corners,sides=self.fixture()
        with self.assertRaises(ValueError):second_projection_frame(corners[::-1],sides,[1000,800])
        sides[0][0,0]=np.nan
        with self.assertRaises(ValueError):second_projection_frame(corners,sides,[1000,800])


if __name__=='__main__':unittest.main()
