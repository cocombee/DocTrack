import unittest
import numpy as np
from rigid_word_flow import backward


class RigidFlowTests(unittest.TestCase):
    def test_empty_corner_of_slanted_word_box_is_available_stock(self):
        body=np.array([[.2,.173],[.3,.173],[.3,.183],[.2,.183]])
        x=np.arange(200,301)
        fixed=np.c_[x,160+.1*(x-200)]/1000
        box=np.array([[.2,.16],[.3,.16],[.3,.17],[.2,.17]])
        angle=-.08;center=np.array([250,178]);scale=np.array([1000,1000])
        c,s=np.cos(angle),np.sin(angle);rotation=np.array([[c,-s],[s,c]])
        moved=((body*scale-center)@rotation+center)/scale
        query=np.vstack([moved,fixed])
        legacy=backward(query,body,[box],[],center,angle,scale,
                        constraint_padding=1)
        actual=backward(query,body,[],[],center,angle,scale,
                        constraint_padding=1,fixed_ink=[fixed])
        self.assertGreater(abs(legacy[:4]-body).max()*1000,1)
        np.testing.assert_allclose(actual[:4],body,atol=1e-7)
        np.testing.assert_array_equal(actual[4:],fixed)

    def test_owned_ink_rotates_exactly_and_fixed_word_stays_stationary(self):
        body=np.array([[.2,.2],[.3,.2],[.3,.22],[.2,.22]])
        fixed=np.array([[.2,.17],[.3,.17],[.3,.18],[.2,.18]])
        angle=-.04;center=np.array([250,210]);scale=np.array([1000,1000])
        c,s=np.cos(angle),np.sin(angle);rot=np.array([[c,-s],[s,c]])
        moved=((body*scale-center)@rot+center)/scale
        result=backward(np.vstack([moved,fixed]),body,[fixed],[],center,angle,scale)
        np.testing.assert_allclose(result[:4],body,atol=1e-7)
        np.testing.assert_array_equal(result[4:],fixed)

    def test_complete_rule_and_points_outside_support_remain_fixed(self):
        body=np.array([[.2,.2],[.3,.2],[.3,.22],[.2,.22]])
        rule=np.c_[np.linspace(.15,.35,50),np.full(50,.25)]
        outside=np.array([[.7,.7],[.8,.8]])
        query=np.vstack([rule,outside])
        result=backward(query,body,[],[rule],[250,210],-.04,[1000,1000])
        np.testing.assert_array_equal(result,query)

    def test_faint_edges_in_the_body_halo_share_the_whole_word_rotation(self):
        body=np.array([[.2,.2],[.3,.2],[.3,.22],[.2,.22]])
        faint=np.array([[.197,.205],[.303,.215]])
        fixed=np.array([[.2,.17],[.3,.17],[.3,.18],[.2,.18]])
        scale=np.array([1000,1000]);center=np.array([250,210]);angle=-.04
        c,s=np.cos(angle),np.sin(angle);rotation=np.array([[c,-s],[s,c]])
        moved=((faint*scale-center)@rotation+center)/scale
        actual=backward(np.vstack([moved,fixed]),body,[fixed],[],center,angle,scale,
                        body_padding=4,constraint_padding=4)
        np.testing.assert_allclose(actual[:2],faint,atol=1e-7)
        np.testing.assert_array_equal(actual[2:],fixed)

    def test_dense_query_exceeding_short_dimensions_remains_supported(self):
        body=np.array([[.2,.2],[.3,.2],[.3,.22],[.2,.22]])
        rule=np.c_[np.linspace(.15,.35,50),np.full(50,.25)]
        query=np.tile([.7,.7],(35000,1))
        result=backward(query,body,[],[rule],[250,210],-.04,[1000,1000],steps=8)
        np.testing.assert_array_equal(result,query)


if __name__=='__main__':unittest.main()
