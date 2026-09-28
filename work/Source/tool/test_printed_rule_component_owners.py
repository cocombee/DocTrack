import copy
import unittest
import numpy as np
from printed_rule_component_owners import orthogonal_rule_components


class PrintedRuleComponentTests(unittest.TestCase):
    def setUp(self):
        self.clouds=[np.array([[.199,y],[.201,y],[.199,y+.015],[.201,y+.015]])
                     for y in (.03,.065,.10,.14)]
        self.guide=np.c_[np.full(256,.2),np.linspace(.01,.95,256)]

    def roles(self,clouds=None,groups=None,guides=None,native_start=4):
        return orthogonal_rule_components(groups or [[0,1,2,3]],
            self.clouds if clouds is None else clouds,native_start,
            [self.guide] if guides is None else guides,1280,1280)

    def test_complete_thin_proxy_pieces_keep_exact_identity_and_points(self):
        before=copy.deepcopy(self.clouds)
        result=self.roles()
        self.assertEqual(len(result),1)
        self.assertEqual(result[0]['component_indices'],[0,1,2,3])
        self.assertEqual(result[0]['normal_axis'],0)
        for a,b in zip(before,self.clouds):np.testing.assert_array_equal(a,b)

    def test_native_owned_components_are_never_reclassified(self):
        self.assertEqual(self.roles(native_start=3),[])

    def test_missing_curved_or_distant_rule_is_insufficient(self):
        curved=self.guide.copy();curved[:,0]+=.001*np.sin(curved[:,1]*10)
        distant=self.guide.copy();distant[:,0]+=.01
        for guides in ([],[curved],[distant],[self.guide[:20]]):
            self.assertEqual(self.roles(guides=guides),[])

    def test_glyph_shaped_and_short_groups_remain_rigid(self):
        self.assertEqual(self.roles(groups=[[0,1]]),[])
        thick=copy.deepcopy(self.clouds);thick[0][1::2,0]+=.006
        self.assertEqual(self.roles(clouds=thick),[])
        short=copy.deepcopy(self.clouds);short[0][2:,1]=short[0][:2,1]+.002
        self.assertEqual(self.roles(clouds=short),[])
        square=copy.deepcopy(self.clouds);square[0][2:,1]=square[0][:2,1]+.003
        self.assertEqual(self.roles(clouds=square),[])


if __name__=='__main__':unittest.main()
