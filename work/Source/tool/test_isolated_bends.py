import unittest
import numpy as np
from isolated_rule_bends import before_stable_tail, anchored_remainder
from joint_stock_projection import crossing_pins, project
from local_receipt_mesh import sample, valid


class IsolatedBendTests(unittest.TestCase):
    def test_bend_uses_its_measured_join_and_excludes_straight_tail(self):
        t=np.linspace(.05,.4,120)
        normal=np.interp(t,[.05,.16,.4],[.04,.02,.02])
        guide=np.c_[t,normal]
        prefix,target=before_stable_tail(guide,1280)
        self.assertLess(len(prefix),len(guide)/2)
        np.testing.assert_array_equal(prefix,guide[:len(prefix)])
        self.assertEqual(target,prefix[-1,1])
        self.assertGreater(prefix[0,1]-target,.01)

    def test_no_straight_anchor_or_no_bend_declines(self):
        t=np.linspace(.05,.4,120)
        for normal in [t*.2,np.full_like(t,.02)]:
            self.assertIsNone(before_stable_tail(np.c_[t,normal],1280))

    def test_continuation_excludes_the_working_anchor(self):
        t=np.linspace(.05,.22,200)
        normal=np.interp(t,[.05,.095,.16,.22],[.02,.02,.018,.02])
        anchor=np.c_[np.linspace(.036,.095,76),np.full(76,.02)]
        remaining,target=anchored_remainder(np.c_[t,normal],[anchor],1280,1051)
        self.assertEqual(target,.02)
        self.assertGreater(remaining[:,0].min(),.095+1/1280)
        self.assertLess(remaining[:,1].min(),.019)
        self.assertIsNone(anchored_remainder(np.c_[t,np.full_like(t,.02)],[anchor],1280,1051))

    def test_a_distant_or_bent_anchor_cannot_drive_the_continuation(self):
        t=np.linspace(.05,.22,200);guide=np.c_[t,.02+.01*np.sin(t*20)]
        for anchor in [np.c_[t,t*.02],np.c_[t,np.full_like(t,.5)]]:
            self.assertIsNone(anchored_remainder(guide,[anchor],1280,1051))

    def test_stock_repair_preserves_ink_and_outside_region(self):
        x,y=np.meshgrid(np.linspace(0,1,33),np.linspace(0,1,33))
        identity=np.stack([x,y],-1)
        candidate=identity.copy();candidate[16,16,0]-=.045
        self.assertFalse(valid(candidate))
        positions=np.array([[.2,.2],[.8,.2],[.2,.8],[.8,.8]])
        boxes=np.array([[[.1,.1],[.3,.1],[.3,.3],[.1,.3]]])
        result=project(candidate,boxes,np.array([0.]),positions,
                       sample(identity,positions),(.04,.04,.96,.96),1000)
        self.assertTrue(valid(result))
        np.testing.assert_array_equal(result[3:11,3:11],candidate[3:11,3:11])
        np.testing.assert_array_equal(result[:2],candidate[:2])
        np.testing.assert_array_equal(result[-2:],candidate[-2:])
        np.testing.assert_allclose(sample(result,positions),positions,atol=1e-11)

    def test_crossing_keeps_normal_and_finds_actual_source_tangent(self):
        x,y=np.meshgrid(np.linspace(0,1,33),np.linspace(0,1,33))
        candidate=np.stack([x,y+.03*x],-1)
        before=np.array([[.5,.2],[.5,.7]])
        source=before.copy()
        positions,targets=crossing_pins(candidate,np.array([[.1,.1]]),
            sample(candidate,np.array([[.1,.1]])),before,source,[2],[0])
        np.testing.assert_array_equal(positions[1:,0],before[:,0])
        np.testing.assert_allclose(sample(candidate,positions),targets,atol=1e-11)


if __name__=='__main__':
    unittest.main()
