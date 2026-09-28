import unittest
import cv2
import numpy as np
from whole_text_row_motion import measured_row,query_for_row
from local_receipt_mesh import sample


class WholeTextRowMotionTests(unittest.TestCase):
    def fixture(self):
        gray=np.full((400,800),240,np.uint8);groups=[];quads=[]
        for x in (110,230,350,470,590):
            y=round(185+.00013*(x-350)**2)
            for left in range(x,x+70,12):gray[y:y+16,left:left+6]=30
            yy,xx=np.where(gray[y:y+16,x:x+70]<240)
            g=np.c_[xx+x,yy+y]/[799,399];groups.append(g)
            a,z=g.min(0),g.max(0)
            quads.append(np.array([[a[0],z[1]],z,[z[0],a[1]],a]))
        return gray,np.full_like(gray,255),quads,groups

    def test_one_row_field_preserves_every_word_and_joins_real_blank_gaps_smoothly(self):
        gray,safe,quads,groups=self.fixture();original=gray.copy()
        proof=measured_row(gray,safe,quads,groups)
        self.assertIsNotNone(proof)
        self.assertFalse(proof['ink_or_owner_points_changed'])
        query,audit=query_for_row(groups,proof,[799,399],[],resolution=(1025,1601))
        self.assertIsNotNone(query)
        self.assertTrue(audit['smooth_real_blank_gaps'])
        self.assertTrue(np.all(np.diff(query[:,:,1],axis=0)>0))
        for owner,dy in zip(proof['owners'],proof['backward_normal_px']):
            destination=groups[owner]-[0,dy/399]
            # Includes both complete physical half-pixel edges of each glyph.
            for offset in ((-.5,-.5),(.5,-.5),(.5,.5),(-.5,.5)):
                source=groups[owner]+np.asarray(offset)/[799,399]
                actual=sample(query,destination+np.asarray(offset)/[799,399])
                np.testing.assert_allclose(actual,source,atol=1e-12)
        np.testing.assert_array_equal(gray,original)

    def test_missing_second_contrast_profile_or_unsafe_word_abstains(self):
        gray,safe,quads,groups=self.fixture()
        weak=gray.copy();weak[gray<240]=232
        self.assertIsNone(measured_row(weak,safe,quads,groups))
        x,y=np.rint(groups[0][0]*[799,399]).astype(int);safe[y,x]=0
        self.assertIsNone(measured_row(gray,safe,quads,groups))

    def test_descenders_can_be_fit_support_outliers_without_losing_their_ink(self):
        gray,safe,quads,groups=self.fixture()
        changed=[]
        for i,g in enumerate(groups):
            points=np.rint(g*[799,399]).astype(int)
            left=int(points[:,0].min());bottom=int(points[:,1].max())
            if i%2==0:gray[bottom+1:bottom+4,left:left+6]=30
            yy,xx=np.where(gray[bottom-15:bottom+4,left:left+70]<240)
            changed.append(np.c_[xx+left,yy+bottom-15]/[799,399])
        counts=[len(g) for g in changed]
        proof=measured_row(gray,safe,quads,changed)
        self.assertIsNotNone(proof)
        self.assertEqual([len(g) for g in changed],counts)
        query,audit=query_for_row(changed,proof,[799,399],[],resolution=(1025,800))
        self.assertIsNotNone(query)
        for i,dy in zip(proof['owners'],proof['backward_normal_px']):
            for offset in ((-.5,-.5),(.5,-.5),(.5,.5),(-.5,.5)):
                corners=changed[i]+np.asarray(offset)/[799,399]
                np.testing.assert_allclose(sample(query,corners-[0,dy/399]),corners,atol=1e-12)

    def test_crossing_print_or_overlapping_word_bodies_cannot_receive_partial_motion(self):
        gray,safe,quads,groups=self.fixture();proof=measured_row(gray,safe,quads,groups)
        crossing=np.array([[.4,.4],[.4,.6]])
        query,audit=query_for_row(groups,proof,[799,399],[dict(points=crossing.tolist())],resolution=(257,801))
        self.assertIsNone(query)
        self.assertIn('unbound print',audit['rejected'])
        changed=[g.copy() for g in groups];changed[1][:,0]-=.14
        query,audit=query_for_row(changed,proof,[799,399],[],resolution=(257,801))
        self.assertIsNone(query)
        self.assertIn('overlap',audit['rejected'])

    def test_unordered_ink_cannot_invent_a_stroke_through_blank_stock(self):
        gray,safe,quads,groups=self.fixture();proof=measured_row(gray,safe,quads,groups)
        separated=np.array([[.4,.35],[.4,.7]])
        query,audit=query_for_row(groups,proof,[799,399],
            [dict(points=separated.tolist(),geometry='ink_cloud')],resolution=(1025,1601))
        self.assertIsNotNone(query)
        self.assertTrue(audit['accepted'])
        np.testing.assert_array_equal(sample(query,separated),separated)


if __name__=='__main__':unittest.main()
