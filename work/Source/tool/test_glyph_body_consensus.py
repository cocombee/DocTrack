import unittest
import cv2
import numpy as np
from rigid_word_row import glyph_body_consensus,word_rotation_evidence


class GlyphBodyConsensusTests(unittest.TestCase):
    def evidence(self,text,angle):
        gray=np.full((600,1000),230,np.uint8)
        cv2.putText(gray,text,(260,310),cv2.FONT_HERSHEY_SIMPLEX,1,70,2)
        transform=cv2.getRotationMatrix2D((400,300),angle,1)
        gray=cv2.warpAffine(gray,transform,(1000,600),borderValue=230)
        gray=cv2.GaussianBlur(gray,(3,3),.7)
        contrast=cv2.GaussianBlur(gray.astype(np.float32),(0,0),5.5)-gray
        y,x=np.where(contrast>6)
        return gray,np.c_[x/999,y/599]

    def test_two_independent_ink_levels_recover_known_word_angle(self):
        gray,group=self.evidence('DOCUMENT',3)
        evidence=glyph_body_consensus(gray,group,pixel_size=[999,599])
        self.assertIsNotNone(evidence)
        self.assertAlmostEqual(evidence['angle'],-np.radians(3),delta=.004)
        self.assertEqual(len(evidence['independent_profiles']),2)
        self.assertNotEqual(evidence['independent_profiles'][0]['threshold'],evidence['independent_profiles'][1]['threshold'])
        self.assertTrue(all(len(profile['points'])>=6 for profile in evidence['independent_profiles']))

    def test_flat_and_short_words_cannot_supply_a_rotation(self):
        for text,angle in [('DOCUMENT',0),('BODY',3)]:
            gray,group=self.evidence(text,angle)
            self.assertIsNone(glyph_body_consensus(gray,group,pixel_size=[999,599]))
            self.assertIsNone(word_rotation_evidence(gray,group,
                                   pixel_size=[999,599],body_consensus=True))

    def test_consistent_baseline_does_not_require_matching_cap_heights(self):
        gray=np.full((600,1000),230,np.uint8)
        # All nine independent glyphs share a known slanted baseline. Their
        # different cap heights cannot supply the separate body-center proof.
        for i,height in enumerate(range(15,24)):
            x=260+i*30;bottom=310+i
            cv2.rectangle(gray,(x,bottom-height+1),(x+8,bottom),70,-1)
        y,x=np.where(gray<100);group=np.c_[x/999,y/599]
        self.assertIsNone(glyph_body_consensus(gray,group,pixel_size=[999,599]))
        evidence=word_rotation_evidence(gray,group,pixel_size=[999,599],
                                        body_consensus=True)
        self.assertIsNotNone(evidence)
        self.assertAlmostEqual(evidence['angle'],np.arctan(1/30),delta=1e-12)
        self.assertEqual(len(evidence['points']),9)
        self.assertLess(evidence['scatter_px'],1e-10)


if __name__=='__main__':unittest.main()
