import unittest
import numpy as np
import cv2
from rule_join_evidence import extend_to_crossings,measured_straight_anchor,measured_chains,continued_thin_sections,registered_material_join


class RuleJoinEvidenceTests(unittest.TestCase):
    def test_native_ridge_estimates_bind_to_complete_original_stroke(self):
        x=np.arange(40,601)/799;trace=np.c_[x,.2+.0002*np.sin(np.arange(len(x))/70)]
        a=trace[:260].copy();b=trace[266:].copy()
        a[:,1]+=.5/399;b[:,1]-=.35/399
        registry=[{'index':3,'source':trace+[.01,.02],'reference':trace}]
        result=registered_material_join(registry,a,b,[],800,400)
        self.assertIsNotNone(result)
        self.assertEqual(result['source_indices'],[259,266])
        np.testing.assert_array_equal(result['source_points'],registry[0]['source'][[259,266]])
        self.assertIsNone(registered_material_join(registry*2,a,b,[],800,400))
        word=np.array([[a[-1,0],.195],[b[0,0],.205]])
        self.assertIsNone(registered_material_join(registry,a,b,[word],800,400))
        wrong=b.copy();wrong[:,1]+=.01
        self.assertIsNone(registered_material_join(registry,a,wrong,[],800,400))
        uncertain=b.copy();uncertain[:,0]+=.4/799
        self.assertIsNone(registered_material_join(registry,a,uncertain,[],800,400))

    def thin_stroke(self):
        h,w=300,800
        gray=np.full((h,w),235,np.uint8);safe=np.full_like(gray,255);words=np.zeros_like(gray)
        x=np.arange(400,761);y=105+4*np.sin(x/160)
        cv2.polylines(gray,[np.rint(np.c_[x,y]).astype(np.int32)],False,70,1)
        cv2.line(gray,(760,50),(760,240),70,1)
        a=np.c_[x[(x>=410)&(x<=500)]/799,y[(x>=410)&(x<=500)]/299]
        b=np.c_[x[(x>=660)&(x<=749)]/799,y[(x>=660)&(x<=749)]/299]
        cross=np.c_[np.full(191,760/799),np.arange(50,241)/299]
        return gray,safe,words,a,b,cross

    def test_complete_thin_component_recovers_two_observed_fragments_at_a_frame(self):
        gray,safe,words,a,b,cross=self.thin_stroke()
        result=continued_thin_sections(gray,safe,[a,b],[cross],words)
        self.assertEqual(len(result),1)
        self.assertLess(result[0][0,0],a[0,0])
        self.assertGreater(result[0][-1,0],b[-1,0])
        self.assertLess(abs(result[0][:,1]*299-(105+4*np.sin(result[0][:,0]*799/160))).max(),1)

    def test_word_or_blank_gap_keeps_thin_components_separate(self):
        gray,safe,words,a,b,cross=self.thin_stroke()
        words[95:115,550:565]=1
        self.assertEqual(continued_thin_sections(gray,safe,[a,b],[cross],words),[])
        words[:]=0;gray[:,550:565]=235
        self.assertEqual(continued_thin_sections(gray,safe,[a,b],[cross],words),[])

    def test_duplicate_fragment_and_branched_ink_cannot_authorize_continuation(self):
        gray,safe,words,a,b,cross=self.thin_stroke()
        self.assertEqual(continued_thin_sections(gray,safe,[a,a],[cross],words),[])
        cv2.line(gray,(580,80),(580,130),70,1)
        self.assertEqual(continued_thin_sections(gray,safe,[a,b],[cross],words),[])

    def test_subpixel_anchor_uses_the_exact_current_raster_ridge(self):
        gray=np.full((300,500),235,np.uint8);safe=np.full_like(gray,255);words=np.zeros_like(gray)
        cv2.line(gray,(40,150),(400,150),70,1)
        guide=np.c_[np.arange(50,391)/499,np.full(341,150.6/299)]
        result=measured_straight_anchor(gray,safe,guide,words)
        self.assertIsNotNone(result)
        np.testing.assert_allclose(result[:,1]*299,150,atol=1e-12)
        words[143:159,:]=1
        self.assertIsNone(measured_straight_anchor(gray,safe,guide,words))

    def test_real_ink_can_join_a_small_gap_to_an_established_straight_section(self):
        gray=np.full((400,800),235,np.uint8);safe=np.full_like(gray,255);words=np.zeros_like(gray)
        x=np.arange(40,601);y=np.where(x<=100,200,200+3*np.sin((x-100)/100))
        cv2.polylines(gray,[np.rint(np.c_[x,y]).astype(np.int32)],False,80,1)
        guide=np.c_[x[x>=100]/799,y[x>=100]/399]
        anchor=np.c_[np.arange(40,91)/799,np.full(51,200/399)]
        result=extend_to_crossings(gray,safe,guide,[],words,collinear=[anchor])
        np.testing.assert_array_equal(result[:len(anchor)],anchor)
        self.assertTrue((np.diff(result[:,0])>0).all())
        words[194:207,94:98]=1
        np.testing.assert_array_equal(extend_to_crossings(gray,safe,guide,[],words,collinear=[anchor]),guide)

    def test_joined_proposal_keeps_both_sections_and_declines_occlusion(self):
        gray=np.full((400,800),235,np.uint8);safe=np.full_like(gray,255);words=np.zeros_like(gray)
        x=np.arange(40,601);y=200+3*np.sin(x/100)
        cv2.polylines(gray,[np.rint(np.c_[x,y]).astype(np.int32)],False,80,1)
        a=np.c_[x[x<300]/799,y[x<300]/399]
        b=np.c_[x[x>310]/799,y[x>310]/399]
        result=measured_chains(gray,safe,[b,a],words)
        self.assertEqual(len(result),1)
        np.testing.assert_array_equal(result[0][:len(a)],a)
        np.testing.assert_array_equal(result[0][-len(b):],b)
        self.assertTrue((np.diff(result[0][:,0])>0).all())
        words[190:215,303:308]=1
        self.assertEqual(measured_chains(gray,safe,[a,b],words),[])
        words[:]=0;gray[:,300:311]=235
        self.assertEqual(measured_chains(gray,safe,[a,b],words),[])

    def test_single_observed_gap_pixel_can_connect_two_measured_fragments(self):
        gray=np.full((400,800),235,np.uint8);safe=np.full_like(gray,255);words=np.zeros_like(gray)
        cv2.line(gray,(40,200),(600,200),80,1)
        a=np.c_[np.arange(40,301)/799,np.full(261,200/399)]
        b=np.c_[np.arange(302,601)/799,np.full(299,200/399)]
        result=measured_chains(gray,safe,[a,b],words)
        self.assertEqual(len(result),1)
        self.assertTrue(np.any(abs(result[0][:,0]-301/799)<1e-14))
        words[197:204,301]=1
        self.assertEqual(measured_chains(gray,safe,[a,b],words),[])

    def evidence(self,row):
        gray=np.full((400,800),235,np.uint8);safe=np.full_like(gray,255);words=np.zeros_like(gray)
        x=np.arange(40,601);y=row+3*np.sin(x/100)
        cv2.polylines(gray,[np.rint(np.c_[x,y]).astype(np.int32)],False,80,1)
        cv2.line(gray,(40,row-40),(40,row+40),80,1)
        guide=np.c_[x[x>=85]/799,y[x>=85]/399]
        crossing=np.array([[40/799,(row-40)/399],[40/799,(row+40)/399]])
        return gray,safe,words,guide,crossing

    def test_supported_join_is_extended_at_different_page_heights(self):
        for row in [100,200,300]:
            gray,safe,words,guide,crossing=self.evidence(row)
            result=extend_to_crossings(gray,safe,guide,[crossing],words)
            self.assertAlmostEqual(result[0,0],40/799)
            self.assertGreater(len(result),len(guide))
            self.assertLess(abs(result[:45,1]*399-(row+3*np.sin(result[:45,0]*799/100))).max(),1.5)

    def test_blank_gap_and_independent_word_ink_cannot_supply_join(self):
        gray,safe,words,guide,crossing=self.evidence(200)
        words[190:215,60:75]=1
        np.testing.assert_array_equal(extend_to_crossings(gray,safe,guide,[crossing],words),guide)
        gray[:,40:85]=235;words[:]=0
        np.testing.assert_array_equal(extend_to_crossings(gray,safe,guide,[crossing],words),guide)


if __name__=='__main__':unittest.main()
