import unittest
import numpy as np
from word_rule_partition import partition_rule_fragments, partition_bracketed_rules, partition_supported_short_sections


class WordRulePartitionTests(unittest.TestCase):
    def short_cell_example(self):
        bodies=[]
        for x,y,cw,ch,removed in [(24,294,11,14,48),(39,295,25,13,131)]:
            yy,xx=np.meshgrid(np.arange(y,y+ch),np.arange(x,x+cw))
            points=np.c_[xx.ravel(),yy.ravel()]
            interior=np.flatnonzero((points[:,0]>x)&(points[:,0]<x+cw-1)&
                                     (points[:,1]>y)&(points[:,1]<y+ch-1))
            bodies.append(np.delete(points,interior[:removed],axis=0))
        glyphs=np.vstack(bodies)
        pieces=[]
        for x,cw,area in [(20,11,23),(33,10,19),(50,9,12)]:
            points=np.vstack([np.c_[np.arange(x,x+cw),np.full(cw,314)],
                              np.c_[np.arange(x,x+cw),np.full(cw,315)],
                              np.c_[np.arange(x,x+cw),np.full(cw,313)]])
            pieces.append(points[:area])
        fragments=np.vstack(pieces)
        group=np.vstack([glyphs,fragments])/[1050,1279]
        guide=np.c_[np.arange(14,558)/1050,np.full(544,314/1279)]
        vertical=np.c_[np.arange(100,1000)/1279,np.full(900,18/1050)]
        return group,glyphs/[1050,1279],fragments/[1050,1279],guide,vertical

    def test_short_first_cell_needs_real_row_and_frame_crossing(self):
        group,glyphs,fragments,guide,vertical=self.short_cell_example()
        self.assertEqual(len(group),354)
        self.assertGreater(len(fragments)/len(group),.15)
        core,roles,audit=partition_supported_short_sections([group],[guide],[vertical],1051,1280)
        np.testing.assert_array_equal(core[0],glyphs)
        np.testing.assert_array_equal(roles[0]['points'],fragments)
        self.assertEqual(roles[0]['components'],3)
        self.assertEqual(sorted(audit[0]['glyph_point_indices']+roles[0]['point_indices']),list(range(len(group))))

    def test_short_flank_without_supported_frame_join_stays_word_owned(self):
        group,_,_,guide,vertical=self.short_cell_example()
        for hs,vs in [([guide],[]),([guide],[vertical+[0,.08]]),
                      ([guide+[0,.02]],[vertical]),([guide[20:200]],[vertical])]:
            core,roles,_=partition_supported_short_sections([group],hs,vs,1051,1280)
            np.testing.assert_array_equal(core[0],group)
            self.assertEqual(roles,[])

    def test_exact_pixel_distance_boundary_is_inclusive_without_an_epsilon(self):
        group,glyphs,_,guide,vertical=self.short_cell_example()
        exact=guide.copy();exact[:,1]=312/1279
        core,roles,_=partition_supported_short_sections([group],[exact],[vertical],1051,1280)
        np.testing.assert_array_equal(core[0],glyphs)
        self.assertEqual(len(roles),1)
        beyond=exact.copy();beyond[:,1]=311.99/1279
        core,roles,_=partition_supported_short_sections([group],[beyond],[vertical],1051,1280)
        np.testing.assert_array_equal(core[0],group)
        self.assertEqual(roles,[])

    def test_two_full_row_flanks_can_support_a_middle_label_without_frame(self):
        group,glyphs,fragments,guide,_=self.short_cell_example()
        offset=np.array([200/1050,0]);group=group+offset
        guide=np.c_[np.arange(14,800)/1050,np.full(786,314/1279)]
        core,roles,_=partition_supported_short_sections([group],[guide],[],1051,1280)
        np.testing.assert_array_equal(core[0],glyphs+offset)
        np.testing.assert_array_equal(roles[0]['points'],fragments+offset)

    def test_contact_punctuation_and_only_two_pieces_cannot_transfer(self):
        group,glyphs,fragments,guide,vertical=self.short_cell_example()
        contact=np.c_[np.full(7,24),np.arange(307,314)]/[1050,1279]
        dots=np.vstack([np.array([[x,314],[x+1,314],[x,315],[x+1,315]]) for x in (20,33,50)])/[1050,1279]
        for actual in [np.vstack([group,contact]),np.vstack([glyphs,dots]),
                       np.vstack([glyphs,fragments[:-12]])]:
            core,roles,_=partition_supported_short_sections([actual],[guide],[vertical],1051,1280)
            np.testing.assert_array_equal(core[0],actual)
            self.assertEqual(roles,[])

    def band_example(self):
        glyphs=[]
        for x in (200,225,250,275):
            yy,xx=np.meshgrid(np.arange(120,144),np.arange(x,x+10))
            glyphs.append(np.c_[xx.ravel(),yy.ravel()])
        glyphs=np.vstack(glyphs)
        bands=[]
        for begin,end in ((190,274),(275,310)):
            yy,xx=np.meshgrid(np.arange(110,116),np.arange(begin,end))
            bands.append(np.c_[xx.ravel(),yy.ravel()])
        band=np.vstack(bands)
        group=np.vstack([glyphs,band])/[1279,799]
        left=np.c_[np.arange(100,190)/1279,np.full(90,112.5/799)]
        right=np.c_[np.arange(310,440)/1279,np.full(130,112.5/799)]
        return group,glyphs/[1279,799],band/[1279,799],[left,right]

    def test_dense_unequal_rule_pieces_need_real_flanks_and_keep_every_glyph(self):
        group,glyphs,band,guides=self.band_example()
        core,printed,audit=partition_bracketed_rules([group],guides,1280,800)
        np.testing.assert_array_equal(core[0],glyphs)
        np.testing.assert_array_equal(printed[0]['points'],band)
        self.assertEqual(printed[0]['left_guide_index'],0)
        self.assertEqual(printed[0]['right_guide_index'],1)
        ids=audit[0]['glyph_point_indices']+printed[0]['point_indices']
        self.assertEqual(sorted(ids),list(range(len(group))))

    def test_a_full_measured_rule_supplies_two_disjoint_observed_flanks(self):
        group,glyphs,band,_=self.band_example()
        guide=np.c_[np.arange(100,440)/1279,np.full(340,112.5/799)]
        core,printed,_=partition_bracketed_rules([group],[guide],1280,800)
        np.testing.assert_array_equal(core[0],glyphs)
        np.testing.assert_array_equal(printed[0]['points'],band)

    def test_one_flank_distant_flanks_and_sparse_punctuation_keep_ownership(self):
        group,glyphs,band,guides=self.band_example()
        sparse=np.vstack([glyphs,band[::20]])
        for original,traces in [(group,guides[:1]),(group,[q+[0,.03] for q in guides]),(sparse,guides)]:
            core,printed,_=partition_bracketed_rules([original],traces,1280,800)
            np.testing.assert_array_equal(core[0],original)
            self.assertEqual(printed,[])

    def test_touching_letter_and_a_branched_band_stay_word_owned(self):
        group,_,_,guides=self.band_example()
        for extra in [np.c_[np.full(6,200),np.arange(115,121)],
                      np.c_[np.full(13,215),np.arange(110,123)]]:
            original=np.vstack([group,extra/[1279,799]])
            core,printed,_=partition_bracketed_rules([original],guides,1280,800)
            np.testing.assert_array_equal(core[0],original)
            self.assertEqual(printed,[])

    def example(self):
        glyphs=[]
        for x in (190,210,230,250):
            yy,xx=np.meshgrid(np.arange(90,102),np.arange(x,x+8))
            glyphs.append(np.c_[xx.ravel(),yy.ravel()])
        glyphs=np.vstack(glyphs)
        dashes=np.vstack([np.c_[np.arange(x,x+7),np.full(7,116)] for x in (194,209)])
        group=np.vstack([glyphs,dashes])/[799,399]
        rule=np.c_[np.arange(232,400)/799,np.full(168,116/399)]
        return group,glyphs/[799,399],dashes/[799,399],rule

    def test_nearby_repeated_divider_fragments_transfer_every_original_point(self):
        group,glyphs,dashes,rule=self.example()
        core,printed,audit=partition_rule_fragments([group],[rule],800,400)
        np.testing.assert_array_equal(core[0],glyphs)
        np.testing.assert_array_equal(printed[0]['points'],dashes)
        self.assertEqual(printed[0]['components'],2)
        self.assertTrue(audit[0]['all_points_accounted'])
        ids=audit[0]['glyph_point_indices']+printed[0]['point_indices']
        self.assertEqual(sorted(ids),list(range(len(group))))

    def test_single_dash_and_small_punctuation_keep_word_ownership(self):
        group,glyphs,dashes,rule=self.example()
        for tail in (dashes[:7],np.array([[194/799,116/399],[209/799,116/399]])):
            original=np.vstack([glyphs,tail])
            core,printed,_=partition_rule_fragments([original],[rule],800,400)
            np.testing.assert_array_equal(core[0],original)
            self.assertEqual(printed,[])

    def test_glyph_contact_and_distant_rule_cannot_transfer_ink(self):
        group,glyphs,dashes,rule=self.example()
        for changed,trace in [(np.vstack([glyphs,dashes-[0,14/399]]),rule-[0,14/399]),
                              (group,rule+[.2,0])]:
            core,printed,_=partition_rule_fragments([changed],[trace],800,400)
            np.testing.assert_array_equal(core[0],changed)
            self.assertEqual(printed,[])

    def test_archive_pixel_edge_basis_keeps_paired_source_indices(self):
        group,glyphs,dashes,rule=self.example()
        edge=(group*[799,399]+.5)/[800,400]
        core,printed,audit=partition_rule_fragments([edge],[rule],800,400,pixel_basis='edges')
        self.assertEqual(len(core[0]),len(glyphs))
        self.assertEqual(len(printed[0]['points']),len(dashes))
        self.assertTrue(audit[0]['all_points_accounted'])


if __name__=='__main__':unittest.main()
