import unittest
import numpy as np
from local_receipt_mesh import sample, valid
from precise_mesh_inverse import inverse
from rigid_word_row import find_rows, refine, refine_word, glyph_baseline, parallel_blocks, shared_motion_owners
import cv2


class RigidWordRowTests(unittest.TestCase):
    def setUp(self):
        u,v=np.meshgrid(np.linspace(0,1,1025),np.linspace(0,1,1025))
        self.old=np.stack([u,v],-1)
        self.quads=[]
        self.groups=[]
        angle=.015
        for x in [200,280,360]:
            q=np.array([[x,200+angle*x],[x+55,200+angle*(x+55)],
                        [x+55,216+angle*(x+55)],[x,216+angle*x]])/1000
            self.quads.append(q)
            self.groups.append(np.vstack([q,q.mean(0)]))
        self.groups.append(np.array([[.22,.27],[.27,.27],[.27,.285],[.22,.285]]))

    def test_row_levels_without_changing_word_lengths_or_surrounding_print(self):
        rule={'normal_axis':1,'points':np.c_[np.linspace(.16,.46,100),np.full(100,.185)].tolist()}
        result,audit=refine(self.old,self.quads,self.groups,[1000,1000],[rule])
        self.assertTrue(audit['accepted'],audit)
        self.assertTrue(valid(result))
        for i,g in enumerate(self.groups):
            actual=inverse(result,sample(self.old,g))
            distance=lambda p:np.linalg.norm((p[:,None]-p[None,:])*1000,axis=2)
            self.assertLess(abs(distance(actual)-distance(g)).max(),.1)
            if i<3:self.assertLess(abs(actual[2,1]-actual[3,1])*1000,.1)
            else:self.assertLess(abs(actual-g).max()*1000,.1)
        actual=inverse(result,np.asarray(rule['points']))
        self.assertLess(abs(actual-np.asarray(rule['points'])).max()*1000,.1)

    def test_conflicting_word_ownership_retains_original_map(self):
        result,audit=refine(self.old,self.quads,[*self.groups,self.groups[0]],[1000,1000],[])
        self.assertIs(result,self.old)
        self.assertEqual(audit['rejected'],'ambiguous independent word ownership')

    def measured_word(self):
        gray=np.full((1000,1000),240,np.uint8);pixels=[]
        for x in range(200,361,20):
            y=round(220-.04*x)
            gray[y-15:y+1,x:x+8]=50
            xx,yy=np.meshgrid(np.arange(x,x+8),np.arange(y-15,y+1))
            pixels.extend(np.c_[xx.ravel(),yy.ravel()].tolist())
        return gray,np.asarray(pixels)/999

    def test_repeated_material_observations_move_rigidly_without_losing_any_owner(self):
        gray,group=self.measured_word();duplicate=group[::2].copy()
        fixed=self.groups[-1];gray[270:286,220:271]=50
        groups=[group,duplicate,fixed];original=[g.copy() for g in groups]
        result,audit=refine_word(self.old,gray,groups,0,[],pixel_size=[999,999],
                                measured_motion_owners=True,precise_fixed_ink=True)
        self.assertTrue(audit['accepted'],audit)
        self.assertEqual(audit['independent_word_owners'],[0,1])
        for index,g in enumerate(groups):
            actual=inverse(result,sample(self.old,g))
            pairs=lambda q:np.linalg.norm((q[:,None]-q[None,:])*999,axis=2)
            ids=np.linspace(0,len(g)-1,min(50,len(g)),dtype=int)
            self.assertLess(abs(pairs(actual[ids])-pairs(g[ids])).max(),.1)
            if index==2:self.assertLess(abs(actual-g).max()*999,.1)
            np.testing.assert_array_equal(g,original[index])

    def test_motion_binding_uses_shared_ink_instead_of_blank_stock_or_envelopes(self):
        gray,group=self.measured_word()
        blank=np.c_[np.arange(210,216),np.full(6,205)]/999
        same_box=np.c_[np.arange(212,216),np.full(4,209)]/999
        gray[209,212:216]=50
        self.assertEqual(shared_motion_owners(gray,[group,group[::3],blank,same_box],0),[0,1])

    def test_single_measured_owner_is_not_ambiguous_with_unrelated_print_in_its_box(self):
        gray,group=self.measured_word()
        xx,yy=np.meshgrid(np.arange(212,216),np.arange(204,208))
        other=np.c_[xx.ravel(),yy.ravel()]/999;gray[204:208,212:216]=50
        result,audit=refine_word(self.old,gray,[group,other],0,[],pixel_size=[999,999],
            measured_motion_owners=True,precise_fixed_ink=True,flow_body_padding=4,stock_padding=32)
        self.assertNotEqual(audit.get('rejected'),'ambiguous independent word ownership')
        self.assertEqual(audit['motion_owner_binding']['owners'],[0])
        if audit['accepted']:
            self.assertLess(abs(inverse(result,sample(self.old,other))-other).max()*999,.1)
        else:self.assertIs(result,self.old)

    def test_word_motion_binding_rejects_missing_print_and_invalid_indices(self):
        gray,group=self.measured_word()
        for index in (-1,1,True):
            with self.assertRaises(ValueError):shared_motion_owners(gray,[group],index)
        with self.assertRaises(ValueError):shared_motion_owners(np.full_like(gray,240),[group],0)

    def test_disagreeing_angles_cannot_drive_a_shared_row(self):
        quads=[q.copy() for q in self.quads]
        quads[-1][2,1]+=.01
        result,audit=refine(self.old,quads,self.groups,[1000,1000],[])
        self.assertIs(result,self.old)
        self.assertEqual(audit['rejected'],'row angle unsupported')

    def test_missing_evidence_retains_original_map(self):
        result,audit=refine(self.old,[],self.groups,[1000,1000],[])
        self.assertIs(result,self.old)
        self.assertFalse(audit['accepted'])

    def test_row_association_requires_adjacency_and_shared_baseline(self):
        words=[{'confidence':1,'points':q.tolist()} for q in self.quads]
        far=self.quads[0]+[.5,0]
        words.append({'confidence':1,'points':far.tolist()})
        words.append({'confidence':.3,'points':self.quads[0].tolist()})
        rows=find_rows(words,[1000,1000])
        self.assertEqual(len(rows),1)
        self.assertEqual(len(rows[0]),3)

    def test_single_word_requires_coherent_individual_glyphs(self):
        gray=np.full((1000,1000),240,np.uint8)
        pixels=[]
        for x in range(200,361,20):
            y=round(220-.04*x)
            xx,yy=np.meshgrid(np.arange(x,x+8),np.arange(y-15,y+1))
            pixels.extend(np.c_[xx.ravel(),yy.ravel()].tolist())
        group=(np.asarray(pixels)+.5)/1000
        evidence=glyph_baseline(gray,group)
        self.assertGreater(evidence['explained'],.9)
        self.assertGreater(evidence['excursion_px'],5)
        result,audit=refine_word(self.old,gray,[group,self.groups[-1]],0,[])
        self.assertTrue(audit['accepted'],audit)
        actual=inverse(result,sample(self.old,group))
        angle=np.arctan2(actual[-1,1]-actual[127,1],actual[-1,0]-actual[127,0])
        self.assertLess(abs(angle),.005)

    def test_unordered_rule_ink_cannot_invent_chords_through_a_word(self):
        gray=np.full((1000,1000),240,np.uint8);pixels=[]
        for x in range(200,361,20):
            y=round(220-.04*x)
            xx,yy=np.meshgrid(np.arange(x,x+8),np.arange(y-15,y+1))
            pixels.extend(np.c_[xx.ravel(),yy.ravel()].tolist())
        group=(np.asarray(pixels)+.5)/1000
        # Row-major pixels from two real vertical strokes. Connecting their
        # consecutive samples invents a ladder across the intervening word.
        cloud=np.array([[x,y] for y in range(185,241) for x in (180,390)])/1000
        ruler={'normal_axis':None,'geometry':'ink_cloud','points':cloud.tolist()}
        result,audit=refine_word(self.old,gray,[group],0,[ruler])
        self.assertTrue(audit['accepted'],audit)
        self.assertLess(abs(inverse(result,sample(self.old,cloud))-cloud).max()*1000,.1)
        actual=inverse(result,sample(self.old,group))
        pairs=lambda a:np.linalg.norm((a[:,None]-a[None,:])*1000,axis=2)
        self.assertLess(abs(pairs(actual[::20])-pairs(group[::20])).max(),.1)
        trace={k:v for k,v in ruler.items() if k!='geometry'}
        declined,report=refine_word(self.old,gray,[group],0,[trace])
        self.assertIs(declined,self.old)
        self.assertFalse(report['accepted'])

    def test_short_word_with_insufficient_glyph_support_stays_fixed(self):
        gray=np.full((1000,1000),240,np.uint8)
        result,audit=refine_word(self.old,gray,self.groups,0,[])
        self.assertIs(result,self.old)
        self.assertEqual(audit['rejected'],'independent glyph baseline unsupported')

    def test_leading_pivot_levels_word_without_moving_surrounding_print(self):
        gray=np.full((1000,1000),240,np.uint8);pixels=[]
        for x in range(200,361,20):
            y=round(220-.04*x)
            xx,yy=np.meshgrid(np.arange(x,x+8),np.arange(y-15,y+1))
            pixels.extend(np.c_[xx.ravel(),yy.ravel()].tolist())
        group=(np.asarray(pixels)+.5)/1000
        fixed=self.groups[-1]
        result,audit=refine_word(self.old,gray,[group,fixed],0,[],
                                 rotation_anchor='leading')
        self.assertTrue(audit['accepted'],audit)
        actual=inverse(result,sample(self.old,group))
        distances=lambda p:np.linalg.norm((p[:,None]-p[None,:])*1000,axis=2)
        self.assertLess(abs(distances(actual)-distances(group)).max(),.1)
        np.testing.assert_allclose(inverse(result,sample(self.old,fixed)),fixed,atol=1e-7)
        self.assertEqual(audit['rotation_anchor'],'leading')
        rejected,report=refine_word(self.old,gray,[group,fixed],0,[],
                                    rotation_anchor='unmeasured')
        self.assertIs(rejected,self.old)
        self.assertFalse(report['accepted'])

    def test_cv_measurement_basis_uses_the_same_actual_glyph_pixels(self):
        gray=np.full((1000,1000),240,np.uint8);pixels=[]
        for x in range(200,361,20):
            y=round(220-.04*x)
            xx,yy=np.meshgrid(np.arange(x,x+8),np.arange(y-15,y+1))
            pixels.extend(np.c_[xx.ravel(),yy.ravel()].tolist())
        pixels=np.asarray(pixels)
        expected=glyph_baseline(gray,(pixels+.5)/1000)
        actual=glyph_baseline(gray,pixels/999,pixel_size=[999,999])
        self.assertEqual(actual,expected)

    def test_parallel_rows_form_one_block_only_with_shared_local_evidence(self):
        below=[q+[0,.025] for q in self.quads]
        blocks=parallel_blocks([self.quads,below],[1000,1000])
        self.assertEqual(len(blocks),1)
        self.assertEqual(len(blocks[0]),6)
        self.assertEqual(parallel_blocks([self.quads,[q+[0,.15] for q in below]],[1000,1000]),[])
        self.assertEqual(parallel_blocks([self.quads,[q+[.5,0] for q in below]],[1000,1000]),[])
        changed=[q.copy() for q in below]
        for q in changed:q[[1,2],1]+=.002
        self.assertEqual(parallel_blocks([self.quads,changed],[1000,1000]),[])


if __name__=='__main__':unittest.main()
