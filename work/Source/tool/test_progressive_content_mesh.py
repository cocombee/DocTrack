import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import cv2
import progressive_content_mesh as engine
from local_mesh_storage import publish
from composite_mesh_sampling import MeshSampler


class ProgressiveContentTests(unittest.TestCase):
    def setUp(self):
        u,v=np.meshgrid(np.linspace(0,1,33),np.linspace(0,1,33))
        self.grid=np.stack([u,v],-1)
        self.base={'rows':33,'columns':33,'grid':self.grid.reshape(-1,2).tolist(),
                   'method':'Accepted base'}
        self.gray=np.full((100,100),200,np.uint8)
        self.horizontal=[np.array([[.05,y],[.95,y]]) for y in [.2,.4,.6]]
        self.vertical=[np.array([[.05,x],[.95,x]]) for x in [.2,.4,.6]]

    def test_established_words_keep_their_current_owners_through_a_nonlinear_parent(self):
        u,v=np.meshgrid(np.linspace(0,1,33),np.linspace(0,1,33))
        field=np.stack([u+.025*u*v,v],-1)
        base={**self.base,'grid':field.reshape(-1,2).tolist()}
        sampler=MeshSampler(base)
        def ink(x,y):
            xx,yy=np.meshgrid(np.linspace(x,x+.05,8),np.linspace(y,y+.02,5))
            return np.c_[xx.ravel(),yy.ravel()]
        first,second,below=ink(.2,.3),ink(.3,.3),ink(.2,.35)
        base['established_text_rows']=[{'points':sampler.sample(np.vstack([first,second])).tolist()}]
        self.assertEqual(engine.fixed_text_word_owners(base,MeshSampler(base),
                          [first,second,below],[800,600]),[0,1])
        transposed=engine.transpose_mesh(base)
        self.assertEqual(engine.fixed_text_word_owners(transposed,MeshSampler(transposed),
                          [q[:,::-1] for q in [first,second,below]],[600,800]),[0,1])

    def test_unrelated_word_is_not_pinned_by_an_established_body(self):
        body=np.array([[.2,.3],[.3,.3],[.3,.32],[.2,.32]])
        base={**self.base,'established_text_rows':[{'points':body.tolist()}]}
        self.assertEqual(engine.fixed_text_word_owners(base,MeshSampler(base),
                          [body,body+[.4,0],body+[0,.05]],[800,600]),[0])

    def test_original_image_preparation_declines_non_grid_without_word_or_extra_warp(self):
        with tempfile.TemporaryDirectory() as folder:
            output=Path(folder)/'mesh.json'
            with patch.object(engine.rows.accepted,'solve',return_value=self.base),patch.object(engine.rows,'current_measurement',return_value=(self.gray,self.gray,self.gray,self.grid)),patch.object(engine,'sections',return_value=([],[])),patch.object(engine.rows,'refine') as prepare,patch.object(engine,'refine_staged') as refine:
                result=engine.solve_coordinate_staged('input',output)
            self.assertIs(result,self.base)
            prepare.assert_not_called();refine.assert_not_called()
            self.assertEqual(json.loads(output.read_text()),self.base)

    def test_failed_preparation_keeps_the_last_published_original_image_map(self):
        prepared={**self.base,'method':'Completed supported row'}
        with tempfile.TemporaryDirectory() as folder:
            output=Path(folder)/'mesh.json'
            with patch.object(engine.rows.accepted,'solve',return_value=self.base),patch.object(engine.rows,'current_measurement',return_value=(self.gray,self.gray,self.gray,self.grid)),patch.object(engine,'sections',return_value=(self.horizontal,self.vertical)),patch.object(engine.rows,'refine',return_value=prepared),patch.object(engine.frame,'refine',side_effect=ValueError('Rejected next phase')),patch.object(engine,'refine_staged',side_effect=lambda path,out,base,**kwargs:base):
                result=engine.solve_coordinate_staged('input',output)
            self.assertIs(result,prepared)
            self.assertEqual(json.loads(output.read_text()),prepared)

    def test_original_plane_captures_print_before_any_content_preparation(self):
        with tempfile.TemporaryDirectory() as folder:
            output=Path(folder)/'mesh.json'
            with patch.object(engine.rows.accepted,'solve') as broad,patch.object(engine.rows,'current_measurement',return_value=(self.gray,self.gray,self.gray,self.grid)),patch.object(engine,'sections',return_value=(self.horizontal,self.vertical)),patch.object(engine.rows,'refine') as prepare,patch.object(engine.frame,'refine') as frame,patch.object(engine,'refine_staged',side_effect=lambda path,out,base,**kwargs:base) as local:
                result=engine.solve_coordinate_staged('input',output,original_plane=True,
                    measured_ink_bands=True,unrecognized_print=True)
            broad.assert_not_called();prepare.assert_not_called();frame.assert_not_called()
            self.assertEqual(result['grid'],[[0.,0.],[1.,0.],[0.,1.],[1.,1.]])
            self.assertNotIn('preserved_mesh',result)
            self.assertTrue(local.call_args.kwargs['partition_words'])
            self.assertTrue(local.call_args.kwargs['unrecognized_print'])
            self.assertEqual(json.loads(output.read_text()),result)

    def test_original_plane_keeps_non_grid_fallback_and_skips_native_requests(self):
        with tempfile.TemporaryDirectory() as folder:
            output=Path(folder)/'mesh.json'
            with patch.object(engine.rows.accepted,'solve',return_value=self.base) as broad,patch.object(engine.rows,'current_measurement',return_value=(self.gray,self.gray,self.gray,self.grid)),patch.object(engine,'sections',return_value=([],[])),patch.object(engine.rows,'refine') as prepare,patch.object(engine,'refine_staged') as local:
                result=engine.solve_coordinate_staged('input',output,original_plane=True,
                    measured_ink_bands=True,unrecognized_print=True)
            broad.assert_called_once_with('input',None)
            prepare.assert_not_called();local.assert_not_called()
            self.assertIs(result,self.base)
            self.assertEqual(json.loads(output.read_text()),self.base)

    def test_original_plane_cannot_run_without_missed_and_measured_ink_protection(self):
        for options in [{},{'measured_ink_bands':True},{'unrecognized_print':True}]:
            with patch.object(engine.rows.accepted,'solve') as broad:
                with self.assertRaisesRegex(ValueError,'complete measured print protection'):
                    engine.solve_coordinate_staged('input','unused',original_plane=True,**options)
            broad.assert_not_called()

    def test_post_warp_reference_can_hide_damage_that_original_material_rejects(self):
        x,y=np.meshgrid(np.linspace(.2,.35,13),np.linspace(.3,.32,5))
        original=np.c_[x.ravel(),y.ravel()]
        bent=self.grid.copy();bent[:,:,1]+=.8*(bent[:,:,0]-.275)**2
        damaged={**self.base,'grid':bent.reshape(-1,2).tolist()}
        from composite_mesh_inverse import inverse_retained
        late=inverse_retained(MeshSampler(damaged),original)
        late_check=engine.retained_word_audit(damaged,damaged,[original],[800,600],references=[late])
        original_check=engine.retained_word_audit(self.base,damaged,[original],[800,600],references=[original])
        self.assertTrue(late_check[0]['accepted'])
        self.assertFalse(original_check[0]['accepted'])
        self.assertGreater(original_check[0]['rigid_shape_error_px'],1.)

    def test_measured_crossing_reveals_the_vertical_stroke_without_crossing_word_ink(self):
        gray=np.full((400,800),235,np.uint8);safe=np.full_like(gray,255);words=np.zeros_like(gray)
        cv2.line(gray,(40,50),(40,350),80,1)
        cv2.line(gray,(40,200),(760,200),80,1)
        horizontal=[np.c_[np.arange(85,761)/799,np.full(676,200/399)]]
        vertical=[np.c_[np.arange(50,198)/399,np.full(148,40/799)]]
        hs,vs=engine.refill_crossings(gray,safe,horizontal,vertical,words)
        self.assertAlmostEqual(hs[0][0,0],40/799)
        self.assertTrue(any(g[0,0]<.2 and g[-1,0]>.8 and
                            abs(g[:,1].mean()-40/799)<.003 for g in vs))
        words[193:208,60:70]=1
        hs,vs=engine.refill_crossings(gray,safe,horizontal,vertical,words)
        self.assertIs(hs,horizontal)
        self.assertIs(vs,vertical)

    def test_receipt_decline_does_not_request_native_evidence(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(engine.rows,'current_measurement',return_value=(self.gray,self.gray,self.gray,self.grid)),patch.object(engine,'sections',return_value=([],[])):
                def unexpected(*args):self.fail('No word request without a printed grid')
                result=engine.refine_staged('input.png',Path(folder)/'mesh.json',self.base,word_provider=unexpected)
        self.assertIs(result,self.base)

    def test_non_grid_paper_requires_measured_print_protection(self):
        with self.assertRaisesRegex(ValueError, 'complete measured original print'):
            engine.refine_staged('unused', 'unused', self.base, non_grid_paper=True)

    def test_non_grid_paper_requires_real_glyph_rows_and_repeated_rules(self):
        gray = np.full((640, 800), 240, np.uint8)
        words = []
        for bottom in (120, 260, 400):
            for start in (160, 240, 320):
                quad = np.array([[start-2, bottom-18], [start+48, bottom-18],
                                 [start+48, bottom+2], [start-2, bottom+2]])
                words.append(dict(text='AB', confidence=.99,
                                  points=(quad / [800, 640]).tolist()))
                for x in (start+4, start+30):
                    gray[bottom-13:bottom+1, x:x+7] = 45
        safe = np.full_like(gray, 255)
        self.assertTrue(engine.supported_paper_rows(gray, safe, words, self.horizontal))
        self.assertFalse(engine.supported_paper_rows(np.full_like(gray,240), safe, words, self.horizontal))
        self.assertFalse(engine.supported_paper_rows(gray, np.zeros_like(safe), words, self.horizontal))
        self.assertFalse(engine.supported_paper_rows(gray, safe, words, self.horizontal[:2]))
        self.assertFalse(engine.supported_paper_rows(gray, safe, words[:6], self.horizontal))

    def test_unsupported_non_grid_paper_keeps_the_original_map(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(engine.rows,'current_measurement',return_value=(self.gray,self.gray,self.gray,self.grid)),patch.object(engine,'sections',return_value=([],[])):
                result=engine.refine_staged('input.png',Path(folder)/'mesh.json',self.base,
                    word_provider=lambda *_: [], non_grid_paper=True,
                    coordinate_mode=True,partition_words=True,measured_ink_bands=True,
                    unrecognized_print=True,complete_strokes=True)
        self.assertIs(result,self.base)

    def test_earlier_faint_word_edges_keep_their_unique_owner(self):
        original=np.array([[.19,.3],[.3,.3],[.3,.32],[.19,.32]])
        cropped=np.array([[.2,.3],[.3,.3],[.3,.32],[.2,.32]])
        other=np.array([[.5,.3],[.6,.3],[.6,.32],[.5,.32]])
        base={**self.base,'retained_word_clouds':[original.tolist()],
              'retained_word_references':[original.tolist()]}
        groups,clouds,references=engine.carry_word_groups(base,[cropped,other],MeshSampler(base))
        self.assertAlmostEqual(groups[0][:,0].min(),.19)
        np.testing.assert_array_equal(groups[1],other)
        np.testing.assert_array_equal(clouds[0],original)
        np.testing.assert_array_equal(references[0],original)

    def test_missing_original_reference_cannot_authorize_word_carry(self):
        group=np.array([[.2,.3],[.3,.3]])
        base={**self.base,'retained_word_clouds':[group.tolist()]}
        with self.assertRaisesRegex(ValueError,'original paired geometry'):
            engine.carry_word_groups(base,[group],MeshSampler(base))

    def test_word_missed_by_later_ocr_remains_an_active_owner(self):
        original=np.array([[.2,.3],[.3,.3],[.3,.32],[.2,.32]])
        new_word=original+[.4,.2]
        base={**self.base,'retained_word_clouds':[original.tolist()],
              'retained_word_references':[original.tolist()]}
        groups,clouds,references=engine.carry_word_groups(base,[new_word],MeshSampler(base))
        self.assertEqual(len(groups),2)
        np.testing.assert_allclose(groups[1],original,atol=1e-12)
        self.assertEqual(len(clouds),2)
        retained={**base,'retained_word_clouds':[g.tolist() for g in clouds],
                  'retained_word_references':[g.tolist() for g in references]}
        again,clouds,references=engine.carry_word_groups(retained,[original],MeshSampler(retained))
        self.assertEqual(len(again),2)
        np.testing.assert_allclose(again[1],new_word,atol=1e-12)
        self.assertEqual(len(clouds),2)

    def test_ambiguous_historical_word_cannot_lose_its_active_constraint(self):
        prior=np.array([[.2,.3],[.5,.3],[.5,.32],[.2,.32]])
        left=np.array([[.2,.3],[.3,.3],[.3,.32],[.2,.32]])
        right=left+[.2,0]
        base={**self.base,'retained_word_clouds':[prior.tolist()],
              'retained_word_references':[prior.tolist()]}
        groups,_,_=engine.carry_word_groups(base,[left,right],MeshSampler(base))
        self.assertEqual(len(groups),3)
        np.testing.assert_array_equal(groups[0],left)
        np.testing.assert_array_equal(groups[1],right)
        np.testing.assert_allclose(groups[2],prior,atol=1e-12)

    def test_word_shape_gate_compares_with_original_instead_of_accumulating_errors(self):
        original=np.array([[.2,.3],[.3,.3],[.3,.32],[.2,.32]])
        first=self.grid.copy();first[:,:,0]*=1.0007
        second=self.grid.copy();second[:,:,0]*=1.0014
        base={**self.base,'grid':first.reshape(-1,2).tolist()}
        result={**self.base,'grid':second.reshape(-1,2).tolist()}
        relative=engine.retained_word_audit(base,result,[original],[1000,1000])
        absolute=engine.retained_word_audit(base,result,[original],[1000,1000],references=[original])
        self.assertTrue(relative[0]['accepted'])
        self.assertFalse(absolute[0]['accepted'])
        self.assertGreater(absolute[0]['pair_distance_change_px'],.1)

    def test_unrecognized_print_guard_is_stricter_and_cannot_loosen_existing_guard(self):
        original=np.array([[.2,.3],[.3,.3],[.3,.32],[.2,.32]])
        altered=self.grid.copy();altered[:,:,0]*=1.0007
        result={**self.base,'grid':altered.reshape(-1,2).tolist()}
        ordinary=engine.retained_word_audit(self.base,result,[original],[1000,1000],references=[original])
        missed=engine.retained_word_audit(self.base,result,[original],[1000,1000],references=[original],limit_px=.03)
        self.assertTrue(ordinary[0]['accepted'])
        self.assertFalse(missed[0]['accepted'])
        for invalid in (0,.10001,np.nan):
            with self.assertRaises(ValueError):
                engine.retained_word_audit(self.base,result,[original],[1000,1000],limit_px=invalid)

    def test_retained_column_rejects_different_rigid_word_translations(self):
        groups=[np.array([[.2,y-.01],[.23,y-.01],[.23,y+.01],[.2,y+.01]])
                for y in (.3,.5,.8)]
        query=self.grid.copy()
        query[:,:,0]-=.005*np.clip((query[:,:,1]-.6)/.1,0,1)
        result={**self.base,'grid':query.reshape(-1,2).tolist(),
                'coordinate_transform':True,'preserved_mesh':self.base,
                'refinement_regions':[[0,0,1,1]]}
        shapes=engine.retained_word_audit(self.base,result,groups,[1000,1000])
        self.assertTrue(all(q['accepted'] for q in shapes))
        points=[[.2,.3],[.2,.5],[.2,.8]]
        record={'normal_axis':0,'points':points,'reference':points}
        columns=engine.column_geometry_audit(result,[record],[1000,1000])
        self.assertFalse(columns[0]['accepted'])
        self.assertAlmostEqual(columns[0]['relative_alignment_change_px'],5)

    def test_column_reference_and_axis_follow_vertical_transpose(self):
        points=[[.2,.3],[.2,.5],[.2,.8]]
        base={**self.base,'established_text_columns':[
            {'normal_axis':0,'points':points,'reference':points}]}
        transposed=engine.transpose_mesh(base)
        record=transposed['established_text_columns'][0]
        self.assertEqual(record['normal_axis'],1)
        np.testing.assert_array_equal(record['reference'],np.array(points)[:,::-1])
        self.assertTrue(engine.column_geometry_audit(transposed,[record],[1000,1000])[0]['accepted'])

    def test_coordinate_stage_rejects_moving_an_established_rule_normal(self):
        rule=np.array([[.2,.3],[.3,.3],[.6,.3],[.7,.3]])
        base={**self.base,'established_rulers':[{'normal_axis':1,'points':rule.tolist()}]}
        grid=self.grid.copy();grid[:,:,1]+=self.grid[:,:,1]*.01
        result={**self.base,'grid':grid.reshape(-1,2).tolist()}
        _,_,_,rules=engine.coordinate_stage_audit(base,result,self.grid,
                      [np.array([[.2,.6],[.3,.6]])],[],[],1000,1000)
        self.assertFalse(rules[0]['accepted'])
        self.assertGreater(rules[0]['normal_position_change_px'],2)

    def test_missing_word_evidence_keeps_the_exact_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            target=Path(folder)/'mesh.json';publish(self.base,target);before=target.read_bytes()
            with patch.object(engine.rows,'current_measurement',return_value=(self.gray,self.gray,self.gray,self.grid)),patch.object(engine,'sections',return_value=(self.horizontal,self.vertical)):
                result=engine.refine_staged('input.png',target,self.base,word_provider=lambda *_:[])
            self.assertEqual(target.read_bytes(),before)
        self.assertIs(result,self.base)

    def test_complete_stroke_guard_rejects_before_any_candidate_is_published(self):
        horizontal=[np.c_[np.linspace(.05,.95,61),np.full(61,y)] for y in (.2,.4,.6)]
        vertical=[p.copy() for p in horizontal]
        candidate=self.grid.copy();candidate[:,:,1]+=.02*np.sin(candidate[:,:,0]*np.pi*2)
        proposed={**self.base,'grid':candidate.reshape(-1,2).tolist(),
                  'coordinate_transform':True,'preserved_mesh':self.base,
                  'refinement_regions':[[0,0,1,1]]}
        events=[]
        with tempfile.TemporaryDirectory() as folder:
            target=Path(folder)/'mesh.json';publish(self.base,target);before=target.read_bytes()
            with patch.object(engine.rows,'current_measurement',return_value=(self.gray,np.full_like(self.gray,255),self.gray,self.grid)), \
                    patch.object(engine,'sections',return_value=(horizontal,vertical)), \
                    patch.object(engine,'ink_groups',return_value=[np.array([[.1,.8]])]*10), \
                    patch.object(engine,'coordinate_word_step',return_value=proposed), \
                    patch.object(engine,'coordinate_horizontal_step',return_value=self.base), \
                    patch.object(engine,'coordinate_vertical_step',return_value=self.base):
                result=engine.refine_staged('input',target,self.base,word_provider=lambda *_:[],
                    max_steps=1,coordinate_mode=True,complete_strokes=True,
                    attempt_observer=events.append)
            self.assertEqual(target.read_bytes(),before)
            self.assertIs(result,self.base)
        self.assertEqual(events[0]['checks'][0]['rejected'],'complete printed stroke bending')
        self.assertFalse(events[0]['accepted'])

    def test_rejected_complete_stroke_proposal_allows_the_next_measured_section(self):
        x=np.linspace(.05,.95,61)
        guides=[np.c_[x,y+amplitude*np.sin(x*2*np.pi)]
                for y,amplitude in ((.2,.02),(.4,.01))]
        bad={**self.base,'method':'Partial stroke kink'}
        good={**self.base,'method':'Complete stroke passes'}
        tested=[]
        def guard(result):
            tested.append(result['method'])
            return result is good
        with patch.object(engine,'continued_thin_sections',return_value=[]), \
                patch.object(engine,'measured_chains',return_value=[]), \
                patch.object(engine,'extend_to_crossings',side_effect=lambda gray,safe,guide,*args,**kwargs:guide), \
                patch.object(engine,'before_stable_tail',return_value=None), \
                patch.object(engine.rows,'refine',side_effect=[bad,good]) as refine:
            result=engine.coordinate_horizontal_step('input',self.base,[],
                (self.gray,np.full_like(self.gray,255),self.gray,self.grid),
                guides,[],[],candidate_guard=guard)
        self.assertIs(result,good)
        self.assertEqual(refine.call_count,2)
        self.assertEqual(tested,['Partial stroke kink','Complete stroke passes'])

    def test_complete_proposal_does_not_split_or_extrapolate_observed_stroke(self):
        x=np.linspace(.05,.95,61);guide=np.c_[x,.2+.02*np.sin(x*2*np.pi)]
        good={**self.base,'method':'Complete observed stroke'}
        with patch.object(engine,'continued_thin_sections') as fragments, \
                patch.object(engine,'measured_chains') as joined, \
                patch.object(engine,'extend_to_crossings') as extend, \
                patch.object(engine,'before_stable_tail') as split, \
                patch.object(engine,'anchored_remainder') as remainder, \
                patch.object(engine.rows,'refine',return_value=good) as refine:
            result=engine.coordinate_horizontal_step('input',self.base,[],
                (self.gray,np.full_like(self.gray,255),self.gray,self.grid),
                self.horizontal,[],[],complete_guides=[guide],candidate_guard=lambda _:True)
        self.assertIs(result,good)
        for helper in (fragments,joined,extend,split,remainder):helper.assert_not_called()
        np.testing.assert_array_equal(refine.call_args.kwargs['guides'][0],guide)
        self.assertTrue(refine.call_args.kwargs['registered_guides'])

    def test_whole_stroke_proposals_cannot_bypass_original_stroke_protection(self):
        with self.assertRaisesRegex(ValueError,'complete original stroke protection'):
            engine.refine_staged('input','unused',self.base,whole_stroke_proposals=True)

    def test_endpoint_targets_use_observed_points_and_retain_both_guard_trials(self):
        x=np.linspace(.05,.95,61);guide=np.c_[x,.2+.025*x+.02*np.sin(x*2*np.pi)]
        unchanged=guide.copy()
        bad={**self.base,'method':'Kinks another stroke'}
        good={**self.base,'method':'Original strokes preserved'}
        tested=[]
        def guard(result):
            tested.append(result['method'])
            return result is good
        with patch.object(engine.rows,'refine',side_effect=[bad,good]) as refine:
            result=engine.coordinate_horizontal_step('input',self.base,[],
                (self.gray,np.full_like(self.gray,255),self.gray,self.grid),
                [],[],[],complete_guides=[guide],complete_endpoint_anchors=True,
                candidate_guard=guard)
        self.assertIs(result,good)
        self.assertEqual(tested,[bad['method'],good['method']])
        self.assertEqual(refine.call_count,2)
        for endpoint,call in zip((0,-1),refine.call_args_list):
            np.testing.assert_array_equal(call.kwargs['guides'][0],unchanged)
            np.testing.assert_array_equal(call.kwargs['target_guide'][:,0],guide[[0,-1],0])
            np.testing.assert_array_equal(call.kwargs['target_guide'][:,1],
                                          [guide[endpoint,1]]*2)
            self.assertTrue(call.kwargs['registered_guides'])
        np.testing.assert_array_equal(guide,unchanged)

    def test_endpoint_targets_cannot_bypass_complete_stroke_registration(self):
        with self.assertRaisesRegex(ValueError,'complete original stroke proposals'):
            engine.refine_staged('unused','unused',self.base,complete_endpoint_anchors=True)
        with self.assertRaisesRegex(ValueError,'complete registered stroke guides'):
            engine.coordinate_horizontal_step('unused',self.base,[],None,[],[],[],
                                              complete_endpoint_anchors=True)

    def test_measured_word_motion_cannot_omit_any_original_print_guard(self):
        flags=dict(coordinate_mode=True,complete_strokes=True,
                   measured_ink_bands=True,unrecognized_print=True)
        for missing in flags:
            options={**flags,missing:False}
            with self.assertRaisesRegex(ValueError,'original material and complete-stroke guards'):
                engine.refine_staged('input','unused',self.base,
                                    measured_motion_owners=True,**options)

    def test_join_rejects_the_kink_introduced_by_a_local_correction(self):
        a=np.c_[np.linspace(.05,.2,30),np.full(30,.02)]
        b=np.c_[np.linspace(.24,.7,30),np.full(30,.02)]
        candidate=self.grid.copy()
        for row,y in enumerate(np.linspace(0,1,33)):
            candidate[row,:,0]-=.01*np.clip((.235-y)/.035,0,1)
        checks=engine.join_audit(self.grid,candidate,[],[a,b],1280,1280)
        self.assertTrue(checks)
        self.assertFalse(all(q['accepted'] for q in checks))
        self.assertGreater(max(q['after_gap_px'] for q in checks),5)

    def test_accepted_stage_is_published_before_a_later_failure(self):
        result={**self.base,'method':'Accepted new step','preserved_mesh':self.base,
                'refinement_regions':[[.1,.1,.5,.5]]}
        calls=0
        def measurement(*_):
            nonlocal calls
            calls+=1
            if calls>1:raise ValueError('Later step failed')
            return self.gray,self.gray,self.gray,self.grid
        observed=[]
        with tempfile.TemporaryDirectory() as folder:
            target=Path(folder)/'mesh.json';publish(self.base,target)
            with patch.object(engine.rows,'current_measurement',side_effect=measurement),patch.object(engine,'sections',return_value=(self.horizontal,self.vertical)),patch.object(engine,'ink_groups',return_value=[np.array([[.1,.1]])]*10),patch.object(engine.normals,'refine',return_value=result):
                retained=engine.refine_staged('input.png',target,self.base,word_provider=lambda *_:[],max_steps=2,attempt_observer=observed.append)
            self.assertEqual(json.loads(target.read_text())['method'],'Accepted new step')
            self.assertIs(retained,result)
            self.assertEqual(retained['preserved_mesh']['method'],'Accepted base')
        self.assertEqual(observed[0]['stage'],'continuation')
        self.assertTrue(observed[0]['accepted'])

    def legacy_missed_print_trial(self, stretch):
        gray=np.full((400,500),230,np.uint8);gray[190:200,250:262]=60
        safe=np.full_like(gray,255)
        native=[np.array([[.08+i*.02,.1],[.09+i*.02,.1],
                          [.09+i*.02,.11],[.08+i*.02,.11]]) for i in range(10)]
        grid=self.grid.copy();grid[:,:,0]*=1.005 if stretch else 1
        proposed={**self.base,'grid':grid.reshape(-1,2).tolist(),
                  'method':'Proposed legacy stage','preserved_mesh':self.base,
                  'refinement_regions':[[0,0,1,1]]}
        captured=[]
        def proposal(*args,**kwargs):
            captured.append(kwargs['word_ink_groups'])
            return proposed
        with tempfile.TemporaryDirectory() as folder:
            target=Path(folder)/'mesh.json';publish(self.base,target)
            with patch.object(engine.rows,'current_measurement',return_value=(gray,safe,gray,self.grid)), \
                    patch.object(engine,'sections',return_value=(self.horizontal,self.vertical)), \
                    patch.object(engine,'ink_groups',return_value=native), \
                    patch.object(engine.normals,'refine',side_effect=proposal), \
                    patch.object(engine.shared,'refine',return_value=self.base), \
                    patch.object(engine,'word_step',return_value=self.base), \
                    patch.object(engine,'horizontal_step',return_value=self.base):
                result=engine.refine_staged('input',target,self.base,word_provider=lambda *_:[],
                                            max_steps=1,unrecognized_print=True)
            stored=json.loads(target.read_text())
        return result,stored,captured

    def test_legacy_missed_print_rejects_stretch_that_current_word_gate_allows(self):
        result,stored,captured=self.legacy_missed_print_trial(True)
        self.assertIs(result,self.base)
        self.assertEqual(stored,self.base)
        self.assertTrue(captured)
        self.assertGreater(len(captured[0]),10)

    def test_legacy_missed_print_keeps_immutable_pairs_in_accepted_checkpoint(self):
        result,stored,captured=self.legacy_missed_print_trial(False)
        self.assertIsNot(result,self.base)
        self.assertEqual(stored['method'],'Proposed legacy stage')
        self.assertEqual(len(stored['unrecognized_print_clouds']),1)
        np.testing.assert_array_equal(stored['unrecognized_print_references'][0],captured[0][-1])
        np.testing.assert_array_equal(stored['unrecognized_print_clouds'][0],
                                      MeshSampler(self.base).sample(captured[0][-1]))
        self.assertGreater(len(stored['retained_word_clouds']),10)
        self.assertTrue(all(q['accepted'] for q in result['progressive_content_audit'][0]['unrecognized_print']))

    def test_native_word_envelopes_stop_rule_tracking(self):
        # Connected small glyphs can evade component-based word classification.
        # Their native envelope must protect both axis measurements regardless.
        group=np.array([[.2,.3],[.4,.35]])
        masks=[]
        def measure(gray,safe,**kwargs):
            masks.append(kwargs['word_mask'].copy())
            return []
        with patch.object(engine,'word_ink_mask',return_value=np.zeros_like(self.gray)),patch.object(engine,'measure_sections',side_effect=measure):
            engine.sections(self.gray,self.gray,[group])
        self.assertTrue(masks[0][29:37,18:42].any())
        self.assertTrue(masks[0][30:35,20:40].all())
        self.assertFalse(masks[0][:20].any())
        np.testing.assert_array_equal(masks[1],masks[0].T)

    def test_precise_tracking_mask_keeps_faint_ink_without_filling_empty_stock(self):
        word=np.array([[.2,.3],[.4,.35]])
        with patch.object(engine,'word_ink_mask',return_value=np.zeros_like(self.gray)):
            mask=engine.supported_word_mask(self.gray,[word],precise_ink=True)
            envelope=engine.supported_word_mask(self.gray,[word])
        pixels=np.rint(word*99).astype(int)
        for x,y in pixels:self.assertTrue(mask[y-2:y+3,x-2:x+3].all())
        self.assertEqual(mask[32,30],0)
        self.assertEqual(envelope[32,30],1)
        np.testing.assert_array_equal(word,np.array([[.2,.3],[.4,.35]]))

    def test_precise_crossing_stops_at_actual_letter_ink(self):
        gray=np.full((400,800),235,np.uint8);safe=np.full_like(gray,255)
        cv2.line(gray,(40,50),(40,350),80,1)
        cv2.line(gray,(40,200),(760,200),80,1)
        horizontal=[np.c_[np.arange(85,761)/799,np.full(676,200/399)]]
        vertical=[np.c_[np.arange(50,198)/399,np.full(148,40/799)]]
        yy,xx=np.meshgrid(np.arange(196,205),np.arange(62,68))
        group=np.c_[xx.ravel()/799,yy.ravel()/399]
        with patch.object(engine,'word_ink_mask',return_value=np.zeros_like(gray)):
            words=engine.supported_word_mask(gray,[group],precise_ink=True)
        hs,vs=engine.refill_crossings(gray,safe,horizontal,vertical,words,[group],precise_ink=True)
        self.assertIs(hs,horizontal)
        self.assertIs(vs,vertical)

    def test_word_rotation_guards_complete_barcodes_and_circular_shapes(self):
        bar=np.array([[.2,.1],[.2,.3]])
        envelope=np.array([[.1,.1],[.3,.3]])
        circle=np.array([[.6,.6],[.7,.7]])
        with patch.object(engine.rows.p.reference,'barcode_rulers',return_value=[bar]),patch.object(engine.rows.p.reference,'measured_envelopes',return_value=[envelope]) as envelopes,patch.object(engine.rows.accepted.accepted,'circular_landmarks',return_value=[circle]):
            rules=engine.word_rulers(self.base,self.grid,[],[],self.gray)
        self.assertEqual(len(rules),3)
        for rule,points in zip(rules,[bar,envelope,circle]):
            self.assertIsNone(rule['normal_axis'])
            np.testing.assert_array_equal(rule['points'],points)
        np.testing.assert_array_equal(envelopes.call_args.args[1][0],bar)

    def test_a_working_text_row_cannot_be_unaligned_by_separate_word_translations(self):
        points=np.array([[.2,.3],[.3,.3],[.6,.3],[.7,.3]])
        base={**self.base,'established_text_rows':[{'points':points.tolist()}]}
        candidate=self.grid.copy()
        candidate[:,:,1]+=np.clip((candidate[:,:,0]-.4)/.1,0,1)*.01
        checks=engine.established_text_audit(base,self.grid,candidate,[1000,1000])
        self.assertFalse(checks[0]['accepted'])
        self.assertGreater(checks[0]['position_change_px'],9)

    def test_completed_word_step_retains_source_ink_for_future_row_guards(self):
        groups=[np.array([[.2,.3],[.3,.3]]),np.array([[.6,.3],[.7,.3]])]
        evidence={'independent_word_owners':[0,1],'region':[.1,.2,.8,.4]}
        result=engine.word_result(self.base,self.grid,self.grid,evidence,groups,'Rigid row')
        np.testing.assert_allclose(result['established_text_rows'][0]['points'],
                                   np.vstack(groups),atol=1e-15,rtol=0)
        rules=engine.word_rulers(result,self.grid,[],[],None)
        self.assertEqual(len(rules),1)
        self.assertIsNone(rules[0]['normal_axis'])


if __name__=='__main__':unittest.main()
