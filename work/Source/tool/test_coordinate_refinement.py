import unittest
from unittest.mock import patch
import numpy as np
import cv2
import section_row_candidate as engine
from composite_mesh_sampling import MeshSampler
from composite_mesh_inverse import inverse_near
from composite_mesh_inverse import inverse_retained
import progressive_content_mesh as progressive
from precise_mesh_inverse import inverse


class CoordinateRefinementTests(unittest.TestCase):
    def test_registered_guides_require_explicit_retained_coordinate_traces(self):
        for options in ({}, {'guides': [np.array([[.2,.2],[.3,.2]])]},
                        {'coordinate_composition': True}):
            with self.assertRaisesRegex(ValueError,'explicit retained coordinate traces'):
                engine.refine('unused',{},registered_guides=True,**options)

    def test_registered_rule_fit_removes_resampling_wave_without_moving_other_ink(self):
        h,w=160,320;gray=np.full((h,w),235,np.uint8)
        x=np.linspace(.05,.45,121)+.00011*np.sin(np.arange(121))
        y=.2+.0015*np.sin(x*15)+.00022*np.sin(5*np.arange(121))
        points=np.c_[x,y]
        cv2.polylines(gray,[np.rint(points*[w-1,h-1]).astype(np.int32)],False,70,1)
        for x0 in range(150,180,6):gray[93:104,x0:x0+3]=70
        yy,xx=np.where(gray[93:104,150:180]<100)
        groups=[np.c_[(xx+150)/(w-1),(yy+93)/(h-1)]]
        u,v=np.meshgrid(np.arange(33)/32,np.arange(33)/32)
        grid=np.stack([u,v],axis=2)
        base={'rows':33,'columns':33,'grid':grid.reshape(-1,2).tolist()}
        results=[]
        with patch.object(engine.p.reference,'extract_rules',return_value=[]), \
                patch.object(engine.p.reference,'barcode_rulers',return_value=[]), \
                patch.object(engine.accepted.accepted,'circular_landmarks',return_value=[]):
            for registered in (False,True):
                results.append(engine.refine('unused',base,guides=[points],
                    word_ink_groups=groups,
                    measurement=(gray,np.full_like(gray,255),gray,grid),
                    coordinate_composition=True,precise_geometry=True,
                    include_wide_ink=True,exact_ordering=True,
                    independent_word_owners=True,measured_ink_bands=True,
                    minimum_resolution=(257,320),preserve_neighbouring_rules=False,
                    registered_guides=registered))
        self.assertIs(results[0],base)
        result=results[1];self.assertIsNot(result,base)
        actual=inverse_retained(MeshSampler(result),points)
        self.assertLess(abs(actual[:,1]-y.mean()).max()*h,.03*max(h,w)/1280)
        retained=inverse_retained(MeshSampler(result),groups[0])
        self.assertLess(abs((retained-groups[0])*[w,h]).max(),.1)
        self.assertTrue(result['continuous_jacobian_audit']['accepted'])

    def test_baseline_supported_word_reaches_solver_when_body_center_proof_declines(self):
        gray=np.full((600,1000),230,np.uint8)
        for i,height in enumerate(range(15,24)):
            x=260+i*30;bottom=310+i
            cv2.rectangle(gray,(x,bottom-height+1),(x+8,bottom),70,-1)
        y,x=np.where(gray<100);groups=[np.c_[x/999,y/599]]
        self.assertIsNone(progressive.word_rows.glyph_body_consensus(
                            gray,groups[0],pixel_size=[999,599]))
        u,v=np.meshgrid(np.arange(17)/16,np.arange(17)/16)
        grid=np.stack([u,v],axis=2)
        base={'rows':17,'columns':17,'grid':grid.reshape(-1,2).tolist()}
        # The real pixel evidence must reach the solver; a solver refusal
        # still preserves the original checkpoint and cannot publish a stage.
        def decline(field,image,owners,index,rulers,**kwargs):
            self.assertIs(image,gray);self.assertIs(owners,groups)
            self.assertEqual(index,0);self.assertTrue(kwargs['body_consensus'])
            self.assertTrue(kwargs['precise_fixed_ink'])
            return field,{'accepted':False,'rejected':'solver refused fixture'}
        with patch.object(progressive,'word_rulers',return_value=[]),patch.object(
                progressive.word_rows,'refine_word',side_effect=decline) as solver:
            result=progressive.coordinate_word_step(base,groups,
                       (gray,np.full_like(gray,255),None,grid),[],[],[])
        self.assertIs(result,base)
        self.assertGreater(solver.call_count,0)

    def test_registered_curve_keeps_material_points_between_uniform_knots(self):
        x=np.linspace(.03,.46,121)+.00008*np.sin(np.arange(121))
        y=.2+.0002*np.sin(5*np.arange(121))
        points=np.c_[x,y];u=np.arange(513)/512
        sampled=np.interp(u,x,y)
        self.assertGreater(abs(np.interp(x,u,sampled)-y).max()*1280,.03)
        fitted=engine.registered_curve_samples(points,513,tolerance=.005/1280)
        self.assertLessEqual(abs(np.interp(x,u,fitted)-y).max()*1280,.005)
        with self.assertRaisesRegex(ValueError,'complete registered trace'):
            engine.registered_curve_samples([[.2,.2],[.2,.3]],513,tolerance=.005/1280)

    def test_shared_row_retargets_only_its_complete_section_and_preserves_fixed_text(self):
        h,w=360,640;gray=np.full((h,w),235,np.uint8)
        left=np.c_[np.arange(32,192)/(w-1),np.full(160,.2)]
        right=np.c_[np.arange(414,609)/(w-1),np.full(195,.215)]
        # Source inversion may put a complete endpoint just below its mesh
        # knot. Its interpolation support must still receive the correction.
        left[-1,0]-=8e-14
        groups=[];quads=[]
        for x0,x1 in ((196,298),(306,406)):
            for x in range(x0,x1,6):gray[72:80,x:x+3]=60
            yy,xx=np.where(gray[72:80,x0:x1]<100)
            groups.append(np.c_[(xx+x0)/(w-1),(yy+72)/(h-1)])
            quads.append(np.array([[x0/(w-1),.197],[x1/(w-1),.197],
                                  [x1/(w-1),.224],[x0/(w-1),.224]]))
        for guide in (left,right):
            cv2.polylines(gray,[np.rint(guide*[w-1,h-1]).astype(np.int32)],False,80,1)
        u,v=np.meshgrid(np.arange(65)/64,np.arange(49)/48)
        grid=np.stack([u+.01*u*v,v+.01*u*(1-v)],axis=2)
        base={'rows':49,'columns':65,'grid':grid.reshape(-1,2).tolist()}
        parent=MeshSampler(base)
        lower=np.c_[np.linspace(.03,.97,121),np.full(121,.5)]
        base['established_rulers']=[{'normal_axis':1,'points':parent.sample(g).tolist()}
                                   for g in (left,right,lower)]
        base['established_text_rows']=[{'points':parent.sample(np.vstack(groups)).tolist()}]
        mask=np.zeros_like(gray)
        for group in groups:
            pixels=np.rint(group*[w-1,h-1]).astype(int);mask[pixels[:,1],pixels[:,0]]=1
        audit=[]
        with patch.object(engine.p.reference,'extract_rules',return_value=[]),patch.object(engine.p.reference,'barcode_rulers',return_value=[]),patch.object(engine.accepted.accepted,'circular_landmarks',return_value=[]):
            result=engine.refine('unused',base,guides=[left,right],word_ink_groups=groups,
                measurement=(gray,np.full_like(gray,255),gray,grid),
                coordinate_composition=True,precise_geometry=True,include_wide_ink=True,
                exact_ordering=True,independent_word_owners=True,measured_ink_bands=True,
                minimum_resolution=(257,321),fixed_word_owners=(0,1),words_mask=mask,
                word_flanked_evidence={'quads':quads,'native_count':2},audit_log=audit)
        self.assertTrue(result is not base,str(audit[-8:]))
        self.assertEqual([q['index'] for q in result['word_flanked_rule_row_audit']['retargeted_rules']],[0])
        candidate=MeshSampler(result)
        for guide in (left,right):
            actual=inverse_retained(candidate,parent.sample(guide))
            self.assertLess(abs(actual[:,1]-.215).max()*h,.03)
        for group in [*groups,lower]:
            actual=inverse_retained(candidate,parent.sample(group))
            self.assertLess(abs((actual-group)*[w,h]).max(),.1)
        checks=progressive.coordinate_stage_audit(base,result,grid,groups,[],[],w,h)
        self.assertTrue(all(q['accepted'] for part in checks for q in part),checks)
        self.assertGreater(checks[3][0]['normal_position_change_px'],5)
        from local_mesh_storage import compact
        stored=compact(result)
        stored_checks=progressive.coordinate_stage_audit(base,stored,grid,groups,[],[],w,h)
        self.assertTrue(all(q['accepted'] for part in stored_checks for q in part),stored_checks)
        # A shorter old registration cannot silently stand in for the full
        # native stroke. Its real tail needs the same fresh raster evidence.
        base['established_rulers'][0]['points']=base['established_rulers'][0]['points'][4:]
        record=result['word_flanked_rule_row_audit']['retargeted_rules'][0]
        record['source_points']=base['established_rulers'][0]['points']
        record['reference_points']=record['reference_points'][4:]
        native_sections=[left[:100],left[104:],right]
        with self.assertRaisesRegex(ValueError,'fresh raster evidence'):
            progressive.coordinate_stage_audit(base,result,grid,groups,native_sections,[],w,h)
        continued=progressive.coordinate_stage_audit(base,result,grid,groups,
            native_sections,[],w,h,measurement=(gray,np.full_like(gray,255)))
        self.assertTrue(all(q['accepted'] for part in continued for q in part),continued)
        physical=next(q for q in continued[2] if q.get('observed_continuation_points'))
        self.assertEqual(physical['observed_continuation_points'],4)
        self.assertEqual(physical['original_registration_points'],len(left)-4)
        self.assertLess(physical['after_gap_px'],.1)
        result['word_flanked_rule_row_audit']['guides'][0][0][1]+=.001
        with self.assertRaisesRegex(ValueError,'observed native stroke'):
            progressive.coordinate_stage_audit(base,result,grid,groups,
                native_sections,[],w,h,measurement=(gray,np.full_like(gray,255)))
        result['word_flanked_rule_row_audit']['guides'][0][0][1]-=.001
        result['word_flanked_rule_row_audit']['retargeted_rules'][0]['reference_points'][0][1]+=.001
        with self.assertRaises(ValueError):
            progressive.coordinate_stage_audit(base,result,grid,groups,[],[],w,h)

    def test_physical_rule_and_column_options_require_their_complete_evidence(self):
        with self.assertRaises(ValueError):
            progressive.refine_staged('unused','unused',{},free_tangent_rules=True)
        with self.assertRaises(ValueError):
            progressive.refine_staged('unused','unused',{},physical_column_origins=True)
        with self.assertRaises(ValueError):
            progressive.refine_staged('unused','unused',{},coordinate_shared_rows=True)

    def test_vertical_step_preserves_unrecognized_text_mask_in_its_reading_direction(self):
        h,w=160,128;gray=np.full((h,w),235,np.uint8)
        words=np.zeros_like(gray);words[61:73,20:100]=1
        u,v=np.meshgrid(np.arange(33)/32,np.arange(33)/32)
        field=np.stack([u,v],axis=2)
        base={'rows':33,'columns':33,'grid':field.reshape(-1,2).tolist()}
        groups=[np.array([[.3,.2],[.4,.2],[.4,.24],[.3,.24]])]
        def horizontal(path,transposed,rotated,measurement,hs,vs,audit,**kw):
            # This mask includes text omitted by the native word groups.
            np.testing.assert_array_equal(kw['words_mask'],words.T)
            self.assertTrue(kw['free_tangent_rules'])
            return transposed
        with patch.object(progressive,'supported_word_mask',return_value=words) as mask,patch.object(progressive,'coordinate_horizontal_step',side_effect=horizontal):
            result=progressive.coordinate_vertical_step('unused',base,groups,
                (gray,np.full_like(gray,255),gray,field),[],[],[],
                measured_ink_bands=True,free_tangent_rules=True)
        self.assertIs(result,base)
        self.assertIs(mask.call_args.args[0],gray)
        self.assertIs(mask.call_args.args[1],groups)

    def test_completed_area_refusal_does_not_repeat_identical_coordinate_geometry(self):
        h,w=160,128;gray=np.full((h,w),235,np.uint8)
        x=np.arange(8,120);y=80+2*np.sin(x/20)
        guide=np.c_[x/(w-1),y/(h-1)]
        cv2.polylines(gray,[np.rint(np.c_[x,y]).astype(np.int32)],False,80,1)
        word=np.array([[.3,.3],[.4,.3],[.4,.34],[.3,.34]])
        u,v=np.meshgrid(np.arange(33)/32,np.arange(33)/32)
        field=np.stack([u,v],axis=2)
        base={'rows':33,'columns':33,'grid':field.reshape(-1,2).tolist()};audit=[]
        with patch.object(engine,'validate_normal_composition',return_value={'accepted':False,'minimum_source_area':.19}) as proof,patch.object(engine.p.reference,'extract_rules',return_value=[]),patch.object(engine.p.reference,'barcode_rulers',return_value=[]),patch.object(engine.accepted.accepted,'circular_landmarks',return_value=[]):
            result=engine.refine('unused',base,guides=[guide],word_ink_groups=[word],
                measurement=(gray,np.full_like(gray,255),gray,field),
                coordinate_composition=True,precise_geometry=True,include_wide_ink=True,
                exact_ordering=True,independent_word_owners=True,audit_log=audit)
        self.assertIs(result,base)
        self.assertEqual(proof.call_count,1)
        self.assertEqual(sum(q.get('rejected')=='identical refused geometry' for q in audit),4)

    def row_example(self):
        h,w=240,400;scale=np.array([w-1,h-1]);angle=.03
        c,s=np.cos(angle),np.sin(angle);rotation=np.array([[c,s],[-s,c]])
        center=np.array([130,95]);groups=[];words=[]
        for x in (50,100,150):
            yy,xx=np.meshgrid(np.arange(90,103),np.arange(x,x+35))
            cloud=np.c_[xx.ravel(),yy.ravel()]
            groups.append(((cloud-center)@rotation+center)/scale)
            quad=np.array([[x-1,89],[x+35,89],[x+35,103],[x-1,103]])
            current=((quad-center)@rotation+center)/scale
            words.append({'text':'Word','confidence':.99,'points':((current*scale+.5)/[w,h]).tolist()})
        u,v=np.meshgrid(np.arange(65)/64,np.arange(49)/48)
        grid=np.stack([u+.02*u*v,v+.01*u*(1-v)],axis=2)
        base={'rows':49,'columns':65,'grid':grid.reshape(-1,2).tolist()}
        gray=np.full((h,w),235,np.uint8)
        return base,groups,words,(gray,np.full_like(gray,255),None,grid),scale

    def test_native_word_row_rotation_preserves_nonlinear_parent_and_every_word_shape(self):
        base,groups,words,measurement,scale=self.row_example();parent=MeshSampler(base);audit=[]
        result=progressive.coordinate_word_row_step(base,groups,words,measurement,[],[],audit,
                                                     minimum_resolution=(241,401))
        self.assertIsNot(result,base,msg=str(audit))
        self.assertTrue(result['continuous_jacobian_audit']['accepted'])
        candidate=MeshSampler(result)
        after=inverse_retained(candidate,parent.sample(np.vstack(groups)))
        offset=0
        for group in groups:
            actual=after[offset:offset+len(group)];offset+=len(group)
            before=group*scale;actual=actual*scale
            a,z=before-before.mean(0),actual-actual.mean(0);u,_,v=np.linalg.svd(a.T@z)
            self.assertLess(np.linalg.norm(a@(u@v)-z,axis=1).max(),.1)
            # The common slanted word-row now has level word centers.
        centers=np.array([part.mean(0) for part in np.split(after,np.cumsum([len(g) for g in groups])[:-1])])*scale
        self.assertLess(np.ptp(centers[:,1]),.1)
        outside=np.array([[.05,.05],[.9,.9]])
        np.testing.assert_array_equal(candidate.sample(outside),parent.sample(outside))

    def test_native_word_row_cannot_rotate_an_already_fixed_word(self):
        base,groups,words,measurement,scale=self.row_example()
        base['established_text_rows']=[{'points':MeshSampler(base).sample(groups[1]).tolist()}]
        result=progressive.coordinate_word_row_step(base,groups,words,measurement,[],[],[],
                                                     minimum_resolution=(241,401))
        self.assertIs(result,base)

    def test_identical_refused_geometry_is_not_rechecked_at_other_strengths(self):
        h,w=160,128;gray=np.full((h,w),235,np.uint8)
        x=np.arange(8,120);y=80+2*np.sin(x/20)
        guide=np.c_[x/(w-1),y/(h-1)]
        cv2.polylines(gray,[np.rint(np.c_[x,y]).astype(np.int32)],False,80,1)
        word=np.array([[.3,.3],[.4,.3],[.4,.34],[.3,.34]])
        u,v=np.meshgrid(np.arange(33)/32,np.arange(33)/32)
        field=np.stack([u,v],axis=2)
        base={'rows':33,'columns':33,'grid':field.reshape(-1,2).tolist()}
        allocation=engine.allocate_stock;calls=[];audit=[]
        def count(*args,**kwargs):
            calls.append(True)
            return allocation(*args,**kwargs)
        with patch.object(engine,'allocate_stock',side_effect=count),patch.object(engine.p.reference,'extract_rules',return_value=[]),patch.object(engine.p.reference,'barcode_rulers',return_value=[]),patch.object(engine.accepted.accepted,'circular_landmarks',return_value=[]):
            result=engine.refine('unused',base,guides=[guide],word_ink_groups=[word],
                measurement=(gray,np.full_like(gray,255),gray,field),
                coordinate_composition=True,precise_geometry=True,include_wide_ink=True,
                exact_ordering=True,independent_word_owners=True,
                extra_protected=[guide],audit_log=audit)
        self.assertIs(result,base)
        self.assertEqual(len(calls),1)
        self.assertEqual(sum(q.get('rejected')=='identical refused geometry' for q in audit),4)
        full=[q for q in audit if 'full_rule_error_px' in q]
        self.assertEqual(len(full),1)
        self.assertGreater(full[0]['other_print_shift_px'],.1)

    def test_measured_pixel_footprints_keep_a_fixed_word_and_a_neighbouring_rule(self):
        h,w=300,256;gray=np.full((h,w),235,np.uint8)
        cv2.putText(gray,'KEEP',(35,120),cv2.FONT_HERSHEY_SIMPLEX,.55,60,1)
        yy,xx=np.where(gray[102:125]<200)
        word=np.c_[xx/(w-1),(yy+102)/(h-1)]
        x=np.arange(10,246);y=145+4*np.sin(x/40)
        guide=np.c_[x/(w-1),y/(h-1)]
        cv2.polylines(gray,[np.rint(np.c_[x,y]).astype(np.int32)],False,90,1)
        fixed=np.c_[np.linspace(.04,.96,101),np.full(101,180/(h-1))]
        u,v=np.meshgrid(np.arange(65)/64,np.arange(97)/96)
        grid=np.stack([u,v+.03*u*(1-v)],axis=2)
        base={'rows':97,'columns':65,'grid':grid.reshape(-1,2).tolist()}
        parent=MeshSampler(base)
        base['established_rulers']=[{'normal_axis':1,'points':parent.sample(fixed).tolist()}]
        with patch.object(engine.p.reference,'extract_rules',return_value=[]),patch.object(engine.p.reference,'barcode_rulers',return_value=[]),patch.object(engine.accepted.accepted,'circular_landmarks',return_value=[]):
            result=engine.refine('unused',base,guides=[guide],word_ink_groups=[word],
                measurement=(gray,np.full_like(gray,255),gray,grid),
                coordinate_composition=True,precise_geometry=True,include_wide_ink=True,
                exact_ordering=True,independent_word_owners=True,measured_ink_bands=True,
                tangent_ink_padding_px=1,fixed_word_owners=[0])
        self.assertIsNot(result,base)
        self.assertTrue(result['continuous_jacobian_audit']['accepted'])
        candidate=MeshSampler(result)
        footprint=engine.pixel_footprint(word,w,h)
        self.assertLess(abs((inverse_retained(candidate,parent.sample(footprint))-footprint)*[w,h]).max(),.01)
        self.assertLess(np.ptp(inverse_retained(candidate,parent.sample(guide))[:,1])*h,.01)
        self.assertLess(abs((inverse_retained(candidate,parent.sample(fixed))-fixed)*[w,h]).max(),.01)

    def test_measured_tangent_halo_can_use_blank_stock_without_moving_adjacent_text(self):
        h,w=300,1000
        gray=np.full((h,w),235,np.uint8)
        cv2.putText(gray,'KEEP',(370,153),cv2.FONT_HERSHEY_SIMPLEX,.6,60,1)
        yy,xx=np.where(gray<200)
        word=np.c_[xx/(w-1),yy/(h-1)]
        x=np.arange(xx.max()+3,900);y=148+4*np.cos((x-x[0])/100)
        guide=np.c_[x/(w-1),y/(h-1)]
        cv2.polylines(gray,[np.rint(np.c_[x,y]).astype(np.int32)],False,90,1)
        u,v=np.meshgrid(np.arange(33)/32,np.arange(33)/32)
        field=np.stack([u,v],axis=2)
        base={'rows':33,'columns':33,'grid':field.reshape(-1,2).tolist()}
        args=dict(guides=[guide],word_ink_groups=[word],
                  measurement=(gray,np.full_like(gray,255),gray,field),
                  minimum_resolution=(301,1000),coordinate_composition=True,
                  precise_geometry=True,include_wide_ink=True,exact_ordering=True,
                  independent_word_owners=True,align_to_existing=True)
        with patch.object(engine.p.reference,'extract_rules',return_value=[]),patch.object(engine.p.reference,'barcode_rulers',return_value=[]),patch.object(engine.accepted.accepted,'circular_landmarks',return_value=[]):
            legacy=engine.refine('unused',base,**args)
            result=engine.refine('unused',base,tangent_ink_padding_px=1,**args)
        self.assertIs(legacy,base)
        self.assertIsNot(result,base)
        parent,candidate=MeshSampler(base),MeshSampler(result)
        actual=inverse_retained(candidate,parent.sample(word))
        line=inverse_retained(candidate,parent.sample(guide))
        self.assertLess(abs((actual-word)*[w,h]).max(),.01)
        self.assertLess(np.ptp(line[:,1])*h,.03)
        self.assertTrue(result['continuous_jacobian_audit']['accepted'])

    def test_a_stock_shortage_can_retry_another_independently_safe_ordering(self):
        h,w=160,128;gray=np.full((h,w),235,np.uint8)
        x=np.arange(8,120);y=80+2*np.sin(x/20)
        guide=np.c_[x/(w-1),y/(h-1)]
        cv2.polylines(gray,[np.rint(np.c_[x,y]).astype(np.int32)],False,80,1)
        word=np.array([[.3,.3],[.4,.3],[.4,.34],[.3,.34]])
        u,v=np.meshgrid(np.arange(33)/32,np.arange(33)/32)
        field=np.stack([u,v],axis=2)
        base={'rows':33,'columns':33,'grid':field.reshape(-1,2).tolist()}
        allocation=engine.allocate_stock;calls=[]
        def retry(*args,**kwargs):
            calls.append(True)
            if len(calls)==1:raise ValueError('Insufficient actual capacity')
            return allocation(*args,**kwargs)
        with patch.object(engine,'allocate_stock',side_effect=retry),patch.object(engine.p.reference,'extract_rules',return_value=[]),patch.object(engine.p.reference,'barcode_rulers',return_value=[]),patch.object(engine.accepted.accepted,'circular_landmarks',return_value=[]):
            result=engine.refine('unused',base,guides=[guide],word_ink_groups=[word],
                    measurement=(gray,np.full_like(gray,255),gray,field),
                    coordinate_composition=True,precise_geometry=True,
                    include_wide_ink=True,exact_ordering=True,independent_word_owners=True)
        self.assertIsNot(result,base)
        self.assertGreater(len(calls),1)
        self.assertTrue(result['continuous_jacobian_audit']['accepted'])
        restored=inverse_retained(MeshSampler(result),MeshSampler(base).sample(word))
        self.assertLess(abs((restored-word)*[w,h]).max(),.01)

    def test_vertical_correction_keeps_complete_word_and_established_spine(self):
        h,w=300,256;gray=np.full((h,w),235,np.uint8)
        cv2.putText(gray,'KEEP',(180,120),cv2.FONT_HERSHEY_SIMPLEX,.55,60,1)
        yy,xx=np.where(gray[102:125]<200)
        word=np.c_[xx/(w-1),(yy+102)/(h-1)]
        y=np.arange(10,290);x=128+4*np.sin(y/300*2*np.pi)
        guide=np.c_[y/(h-1),x/(w-1)]
        cv2.polylines(gray,[np.rint(np.c_[x,y]).astype(np.int32)],False,90,1)
        u,v=np.meshgrid(np.arange(65)/64,np.arange(97)/96)
        field=np.stack([u+.03*v*(1-u),v],axis=2)
        base={'rows':97,'columns':65,'grid':field.reshape(-1,2).tolist()}
        parent=MeshSampler(base)
        fixed=np.c_[np.full(100,160/(w-1)),np.linspace(.04,.96,100)]
        base['established_rulers']=[{'normal_axis':0,'points':parent.sample(fixed).tolist()}]
        with patch.object(engine.p.reference,'extract_rules',return_value=[]),patch.object(engine.p.reference,'barcode_rulers',return_value=[]),patch.object(engine.accepted.accepted,'circular_landmarks',return_value=[]):
            result=progressive.coordinate_vertical_step('unused',base,[word],
                        (gray,np.full_like(gray,255),gray,field),[],[guide],[])
        self.assertIsNot(result,base)
        self.assertIs(result['preserved_mesh'],base)
        self.assertTrue(result['continuous_jacobian_audit']['accepted'])
        candidate=MeshSampler(result)
        actual=inverse_retained(candidate,parent.sample(word))
        self.assertLess(abs((actual-word)*[w,h]).max(),.01)
        line=inverse_retained(candidate,parent.sample(guide[:,::-1]))
        self.assertLess(np.ptp(line[:,0])*w,.01)
        ruler=inverse_retained(candidate,parent.sample(fixed))
        self.assertLess(abs((ruler-fixed)*[w,h]).max(),.01)

    def test_normal_correction_keeps_nonlinear_parent_and_protected_word(self):
        h,w=300,256
        gray=np.full((h,w),235,np.uint8)
        cv2.putText(gray,'KEEP',(35,120),cv2.FONT_HERSHEY_SIMPLEX,.55,60,1)
        yy,xx=np.where(gray[102:125]<200)
        word=np.c_[xx/(w-1),(yy+102)/(h-1)]
        x=np.arange(10,246);y=145+4*np.sin(x/256*2*np.pi)
        guide=np.c_[x/(w-1),y/(h-1)]
        cv2.polylines(gray,[np.rint(np.c_[x,y]).astype(np.int32)],False,90,1)
        fixed=np.c_[np.linspace(.04,.96,101),np.full(101,180/(h-1))]
        u,v=np.meshgrid(np.arange(65)/64,np.arange(97)/96)
        grid=np.stack([u,v+.03*u*(1-v)],axis=2)
        base={'rows':97,'columns':65,'grid':grid.reshape(-1,2).tolist(),'method':'Accepted nonlinear map'}
        parent=MeshSampler(base)
        base['established_rulers']=[{'normal_axis':1,'points':parent.sample(fixed).tolist()}]
        with patch.object(engine.p.reference,'extract_rules',return_value=[]),patch.object(engine.p.reference,'barcode_rulers',return_value=[]),patch.object(engine.accepted.accepted,'circular_landmarks',return_value=[]):
            result=engine.refine('unused.png',base,guides=[guide],word_ink_groups=[word],measurement=(gray,np.full_like(gray,255),gray,grid),minimum_resolution=(1025,256),coordinate_composition=True,precise_geometry=True,include_wide_ink=True,exact_ordering=True,independent_word_owners=True)
        self.assertIsNot(result,base)
        self.assertTrue(result['coordinate_transform'])
        self.assertIs(result['preserved_mesh'],base)
        self.assertTrue(result['continuous_jacobian_audit']['accepted'])
        candidate=MeshSampler(result)
        query=np.asarray(result['grid']).reshape(result['rows'],result['columns'],2)
        after=inverse_near(candidate,parent.sample(word),inverse(query,word))
        self.assertLess(abs((after-word)*[w,h]).max(),.01)
        line=inverse_near(candidate,parent.sample(guide),inverse(query,guide))
        self.assertLess(np.ptp(line[:,1])*h,.1)
        ruler=inverse_near(candidate,parent.sample(fixed),inverse(query,fixed))
        self.assertLess(abs(ruler[:,1]-fixed[:,1]).max()*h,.01)
        points=np.array([[.5,.1],[.5,.8],[.2,.9]])
        np.testing.assert_array_equal(candidate.sample(points),parent.sample(points))

    def test_measured_rule_corners_stay_on_the_sampling_lattice(self):
        h,w=300,256;gray=np.full((h,w),235,np.uint8)
        cv2.putText(gray,'KEEP',(35,120),cv2.FONT_HERSHEY_SIMPLEX,.55,60,1)
        yy,xx=np.where(gray[102:125]<200)
        word=np.c_[xx/(w-1),(yy+102)/(h-1)]
        x=np.arange(10,246);y=np.rint(145+4*np.sin(x/40))
        guide=np.c_[x/(w-1),y/(h-1)]
        cv2.polylines(gray,[np.c_[x,y].astype(np.int32)],False,90,1)
        u,v=np.meshgrid(np.arange(65)/64,np.arange(97)/96)
        field=np.stack([u,v],axis=2);base={'rows':97,'columns':65,'grid':field.reshape(-1,2).tolist()}
        parent=MeshSampler(base)
        with patch.object(engine.p.reference,'extract_rules',return_value=[]),patch.object(engine.p.reference,'barcode_rulers',return_value=[]),patch.object(engine.accepted.accepted,'circular_landmarks',return_value=[]):
            result=engine.refine('unused',base,guides=[guide],word_ink_groups=[word],
                        measurement=(gray,np.full_like(gray,255),gray,field),
                        minimum_resolution=(1025,513),coordinate_composition=True,
                        precise_geometry=True,include_wide_ink=True,exact_ordering=True,
                        independent_word_owners=True)
        self.assertIsNot(result,base)
        self.assertEqual((result['columns']-1)%(w-1),0)
        self.assertEqual((result['rows']-1)%(2*(h-1)),0)
        line=inverse_retained(MeshSampler(result),parent.sample(guide))
        self.assertLess(np.ptp(line[:,1])*h,1e-6)

    def test_legacy_source_grid_refiner_declines_coordinate_layers(self):
        with self.assertRaisesRegex(ValueError,'coordinate-aware'):
            engine.refine('unused.png',{'coordinate_transform':True})


if __name__=='__main__':unittest.main()
