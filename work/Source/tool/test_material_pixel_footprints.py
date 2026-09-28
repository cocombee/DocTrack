import copy
import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np

import material_pixel_footprints as pixels
import progressive_content_mesh as engine
from composite_mesh_sampling import MeshSampler, transpose_mesh
from progressive_content_mesh import retained_word_audit, refine_staged


class MaterialPixelFootprintTests(unittest.TestCase):
    def setUp(self):
        u,v=np.meshgrid(np.linspace(0,1,33),np.linspace(0,1,33))
        self.base=dict(rows=33,columns=33,grid=np.stack([u,v],-1).reshape(-1,2).tolist())
        self.body=np.array([[.25,.3],[.255,.3],[.26,.3],
                            [.25,.32],[.255,.32],[.26,.32]])

    def test_registered_corners_reject_damage_that_all_pixel_centers_miss(self):
        u,v=np.meshgrid(np.linspace(0,1,1601),[0.,1.])
        field=np.stack([u,v+.0006*np.sin(np.pi*u*400)],-1)
        damaged=dict(rows=2,columns=1601,grid=field.reshape(-1,2).tolist())
        centers=retained_word_audit(self.base,damaged,[self.body],[401,301],
                                    references=[self.body])
        self.assertTrue(centers[0]['accepted'])
        owners=pixels.capture(self.base,[self.body],[401,301])
        before=copy.deepcopy(owners)
        checked=pixels.audit(damaged,owners)
        self.assertFalse(checked[0]['accepted'])
        self.assertGreater(checked[0]['rigid_shape_error_px'],.15)
        self.assertEqual(owners,before)

    def test_complete_pair_check_finds_unsampled_opposite_motion(self):
        before=np.c_[np.arange(200,dtype=float),np.zeros(200)]
        after=before.copy();after[1,0]+=.09;after[3,0]-=.09
        checked=pixels.shape_and_pairs(before,after,.1)
        self.assertLess(checked['rigid_shape_error_px'],.1)
        self.assertGreater(checked['pair_distance_change_or_bound_px'],.17)
        self.assertFalse(checked['accepted'])
        self.assertEqual(checked['pair_proof'],'Observed pair violation certificate')

    def test_registration_pairs_real_frame_through_parent_and_transposes_both(self):
        field=np.asarray(self.base['grid']).reshape(33,33,2)
        field[:,:,0]+=.025*field[:,:,0]*field[:,:,1]
        parent={**self.base,'grid':field.reshape(-1,2).tolist()}
        owners=pixels.capture(parent,[self.body],[801,601],kind='unknown')
        source,refs=pixels.paired_points(owners[0])
        np.testing.assert_array_equal(source,MeshSampler(parent).sample(refs))
        self.assertTrue(pixels.audit(parent,owners)[0]['accepted'])
        registered={**parent,'retained_pixel_footprints':owners}
        saved=copy.deepcopy(registered)
        swapped=transpose_mesh(registered)
        np.testing.assert_array_equal(pixels.paired_points(swapped['retained_pixel_footprints'][0])[0],
                                      source[:,::-1])
        self.assertEqual(swapped['retained_pixel_footprints'][0]['pixel_size'],[601.,801.])
        self.assertTrue(pixels.audit(swapped,swapped['retained_pixel_footprints'])[0]['accepted'])
        self.assertEqual(transpose_mesh(swapped),registered)
        self.assertEqual(registered,saved)

    def test_lossless_storage_keeps_all_bits_and_rejects_changed_payload(self):
        legacy=pixels.capture(self.base,[self.body],[401,301],packed=False)[0]
        encoded=pixels.packed_owner(legacy)
        for before,after in zip(pixels.paired_points(legacy),pixels.paired_points(encoded)):
            np.testing.assert_array_equal(before.view('u8'),after.view('u8'))
        self.assertEqual(pixels.audit(self.base,[legacy]),pixels.audit(self.base,[encoded]))
        changed={**encoded,'points_sha256':'0'*64}
        with self.assertRaisesRegex(ValueError,'payload changed'):
            pixels.audit(self.base,[changed])

    def test_guard_cannot_register_a_late_damaged_reference_or_raise_unknown_limit(self):
        late={**self.base,'retained_word_clouds':[self.body.tolist()],
              'retained_word_references':[self.body.tolist()]}
        with self.assertRaisesRegex(ValueError,'before the first content correction'):
            refine_staged('unused','unused',late,coordinate_mode=True,partition_words=True,
                measured_ink_bands=True,unrecognized_print=True,complete_strokes=True,
                full_pixel_footprints=True)
        owners=pixels.capture(self.base,[self.body],[401,301],kind='unknown')
        self.assertEqual(pixels.audit(self.base,owners)[0]['limit_px'],.03)
        with self.assertRaises(ValueError):
            pixels.capture(self.base,[self.body],[401,301],kind='unknown_looser')

    def test_staged_correction_rolls_back_center_only_success_before_publication(self):
        u,v=np.meshgrid(np.linspace(0,1,1601),[0.,1.])
        candidate=dict(rows=2,columns=1601,
            grid=np.stack([u,v+.0006*np.sin(np.pi*u*400)],-1).reshape(-1,2).tolist())
        groups=[self.body+[.05*i,0] for i in range(10)]
        gray=np.full((301,401),230,np.uint8);safe=np.full_like(gray,255)
        h=[np.array([[.05,y],[.95,y]]) for y in (.1,.5,.9)]
        base={**self.base,'complete_printed_strokes':[dict(axis=0,
              source_points=[[.1,.1],[.5,.1],[.9,.1]],
              reference_points=[[.1,.1],[.5,.1],[.9,.1]])]}
        events=[]
        with tempfile.TemporaryDirectory() as folder:
            output=Path(folder)/'mesh.json'
            with patch.object(engine.rows,'current_measurement',return_value=(gray,safe,gray,
                         np.asarray(self.base['grid']).reshape(33,33,2))), \
                    patch.object(engine,'sections',return_value=(h,h)), \
                    patch.object(engine,'ink_observations',return_value=[]), \
                    patch.object(engine,'partition_shared_ink',return_value=(groups,{'roles':[]})), \
                    patch.object(engine,'partition_rule_fragments',return_value=(groups,[],[])), \
                    patch.object(engine,'column_constraints',return_value=([],{},[])), \
                    patch.object(engine,'coordinate_word_step',return_value=candidate), \
                    patch.object(engine,'coordinate_horizontal_step',return_value=base), \
                    patch.object(engine,'coordinate_vertical_step',return_value=base), \
                    patch('complete_stroke_guard.audit_strokes',return_value=[{'accepted':True}]):
                result=refine_staged('input',output,base,word_provider=lambda *_:[],max_steps=1,
                    coordinate_mode=True,partition_words=True,measured_ink_bands=True,
                    unrecognized_print=True,complete_strokes=True,full_pixel_footprints=True,
                    attempt_observer=events.append)
            self.assertIs(result,base)
            self.assertFalse(events[0]['accepted'])
            self.assertTrue(any(q.get('rejected')=='complete original pixel footprint'
                                for q in events[0]['checks']))
            self.assertFalse(output.exists())


if __name__=='__main__':
    unittest.main()
