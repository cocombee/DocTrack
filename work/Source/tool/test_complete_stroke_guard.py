import unittest

import cv2
import numpy as np

from complete_stroke_guard import audit_strokes, bend_change, capture_strokes, current_stroke_guides
from composite_mesh_sampling import transpose_mesh


class CompleteStrokeGuardTests(unittest.TestCase):
    def setUp(self):
        self.base = dict(rows=2, columns=2, grid=[[0,0],[1,0],[0,1],[1,1]])
        self.x = np.linspace(0, 1, 401)

    def test_rigid_rotation_and_translation_preserve_all_scales(self):
        points = np.c_[self.x*400, 20*np.sin(self.x*np.pi)]
        theta=.2
        rotation=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
        self.assertTrue(bend_change(points, points@rotation+[20,10])['accepted'])

    def test_straightening_a_complete_smooth_curve_passes(self):
        points = np.c_[self.x*400, 20*np.sin(self.x*np.pi)]
        after = points.copy();after[:,1]=0
        self.assertTrue(bend_change(points, after)['accepted'])

    def test_lower_overall_error_does_not_hide_a_new_local_kink(self):
        points = np.c_[self.x*400, 20*np.sin(self.x*np.pi)]
        after = points.copy();after[:,1]=((self.x>.4)&(self.x<.6)).astype(float)
        check=bend_change(points,after)
        self.assertLess(check['after_line_error_px'],check['before_line_error_px'])
        self.assertGreater(check['added_local_bend_px'],.1)
        self.assertFalse(check['accepted'])

    def test_connected_trace_capture_keeps_every_original_material_sample(self):
        gray=np.full((400,400),240,np.uint8);safe=np.full_like(gray,255);words=np.zeros_like(gray)
        cv2.line(gray,(40,100),(340,100),80,2)
        traces=[np.c_[np.arange(a,b),np.full(b-a,100)]/399 for a,b in ((40,150),(162,341))]
        owners=capture_strokes(self.base,traces,[],gray,safe,words)
        self.assertEqual(len(owners),1)
        reference=np.asarray(owners[0]['reference_points'])
        for trace in traces:
            self.assertTrue(all(any(np.array_equal(p,q) for q in reference) for p in trace))
        np.testing.assert_allclose(owners[0]['source_points'],reference,atol=1e-15,rtol=0)

    def test_missing_print_does_not_create_a_complete_bridge(self):
        gray=np.full((400,400),240,np.uint8);safe=np.full_like(gray,255);words=np.zeros_like(gray)
        traces=[np.c_[np.arange(a,b),np.full(b-a,100)]/399 for a,b in ((40,150),(162,341))]
        owners=capture_strokes(self.base,traces,[],gray,safe,words)
        self.assertEqual(len(owners),2)
        self.assertEqual(sum(len(o['reference_points']) for o in owners),sum(map(len,traces)))

    def test_unordered_cloud_cannot_be_captured_as_a_polyline(self):
        gray=np.full((400,400),240,np.uint8);safe=np.full_like(gray,255);words=np.zeros_like(gray)
        p=np.c_[np.linspace(.1,.9,40),np.full(40,.25)]
        p=p[np.random.default_rng(0).permutation(len(p))]
        with self.assertRaisesRegex(ValueError,'ordered observed trace'):
            capture_strokes(self.base,[p],[],gray,safe,words)

    def test_exact_material_audit_rejects_new_curve_and_preserves_its_registry(self):
        source=np.c_[np.linspace(.1,.9,101),np.full(101,.5)]
        owners=[dict(axis=0,source_points=source.tolist(),reference_points=source.tolist())]
        u,v=np.meshgrid(np.linspace(0,1,65),np.linspace(0,1,65))
        grid=np.stack([u,v+.003*np.sin(2*np.pi*u)],-1)
        result=dict(rows=65,columns=65,grid=grid.reshape(-1,2).tolist())
        checks=audit_strokes(self.base,result,owners,[1001,1001])
        self.assertFalse(checks[0]['accepted'])
        np.testing.assert_array_equal(owners[0]['reference_points'],source)
        np.testing.assert_array_equal(owners[0]['source_points'],source)
        swapped=transpose_mesh({**result,'complete_printed_strokes':owners})
        self.assertEqual(swapped['complete_printed_strokes'][0]['axis'],1)
        np.testing.assert_array_equal(swapped['complete_printed_strokes'][0]['source_points'],source[:,::-1])
        transposed_checks=audit_strokes(transpose_mesh(self.base),swapped,
                                        swapped['complete_printed_strokes'],[1001,1001])
        self.assertEqual(checks[0]['accepted'],transposed_checks[0]['accepted'])

    def test_missing_or_changed_material_cannot_qualify(self):
        with self.assertRaises(ValueError):audit_strokes(self.base,self.base,[],[400,400])
        with self.assertRaises(ValueError):bend_change([[0,0],[1,0],[2,0]],[[0,0],[1,0],[1,0]])

    def test_complete_proposal_selects_observed_members_without_changing_owners(self):
        points=np.c_[np.linspace(.1,.9,41),np.full(41,.3)]
        points=np.insert(points,10,points[10]+[0,.001],axis=0)
        owners=[dict(axis=0,source_points=points.tolist(),reference_points=points.tolist())]
        before=[dict(o) for o in owners]
        horizontal,vertical=current_stroke_guides(self.base,owners)
        self.assertEqual(len(horizontal),1);self.assertEqual(vertical,[])
        self.assertEqual(len(horizontal[0]),41)
        self.assertTrue(all(np.min(np.linalg.norm(points-q,axis=1))<1e-14 for q in horizontal[0]))
        self.assertEqual(owners,before)
        swapped=transpose_mesh({**self.base,'complete_printed_strokes':owners})
        th,tv=current_stroke_guides(swapped,swapped['complete_printed_strokes'])
        self.assertEqual(th,[])
        np.testing.assert_allclose(tv[0],horizontal[0],atol=1e-15,rtol=0)


if __name__=='__main__':unittest.main()
