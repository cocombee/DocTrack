import unittest
import numpy as np
from general_page_mesh import solve_general
from shared_material_mesh import shared_material_field,barcode_material_owners,fit_material_constraints
from material_pixel_footprints import capture,audit
from complete_stroke_guard import audit_strokes


IDENTITY=dict(rows=2,columns=2,grid=[[0,0],[1,0],[0,1],[1,1]])


class SharedMaterialMeshTests(unittest.TestCase):
    def test_owner_specific_halo_merges_only_nearby_guarded_bodies(self):
        owners=capture(IDENTITY,[np.array([[.26,.5]]),np.array([[.385,.5]])],[1025,1025])
        _,plain=fit_material_constraints(IDENTITY,owners,[1025,1025],(17,17))
        self.assertEqual(plain['material_clusters'],2)
        owners[0]['guard_halo_cells']=1
        _,guarded=fit_material_constraints(IDENTITY,owners,[1025,1025],(17,17))
        self.assertEqual(guarded['material_clusters'],1)
        self.assertEqual(guarded['original_owners'],2)

    def test_material_only_field_has_zero_rule_residual(self):
        x=np.linspace(.1,.9,65)
        bent=np.c_[x,.4+.003*np.sin(np.pi*x)]
        field,_=solve_general([(bent,2.)],[])
        seed=dict(rows=33,columns=33,grid=field.reshape(-1,2).tolist())
        pixels=np.c_[np.arange(20)/200,np.arange(20)/300]
        owners=capture(IDENTITY,[pixels+[.25,.35],pixels+[.65,.55]],[201,301])
        ledger=dict(retained_pixel_footprints=owners,complete_printed_strokes=[])
        result,report=shared_material_field(seed,ledger,[201,301],shape=(257,257),iterations=20)
        self.assertTrue(report['accepted'],report)
        self.assertEqual(report['maximum_ruling_normal_pin_error_px'],0)
        mesh=dict(rows=257,columns=257,grid=result.reshape(-1,2).tolist())
        self.assertTrue(all(q['accepted'] for q in audit(mesh,owners)))

    def test_general_surface_preserves_complete_bodies_and_straightens_rule(self):
        x=np.linspace(.1,.9,80)
        rules=[np.c_[x,y+.006*np.sin(np.pi*x)] for y in (.2,.7)]
        field,_=solve_general([(q,4.) for q in rules],[])
        seed=dict(rows=33,columns=33,grid=field.reshape(-1,2).tolist())
        a,b=np.meshgrid(np.arange(8),np.arange(9))
        body=np.c_[a.reshape(-1)/200,b.reshape(-1)/300]
        owners=capture(IDENTITY,[body+[.3,.4],body+[.6,.42]],[201,301])
        ledger=dict(retained_pixel_footprints=owners,
            complete_printed_strokes=[dict(axis=0,source_points=q.tolist(),reference_points=q.tolist()) for q in rules])
        result,report=shared_material_field(seed,ledger,[201,301],shape=(257,257),iterations=20)
        self.assertTrue(report['accepted'],report)
        mesh=dict(rows=257,columns=257,grid=result.reshape(-1,2).tolist())
        self.assertTrue(all(q['accepted'] for q in audit(mesh,owners)))
        self.assertTrue(all(q['accepted'] for q in audit_strokes(IDENTITY,mesh,ledger['complete_printed_strokes'],[201,301])))
        self.assertEqual(report['local_corrections'],0)

    def test_vertical_protection_requires_actual_original_material(self):
        x=np.linspace(.1,.9,40);line=np.c_[x,np.full_like(x,.2)]
        grid,_=solve_general([(line,4.)],[])
        seed=dict(rows=33,columns=33,grid=grid.reshape(-1,2).tolist())
        ledger=dict(retained_pixel_footprints=capture(IDENTITY,[np.array([[.4,.5],[.41,.5]])],[201,301]),
            complete_printed_strokes=[dict(axis=1,source_points=[[.2,.3],[.2,.4],[.2,.5]])])
        with self.assertRaisesRegex(ValueError,'lacks complete material ownership'):
            shared_material_field(seed,ledger,[201,301],shape=(33,33),protected_vertical=True)

    def test_bar_registration_keeps_actual_bar_pixels_and_all_trace_members(self):
        gray=np.full((300,400),240,np.uint8);safe=np.full_like(gray,255)
        strokes=[]
        for x in range(100,240,15):
            gray[70:161,x-2:x+3]=20
            p=np.c_[np.full(91,x),np.arange(70,161)]/[399,299]
            strokes.append(dict(axis=1,source_points=p.tolist()))
        ledger=dict(complete_printed_strokes=strokes,retained_pixel_footprints=[])
        owners,report=barcode_material_owners(gray,safe,ledger)
        self.assertTrue(report['accepted'],report)
        self.assertEqual(len(owners),len(strokes))
        self.assertEqual(report['complete_vertical_traces'],len(strokes))
        self.assertGreater(report['original_bar_pixels'],len(strokes)*100)
        self.assertTrue(all(q['accepted'] for q in audit(IDENTITY,owners)))


if __name__=='__main__':unittest.main()
