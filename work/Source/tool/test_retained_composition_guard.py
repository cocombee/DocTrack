import unittest
import numpy as np
from composite_mesh_sampling import MeshSampler
from composite_mesh_inverse import differential
from local_mesh_storage import compact
from retained_composition_guard import validate_retained_composition


class RetainedCompositionTests(unittest.TestCase):
    def nested(self):
        u,v=np.meshgrid(np.arange(17)/16,np.arange(25)/24)
        old=np.stack([u+.025*u*v,v+.02*u*(1-v)],axis=2)
        base={'rows':25,'columns':17,'grid':old.reshape(-1,2).tolist()}
        weight=np.clip(1-abs(u-.5)/.25,0,1)*np.clip(1-abs(v-.5)/.25,0,1)
        q=np.stack([u,v+.01*weight],axis=2)
        first={'rows':25,'columns':17,'grid':q.reshape(-1,2).tolist(),
               'coordinate_transform':True,'preserved_mesh':base,
               'refinement_regions':[[.25,.25,.75,.75]]}
        q=np.stack([u-.009*weight,v],axis=2)
        second={**first,'grid':q.reshape(-1,2).tolist(),'preserved_mesh':first}
        return MeshSampler(compact(second,recursive=True)),u,v,weight

    def test_overlapping_maps_have_a_conservative_continuous_bound(self):
        parent,u,v,weight=self.nested()
        query=np.stack([u+.005*weight,v-.004*weight],axis=2)
        report=validate_retained_composition(parent,query,[[.2,.2,.8,.8]])
        self.assertTrue(report['accepted'],report)
        q=MeshSampler({'rows':25,'columns':17,'grid':query.reshape(-1,2).tolist()})
        x,y=np.meshgrid(np.linspace(.201,.799,251),np.linspace(.201,.799,251))
        points=np.c_[x.ravel(),y.ravel()]
        actual=np.linalg.det(differential(parent,q.sample(points))@differential(q,points)).min()
        self.assertLessEqual(report['continuous_lower_bound'],actual+1e-9)
        self.assertGreater(report['continuous_lower_bound'],.7)
        declined=validate_retained_composition(parent,query,[[.2,.2,.8,.8]],minimum=1.2)
        self.assertFalse(declined['accepted'])

    def test_fragment_certificate_avoids_unrelated_whole_cell_area_minimum(self):
        from normal_composition_guard import validate_source_composition
        # The old bilinear cell has area .21 at its left edge, while the
        # requested correction touches only the wider right part. Multiplying
        # whole-cell minima declines a patch whose exact area is above .6.
        u,v=np.meshgrid(np.arange(2),np.arange(2))
        old=np.stack([u,.21*v+.79*u*v],axis=2)
        parent=MeshSampler({'rows':2,'columns':2,
                            'grid':old.reshape(-1,2).tolist()})
        u,v=np.meshgrid(np.arange(33)/32,np.arange(33)/32)
        query=np.stack([u,.8*v],axis=2);regions=[[.72,.3,.9,.6]]
        coarse=validate_source_composition(parent,query,regions)
        precise=validate_retained_composition(parent,query,regions)
        self.assertFalse(coarse['accepted'])
        self.assertTrue(precise['accepted'],precise)
        self.assertEqual(precise['required_minimum'],.2)
        # Include every complete query cell touched by the requested patch.
        exact_minimum=.8*(.21+.79*(23/32))
        self.assertLessEqual(precise['continuous_lower_bound'],exact_minimum)
        self.assertAlmostEqual(precise['continuous_lower_bound'],
                               exact_minimum,delta=1e-9)
        folded=query.copy();folded[14,25,1]-=.1
        self.assertFalse(validate_retained_composition(
                              parent,folded,regions)['accepted'])

    def test_forged_area_metadata_cannot_authorize_a_folded_older_primitive(self):
        u,v=np.meshgrid(np.arange(9)/8,np.arange(9)/8)
        grid=np.stack([u,v],axis=2);base={'rows':9,'columns':9,'grid':grid.reshape(-1,2).tolist()}
        folded=grid.copy();folded[4,4,1]-=.3
        parent=MeshSampler({**base,'grid':folded.reshape(-1,2).tolist(),
                            'coordinate_transform':True,'preserved_mesh':base,
                            'refinement_regions':[[.25,.25,.75,.75]],
                            'continuous_jacobian_audit':{'accepted':True,'continuous_lower_bound':1.0}})
        report=validate_retained_composition(parent,grid,[[.3,.3,.7,.7]])
        self.assertFalse(report['accepted'])

    def test_folded_new_query_is_rejected_before_parent_area_can_compensate(self):
        parent,u,v,weight=self.nested()
        q=np.stack([u,v],axis=2);q[12,8,1]-=.2
        report=validate_retained_composition(parent,q,[[.2,.2,.8,.8]])
        self.assertFalse(report['accepted'])
        self.assertEqual(report['rejected'],'folded coordinate primitive')

    def test_risky_interval_is_proved_by_subdivision_without_lowering_the_gate(self):
        parent,u,v,weight=self.nested();query=np.stack([u+.005*weight,v-.004*weight],axis=2)
        threshold=.9553948621227335
        coarse=validate_retained_composition(parent,query,[[.2,.2,.8,.8]],
                                            minimum=threshold,refinement_depth=0)
        refined=validate_retained_composition(parent,query,[[.2,.2,.8,.8]],minimum=threshold)
        self.assertFalse(coarse['accepted'])
        self.assertTrue(refined['accepted'])
        self.assertEqual(refined['required_minimum'],threshold)
        self.assertGreater(refined['refined_cell_fragments'],0)


if __name__=='__main__':unittest.main()
