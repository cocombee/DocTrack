import unittest
import numpy as np
from composite_mesh_sampling import MeshSampler
from composite_mesh_inverse import differential
from local_mesh_storage import compact
from normal_composition_guard import validate_normal_composition,validate_source_composition,owned_rectangles,exact_cell


class NormalCompositionGuardTests(unittest.TestCase):
    def test_checks_retained_patch_and_parent_in_their_actual_domains(self):
        u,v=np.meshgrid(np.arange(9)/8,np.arange(13)/12)
        grid=np.stack([u,v],axis=2)
        parent={'rows':13,'columns':9,'grid':grid.reshape(-1,2).tolist()}
        changed=grid.copy();changed[:,:,1]*=.5
        mesh=MeshSampler(compact({'rows':13,'columns':9,'grid':changed.reshape(-1,2).tolist(),'preserved_mesh':parent,'refinement_regions':[[.25,.25,.75,.75]]}))
        query=grid.copy();query[:,:,1]=v*.8
        result=validate_normal_composition(mesh,query,[[.1,.1,.9,.9]])
        self.assertTrue(result['accepted'])
        self.assertAlmostEqual(result['continuous_lower_bound'],.4,places=12)
        result=validate_normal_composition(mesh,query,[[.1,.1,.9,.9]],minimum=.45)
        self.assertFalse(result['accepted'])
        self.assertGreater(result['analytically_minimized_cells'],0)
        self.assertAlmostEqual(result['continuous_lower_bound'],.4,places=12)

    def test_exact_minimum_bounds_dense_independent_derivatives(self):
        u,v=np.meshgrid(np.arange(5)/4,np.arange(6)/5)
        old=np.stack([u+.08*u*v,v+.13*u*(1-v)],axis=2)
        layer=MeshSampler({'rows':6,'columns':5,'grid':old.reshape(-1,2).tolist()})
        q=np.stack([u,v+.06*(1-u)*v*(1-v)],axis=2)
        for row,col in [(0,0),(2,1),(4,3)]:
            result=exact_cell(layer,[0,0,1,1],q,row,col)
            # At a shared new-cell boundary, the sampler selects the next
            # cell's one-sided derivative. Check this cell's interior limits;
            # the guard independently checks the neighbour's derivative too.
            x,y=np.meshgrid(np.linspace(col/4+1e-8,(col+1)/4-1e-8,91),np.linspace(row/5+1e-8,(row+1)/5-1e-8,91))
            points=np.c_[x.ravel(),y.ravel()]
            query_sampler=MeshSampler({'rows':6,'columns':5,'grid':q.reshape(-1,2).tolist()})
            j=differential(layer,query_sampler.sample(points)) @ differential(query_sampler,points)
            numerical=np.linalg.det(j).min()
            self.assertLessEqual(result,numerical+1e-10)
            self.assertLess(numerical-result,1e-4)

    def test_false_topology_and_unsupported_tangent_warp_are_rejected(self):
        u,v=np.meshgrid(np.arange(5)/4,np.arange(5)/4)
        grid=np.stack([u,v],axis=2);layer=MeshSampler({'rows':5,'columns':5,'grid':grid.reshape(-1,2).tolist()})
        query=grid.copy();query[:,:,1]=v*.1
        result=validate_normal_composition(layer,query,[[0,0,1,1]])
        self.assertFalse(result['accepted'])
        query[:,:,0]+=.01
        with self.assertRaises(ValueError):validate_normal_composition(layer,query,[[0,0,1,1]])

    def test_global_proof_covers_both_derivatives_at_shared_seams(self):
        u,v=np.meshgrid(np.arange(5)/4,np.arange(6)/5)
        old=np.stack([u+.08*u*v,v+.13*u*(1-v)],axis=2)
        layer=MeshSampler({'rows':6,'columns':5,'grid':old.reshape(-1,2).tolist()})
        q=np.stack([u,v+.06*(1-u)*v*(1-v)],axis=2)
        result=validate_normal_composition(layer,q,[[0,0,1,1]],minimum=1.1)
        query_sampler=MeshSampler({'rows':6,'columns':5,'grid':q.reshape(-1,2).tolist()})
        x,y=np.meshgrid(np.linspace(0,1,401),np.linspace(0,1,401))
        points=np.c_[x.ravel(),y.ravel()]
        j=differential(layer,query_sampler.sample(points)) @ differential(query_sampler,points)
        numerical=np.linalg.det(j).min()
        self.assertLessEqual(result['continuous_lower_bound'],numerical+1e-10)
        self.assertLess(numerical-result['continuous_lower_bound'],1e-4)

    def test_disjoint_later_coordinate_step_uses_exact_older_layers(self):
        u,v=np.meshgrid(np.arange(9)/8,np.arange(9)/8)
        identity=np.stack([u,v],axis=2)
        parent={'rows':9,'columns':9,'grid':identity.reshape(-1,2).tolist()}
        query=identity.copy();query[4,4,1]+=.02
        prior=MeshSampler({'rows':9,'columns':9,'grid':query.reshape(-1,2).tolist(),'coordinate_transform':True,'preserved_mesh':parent,'refinement_regions':[[.375,.375,.625,.625]]})
        next_query=identity.copy();next_query[:,:,1]*=.9
        report=validate_normal_composition(prior,next_query,[[.05,.05,.2,.2]])
        self.assertTrue(report['accepted'])
        self.assertAlmostEqual(report['continuous_lower_bound'],.9,places=12)
        with self.assertRaisesRegex(ValueError,'own continuous proof'):
            validate_normal_composition(prior,next_query,[[.4,.4,.6,.6]])

    def test_general_composition_bound_contains_dense_derivative_minimum(self):
        u,v=np.meshgrid(np.arange(9)/8,np.arange(13)/12)
        old=np.stack([u+.05*u*v,v+.03*u*(1-v)],axis=2)
        layer=MeshSampler({'rows':13,'columns':9,'grid':old.reshape(-1,2).tolist()})
        query=np.stack([u+.015*v*(1-v),v-.02*u*v*(1-v)],axis=2)
        report=validate_source_composition(layer,query,[[0,0,1,1]])
        self.assertTrue(report['accepted'])
        sampler=MeshSampler({'rows':13,'columns':9,'grid':query.reshape(-1,2).tolist()})
        x,y=np.meshgrid(np.linspace(.001,.99,121),np.linspace(.001,.99,121));points=np.c_[x.ravel(),y.ravel()]
        actual=np.linalg.det(differential(layer,sampler.sample(points)) @ differential(sampler,points)).min()
        self.assertLessEqual(report['continuous_lower_bound'],actual+1e-10)
        compressed=query.copy();compressed[:,:,1]*=.1
        self.assertFalse(validate_source_composition(layer,compressed,[[0,0,1,1]])['accepted'])


if __name__=='__main__':unittest.main()
