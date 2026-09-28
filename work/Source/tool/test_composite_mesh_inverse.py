import unittest
import numpy as np
from composite_mesh_sampling import MeshSampler
from composite_mesh_inverse import differential,inverse_near,inverse_retained
from local_mesh_storage import compact


class CompositeInverseTests(unittest.TestCase):
    def sampler(self):
        u,v=np.meshgrid(np.arange(7)/6,np.arange(9)/8)
        parent={'rows':9,'columns':7,'grid':np.stack([u+.05*u*v,v+.03*u*(1-v)],axis=2).reshape(-1,2).tolist()}
        query=np.stack([u,v+.02*(1-u)*(1-v)],axis=2)
        return MeshSampler(compact({'rows':9,'columns':7,'grid':query.reshape(-1,2).tolist(),'preserved_mesh':parent,'coordinate_transform':True,'refinement_regions':[[.2,.2,.8,.8]]}))

    def test_recursive_derivative_matches_independent_difference(self):
        sampler=self.sampler();points=np.array([[.31,.37],[.72,.63],[.1,.15]])
        actual=differential(sampler,points);epsilon=1e-7
        numeric=np.stack([(sampler.sample(points+np.eye(2)[i]*epsilon)-sampler.sample(points-np.eye(2)[i]*epsilon))/(2*epsilon) for i in range(2)],axis=2)
        np.testing.assert_allclose(actual,numeric,atol=2e-9,rtol=0)

    def test_inverse_retains_true_composition(self):
        sampler=self.sampler();points=np.array([[.31,.37],[.72,.63],[.1,.15]])
        recovered=inverse_near(sampler,sampler.sample(points),points+.002)
        np.testing.assert_allclose(recovered,points,atol=1e-11,rtol=0)

    def test_invalid_and_nonconvergent_inverse_are_rejected(self):
        sampler=self.sampler()
        with self.assertRaises(ValueError):inverse_near(sampler,[[np.nan,.3]],[[.3,.3]])
        with self.assertRaises(ValueError):inverse_near(sampler,[[.4,.4]],[[.1,.1]],iterations=0)

    def test_separate_inverse_avoids_alias_fold_in_a_sampled_composition(self):
        from coordinate_stock import minimum_cell_jacobian
        u,v=np.meshgrid(np.arange(17)/16,np.arange(17)/16)
        old=np.stack([u+10*v*v,v],axis=2)
        parent={'rows':17,'columns':17,'grid':old.reshape(-1,2).tolist()}
        u,v=np.meshgrid(np.arange(3)/2,np.arange(3)/2)
        query=np.stack([u,v+.4*u*(1-v)],axis=2)
        sampler=MeshSampler({'rows':3,'columns':3,'grid':query.reshape(-1,2).tolist(),'coordinate_transform':True,'preserved_mesh':parent,'refinement_regions':[[0,0,1,1]]})
        self.assertLess(minimum_cell_jacobian(sampler.dense(3,3)).min(),0)
        points=np.array([[.13,.19],[.37,.53],[.81,.77]])
        recovered=inverse_retained(sampler,sampler.sample(points))
        np.testing.assert_allclose(recovered,points,atol=1e-11,rtol=0)


if __name__=='__main__':unittest.main()
