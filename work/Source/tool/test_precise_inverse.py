import unittest
from unittest.mock import patch
import numpy as np
from local_receipt_mesh import sample,valid
from precise_mesh_inverse import inverse, _cell_inverse


class PreciseInverseTests(unittest.TestCase):
    def test_search_verifies_actual_bilinear_root_when_local_iteration_stalls(self):
        x,y=np.meshgrid(np.linspace(0,1,17),np.linspace(0,1,25))
        grid=np.stack([x+.1*np.sin(y*15)*x*(1-x),
                       y+.08*x*y*(1-y)],-1)
        self.assertTrue(valid(grid))
        uv=np.array([[.22,.31],[.5,.5],[.8,.77],[0,0],[1,1]])
        source=sample(grid,uv)
        with patch('precise_mesh_inverse._newton_inverse',return_value=np.zeros_like(uv)):
            recovered=inverse(grid,source)
        np.testing.assert_allclose(recovered,uv,atol=1e-10)
        np.testing.assert_allclose(sample(grid,recovered),source,atol=1e-11)

    def test_search_rejects_nonexistent_or_ambiguous_inverse(self):
        x,y=np.meshgrid(np.linspace(0,1,5),np.linspace(0,1,5))
        identity=np.stack([x,y],-1)
        with self.assertRaisesRegex(ValueError,'no exact inverse'):
            _cell_inverse(identity,np.array([1.2,.5]))
        folded=np.stack([1-abs(2*x-1),y],-1)
        with self.assertRaisesRegex(ValueError,'more than one inverse'):
            _cell_inverse(folded,np.array([.6,.5]))

    def test_recovers_points_across_abrupt_compression_and_curved_cells(self):
        x,y=np.meshgrid(np.linspace(0,1,513),np.linspace(0,1,1025))
        normal=np.interp(y,[0,.49,.50,.51,1],[0,.495,.498,.505,1])
        grid=np.stack([x+.015*np.sin(normal*25)*x*(1-x),normal],-1)
        self.assertTrue(valid(grid))
        rng=np.random.default_rng(41)
        uv=rng.uniform(.01,.99,(5000,2))
        recovered=inverse(grid,sample(grid,uv))
        np.testing.assert_allclose(recovered,uv,atol=1e-10)


if __name__=='__main__':
    unittest.main()
