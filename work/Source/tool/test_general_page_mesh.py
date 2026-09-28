import unittest
import numpy as np
from general_page_mesh import solve_general


class GeneralPageMeshTests(unittest.TestCase):
    def test_shared_surface_moves_adjacent_material_and_keeps_frame(self):
        x=np.linspace(.1,.9,65)
        curves=[np.c_[x,y+.012*np.sin(np.pi*x)] for y in (.3,.5,.7)]
        grid,certificate=solve_general([(q,4.) for q in curves],[])
        self.assertTrue(certificate['accepted'])
        self.assertEqual(certificate['boundary_maximum_displacement'],0)
        # The unmeasured page between guides also moves with their surface.
        self.assertGreater(abs(grid[13,16,1]-13/32),.001)
        self.assertLess(abs(grid[16,16,1]-grid[13,16,1]-3/32),.003)

    def test_flat_supported_page_stays_identity(self):
        x=np.linspace(.1,.9,65)
        grid,certificate=solve_general([(np.c_[x,np.full_like(x,y)],4.)
                                       for y in (.3,.5,.7)],[])
        u,v=np.meshgrid(np.linspace(0,1,33),np.linspace(0,1,33))
        np.testing.assert_allclose(grid,np.stack([u,v],-1),atol=1e-12)
        self.assertTrue(certificate['accepted'])

    def test_horizontal_and_vertical_evidence_share_one_lattice(self):
        t=np.linspace(.1,.9,65)
        h=[(np.c_[t,.5+.006*np.sin(np.pi*t)],4.)]
        v=[(np.c_[.5+.005*np.sin(np.pi*t),t],4.)]
        grid,certificate=solve_general(h,v)
        self.assertTrue(certificate['accepted'])
        self.assertGreater(abs(grid[16,16,0]-.5),.001)
        self.assertGreater(abs(grid[16,16,1]-.5),.001)


if __name__=='__main__':unittest.main()
