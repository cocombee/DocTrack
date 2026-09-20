import unittest
import numpy as np
from forward_ruler import forward_ruler_mesh
from mesh_ruler import ruler_mesh
from dewarp_mesh import inverse_points,IDENTITY,valid

class RelativeRulerTests(unittest.TestCase):
    def test_separated_fragments_recover_shared_tilt(self):
        guides=[];whole=[]
        for y in [.2,.35,.5,.65,.8]:
            all_points=[]
            for lo,hi in [(.05,.4),(.6,.95)]:
                x=np.linspace(lo,hi,17);p=np.column_stack([x,y+.03*(x-.5)])
                guides.append((p,3));all_points.append(p)
            whole.append(np.concatenate(all_points))
        g=forward_ruler_mesh(guides);old=ruler_mesh(guides)
        self.assertTrue(valid(g,edge_extension=True))
        actual=np.mean([np.std(inverse_points(g,p)[:,1]) for p in whole])
        previous=np.mean([np.std(inverse_points(old,p)[:,1]) for p in whole])
        self.assertLess(actual,.0005)
        self.assertLess(actual,previous*.2)
    def test_curved_rows_preserve_page_height(self):
        x=np.linspace(.05,.95,33)
        rows=[np.column_stack([x,y+.015*np.sin(2*np.pi*x)*(1+y)]) for y in [.2,.35,.5,.65,.8]]
        g=forward_ruler_mesh([(p,3) for p in rows])
        self.assertTrue(valid(g,edge_extension=True))
        recovered=[inverse_points(g,p) for p in rows]
        self.assertGreater(np.mean(recovered[-1][:,1])-np.mean(recovered[0][:,1]),.55)
        self.assertLess(max(np.std(p[:,1]) for p in recovered),.001)

    def test_flat_text_keeps_identity(self):
        guides=[(np.column_stack([np.linspace(.1,.9,33),np.full(33,y)]),3) for y in [.2,.5,.8]]
        np.testing.assert_allclose(forward_ruler_mesh(guides),IDENTITY,atol=1e-9)

if __name__=='__main__':unittest.main()
