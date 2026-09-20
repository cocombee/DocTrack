import unittest
import cv2
import numpy as np
from dewarp_mesh import IDENTITY,calibrate_corners,valid,inverse_points
from printed_line_mesh import extract_rules
from mesh_ruler import ruler_mesh

class SurfaceMeshTests(unittest.TestCase):
    def test_corner_calibration_retains_edge_bend(self):
        raw=IDENTITY.copy()
        raw[:,:,0]=.04+.92*raw[:,:,0]+.015*np.sin(np.pi*raw[:,:,1])
        raw[:,:,1]=.03+.94*raw[:,:,1]
        result=calibrate_corners(raw)
        np.testing.assert_allclose(result[[0,0,-1,-1],[0,-1,-1,0]],[[0,0],[1,0],[1,1],[0,1]],atol=1e-6)
        self.assertGreater(result[16,0,0],.01)
        self.assertTrue(valid(result,edge_extension=True))

    def test_partial_slanted_rules_are_traced(self):
        image=np.full((640,400),235,np.uint8)
        for y in (160,290,420):
            for x in range(60,360):image[round(y+.2*(x-200)),x]=20
        rules=extract_rules(image)
        self.assertGreaterEqual(len(rules),3)
        grid=ruler_mesh([(p,4.) for p in rules])
        self.assertTrue(valid(grid,edge_extension=True))
        for p in rules:self.assertLess(np.std(inverse_points(grid,p)[:,1]),.0006)

    def test_rules_near_image_border_do_not_read_outside(self):
        image=np.full((1280,800),235,np.uint8)
        for y in (4,1200,1276):
            cv2.line(image,(5,y),(790,min(1278,y+2)),20,2)
        for rule in extract_rules(image):
            self.assertTrue(np.isfinite(rule).all())
            self.assertTrue(((rule>=0)&(rule<=1)).all())

    def test_exterior_limit_is_solved_without_flattening_mesh(self):
        y=np.linspace(.1,.9,33)
        guides=[(np.column_stack([x+.04*(y-.5),y]),4.) for x in (.7,.85,.95)]
        g=ruler_mesh([],guides)
        self.assertTrue(valid(g,edge_extension=True))
        for p,_ in guides:self.assertLess(np.std(inverse_points(g,p)[:,0]),.001)

if __name__=='__main__':unittest.main()
