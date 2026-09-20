import unittest
import cv2
import numpy as np
from printed_line_mesh import extract_rules,rule_mesh
from dewarp_mesh import valid,inverse_points
from mesh_ruler import ruler_mesh,circular_landmarks

class PrintedRuleTests(unittest.TestCase):
    def test_bowed_slanted_rules_become_horizontal(self):
        im=np.full((640,400),235,np.uint8)
        for baseline in (180,310,440):
            for x in range(24,380):
                if x%9<6:
                    y=round(baseline+12*(x/400-.5)+8*(x/400-.5)**2)
                    im[y,x]=30
        rules=extract_rules(im)
        self.assertGreaterEqual(len(rules),2)
        grid=rule_mesh(rules)
        self.assertTrue(valid(grid))
        for rule in rules:
            corrected=inverse_points(grid,rule)
            self.assertLess(np.std(corrected[:,1]),np.std(rule[:,1])*.25)
        np.testing.assert_allclose(grid[0,:,1],0)
        np.testing.assert_allclose(grid[-1,:,1],1)
        np.testing.assert_allclose(grid[:,0,0],0)
        np.testing.assert_allclose(grid[:,-1,0],1)
    def test_joint_rulers_straighten_both_axes(self):
        x=np.linspace(.08,.92,65)
        horizontal=[(np.column_stack([x,y+.012*np.sin(np.pi*x)]),4.) for y in (.2,.5,.8)]
        vertical=[(np.column_stack([xx+.01*np.sin(np.pi*x),x]),4.) for xx in (.2,.5,.8)]
        grid=ruler_mesh(horizontal,vertical)
        self.assertTrue(valid(grid,edge_extension=True))
        for axis,guides in ((1,horizontal),(0,vertical)):
            for observed,_ in guides:
                self.assertLess(np.std(inverse_points(grid,observed)[:,axis]),.0005)

    def test_bottom_text_keeps_last_two_rules_correction(self):
        x=np.linspace(.07,.95,33)
        bend=.015*np.abs(x-.5)
        rules=[np.column_stack([x,y+bend]) for y in (.82,.92)]
        grid=rule_mesh(rules)
        text=np.column_stack([x,.965+bend])
        self.assertTrue(valid(grid,edge_extension=True))
        self.assertLess(np.std(inverse_points(grid,text)[:,1]),.00015)

    def test_flat_rulers_do_not_warp_round_shape(self):
        x=np.linspace(.05,.95,33)
        grid=ruler_mesh([(np.column_stack([x,y+0*x]),4.) for y in (.2,.8)],
            [(np.column_stack([xx+0*x,x]),4.) for xx in (.2,.8)])
        theta=np.linspace(0,2*np.pi,50)
        circle=np.column_stack([.5+.1*np.cos(theta),.5+.1*np.sin(theta)])
        np.testing.assert_allclose(inverse_points(grid,circle),circle,atol=1e-8)
        im=np.full((400,400),235,np.uint8)
        cv2.circle(im,(200,200),35,30,2)
        self.assertGreaterEqual(len(circular_landmarks(im)),1)

    def test_blank_surface_has_no_invented_alignment(self):
        self.assertEqual(extract_rules(np.full((640,400),220,np.uint8)),[])
        self.assertIsNone(rule_mesh([]))

if __name__=='__main__':unittest.main()
