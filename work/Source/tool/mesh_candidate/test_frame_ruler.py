import unittest
import cv2
import numpy as np
from frame_ruler import frame_rulers,band_rulers

class FrameRulerTests(unittest.TestCase):
    def test_rectangular_print_panel_has_four_guides(self):
        im=np.full((800,640),235,np.uint8)
        cv2.rectangle(im,(100,100),(500,650),110,-1)
        h,v=frame_rulers(im)
        self.assertEqual((len(h),len(v)),(2,2))
        for p in h:self.assertLess(np.ptp(p[:,1]),.002)
        for p in v:self.assertLess(np.ptp(p[:,0]),.002)

    def test_round_shape_does_not_become_rectangle(self):
        im=np.full((800,640),235,np.uint8)
        cv2.ellipse(im,(320,400),(210,290),0,0,360,100,-1)
        self.assertEqual(frame_rulers(im),([],[]))

    def test_no_panel_on_blank_paper(self):
        self.assertEqual(frame_rulers(np.full((800,640),230,np.uint8)),([],[]))

class BandRulerTests(unittest.TestCase):
    def band(self):
        image=np.full((900,640,3),235,np.uint8)
        for y in range(50,850):
            x=70+round(6*np.sin(y/899*2*np.pi))
            image[y,x-16:x+17]=[60,150,70]
        return image

    def test_decorative_band_alone_cannot_warp_page(self):
        self.assertEqual(band_rulers(self.band(),[]),[])
        y=np.linspace(.1,.9,50)
        straight=[np.column_stack([x+0*y,y]) for x in (.35,.7)]
        self.assertEqual(band_rulers(self.band(),straight),[])

    def test_two_columns_corroborate_band_bending(self):
        y=np.linspace(.1,.9,50)
        columns=[np.column_stack([x+6/639*np.sin(y*2*np.pi),y]) for x in (.35,.7)]
        self.assertEqual(len(band_rulers(self.band(),columns)),2)

if __name__=='__main__':unittest.main()
