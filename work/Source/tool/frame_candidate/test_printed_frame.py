import unittest
import cv2
import numpy as np
from printed_frame import printed_frame

class PrintedFrameTests(unittest.TestCase):
    def test_closed_border_returns_four_sides(self):
        gray=np.full((400,600),230,np.uint8)
        cv2.rectangle(gray,(22,20),(579,379),30,2)
        h,v=printed_frame(gray)
        self.assertEqual((len(h),len(v)),(2,2))
        self.assertLess(np.ptp(v[0][:,0]),.003)
    def test_clipped_frame_cannot_establish_four_sides(self):
        gray=np.full((400,600),230,np.uint8)
        cv2.rectangle(gray,(-5,20),(579,379),30,2)
        self.assertEqual(printed_frame(gray),([],[]))
    def test_ellipse_is_not_a_rectangular_frame(self):
        gray=np.full((400,600),230,np.uint8)
        cv2.ellipse(gray,(300,200),(277,177),0,0,360,30,2)
        self.assertEqual(printed_frame(gray),([],[]))

if __name__=='__main__':unittest.main()
