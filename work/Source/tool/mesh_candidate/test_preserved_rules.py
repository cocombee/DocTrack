import unittest
import cv2,numpy as np
from preserved_rules import accepted_output_rules,preserves_output_rules
from dewarp_mesh import IDENTITY,inverse_points

class OutputPreservationTests(unittest.TestCase):
    def test_already_straight_printed_lines_cannot_be_bent(self):
        gray=np.full((512,512),235,np.uint8)
        for y in [100,250,400]:cv2.line(gray,(35,y),(480,y),40,2)
        observed=accepted_output_rules(gray,IDENTITY)
        self.assertGreaterEqual(len(observed),3)
        self.assertTrue(preserves_output_rules(IDENTITY,observed,inverse_points))
        bent=IDENTITY.copy();bent[:,:,1]+=.02*np.sin(np.pi*IDENTITY[:,:,0])
        self.assertFalse(preserves_output_rules(bent,observed,inverse_points))

if __name__=='__main__':unittest.main()
