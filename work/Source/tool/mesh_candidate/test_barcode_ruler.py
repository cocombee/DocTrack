import unittest
import cv2
import numpy as np
from barcode_ruler import barcode_rulers
from mesh_ruler import ruler_mesh
from dewarp_mesh import inverse_points,valid

class BarcodeTests(unittest.TestCase):
    def test_sheared_bars_return_to_parallel_vertical_lines(self):
        a=np.full((640,500),235,np.uint8)
        for x in range(90,390,9):
            p=np.array([[x,520],[x+3,520],[x+11,575],[x+8,575]],np.int32)
            cv2.fillConvexPoly(a,p,20)
        guides=barcode_rulers(a)
        self.assertGreaterEqual(len(guides),20)
        grid=ruler_mesh([],[(p,4.) for p in guides])
        self.assertTrue(valid(grid,edge_extension=True))
        self.assertLess(max(np.std(inverse_points(grid,p)[:,0]) for p in guides),.001)

    def test_text_with_tall_letters_is_not_a_barcode(self):
        a=np.full((640,500),235,np.uint8)
        cv2.putText(a,'THIS IS A BILL',(15,350),cv2.FONT_HERSHEY_SIMPLEX,1.7,20,3)
        self.assertEqual(barcode_rulers(a),[])

    def test_table_columns_are_not_a_barcode(self):
        a=np.full((640,500),235,np.uint8)
        for x in range(30,480,70):cv2.line(a,(x,200),(x,290),20,2)
        self.assertEqual(barcode_rulers(a),[])

if __name__=='__main__':unittest.main()
