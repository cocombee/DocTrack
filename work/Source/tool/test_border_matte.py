import unittest
import cv2,numpy as np
from evaluate_border_matte import paper_matte

def box(a,b):
 p=[{'x':a,'y':a},{'x':b,'y':a},{'x':b,'y':b},{'x':a,'y':b}]
 return [{'start':p[i],'end':p[(i+1)%4]} for i in range(4)]

class MaterialMatteTests(unittest.TestCase):
 def test_exterior_removed_but_printed_border_and_pixels_preserved(self):
  image=np.full((160,160,4),(20,25,30,255),np.uint8)
  image[30:130,30:130]=(220,225,230,255)
  cv2.rectangle(image,(33,33),(126,126),(20,25,30,255),1)
  cv2.putText(image,'ABC',(55,90),cv2.FONT_HERSHEY_SIMPLEX,.5,(20,25,30,255),1)
  out,n=paper_matte(image,box(28/160,132/160))
  self.assertGreater(n,0)
  self.assertEqual(out[80,29,3],0)
  self.assertEqual(out[80,33,3],255)
  np.testing.assert_array_equal(out[:,:,:3],image[:,:,:3])
  np.testing.assert_array_equal(out[40:120,40:120],image[40:120,40:120])

if __name__=='__main__':unittest.main()
