import unittest
import cv2
import numpy as np
from measured_ink_bands import rule_ink_footprints


class MeasuredRuleInkTests(unittest.TestCase):
    def example(self):
        h,w=800,1280;gray=np.full((h,w),210,np.uint8)
        cv2.line(gray,(20,110),(1250,110),80,2)
        cv2.putText(gray,'STORE',(220,143),cv2.FONT_HERSHEY_SIMPLEX,.8,70,2)
        gray=cv2.GaussianBlur(gray,(0,0),1.5)
        contrast=cv2.GaussianBlur(gray.astype(np.float32),(0,0),7)-gray
        y,x=np.where((contrast>6)&(np.indices(gray.shape)[0]>125))
        word=np.c_[x/(w-1),y/(h-1)]
        guide=np.c_[np.arange(20,1251)/(w-1),np.full(1231,110/(h-1))]
        return gray,np.full_like(gray,255),word,guide

    def test_complete_blurred_stroke_extends_beyond_center_guard_without_owning_text(self):
        gray,safe,word,guide=self.example()
        clouds=rule_ink_footprints(gray,safe,[guide],[],[word])
        self.assertEqual(len(clouds),1)
        points=clouds[0]['points'];pixels=np.rint(points*[1279,799]).astype(int)
        self.assertGreaterEqual(np.ptp(pixels[:,1]),6)
        text=set(map(tuple,np.rint(word*[1279,799]).astype(int)))
        self.assertFalse(set(map(tuple,pixels))&text)
        self.assertEqual(clouds[0]['normal_axis'],1)

    def test_transposed_native_raster_preserves_the_same_complete_stroke(self):
        gray,safe,word,guide=self.example()
        self.assertFalse(gray.T.flags.c_contiguous)
        normal=rule_ink_footprints(gray,safe,[guide],[],[word])[0]
        vertical=rule_ink_footprints(gray.T,safe.T,[],[guide],[word[:,::-1]])[0]
        a=np.rint(normal['points']*[1279,799]).astype(int)
        b=np.rint(vertical['points'][:,::-1]*[1279,799]).astype(int)
        self.assertEqual(set(map(tuple,a)),set(map(tuple,b)))
        self.assertEqual(vertical['normal_axis'],0)

    def test_blank_unsafe_and_invalid_measurements_cannot_supply_ink(self):
        gray,safe,word,guide=self.example()
        self.assertFalse(rule_ink_footprints(np.full_like(gray,210),safe,[guide],[],[]))
        self.assertFalse(rule_ink_footprints(gray,np.zeros_like(safe),[guide],[],[word]))
        with self.assertRaises(ValueError):
            rule_ink_footprints(gray,safe,[np.array([[np.nan,0],[1,1]])],[],[word])


if __name__=='__main__':unittest.main()
