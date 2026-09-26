import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import cv2
import numpy as np
import protected_ink_mesh as candidate


class ProtectedInkTests(unittest.TestCase):
    def test_paragraph_columns_are_not_words_but_real_sideways_words_are_protected(self):
        gray=np.full((1280,1000),235,np.uint8)
        for y in range(180,920,35):
            cv2.putText(gray,'normal horizontal words have fixed letter spacing',(70,y),
                        cv2.FONT_HERSHEY_SIMPLEX,.9,20,2)
        mask=np.full(gray.shape,255,np.uint8)
        old,structures,_=candidate.accepted.orthogonal_guards(gray,mask)
        self.assertGreater(len(old),len(structures))
        self.assertEqual(len(candidate.verified_orthogonal_ink(gray,mask)),len(structures))
        vertical=np.full((1280,1000),235,np.uint8)
        cv2.putText(vertical,'SIDEWAYS WORDS MUST KEEP THEIR SHAPE',(70,500),
                    cv2.FONT_HERSHEY_SIMPLEX,1.1,20,2)
        vertical=cv2.rotate(vertical,cv2.ROTATE_90_COUNTERCLOCKWISE)
        mask=np.full(vertical.shape,255,np.uint8)
        _,structures,_=candidate.accepted.orthogonal_guards(vertical,mask)
        self.assertGreater(len(candidate.verified_orthogonal_ink(vertical,mask)),len(structures))

    def curved_page(self):
        h,w=1280,1000
        gray=np.full((h,w),235,np.uint8)
        for text,y,scale in (
            ('Preserved working text above the curved row',350,1.1),
            ('This is curved text that must become perfectly straight',420,1.),
            ('Preserved working text below the curved row',490,1.1)):
            cv2.putText(gray,text,(60,y),cv2.FONT_HERSHEY_SIMPLEX,scale,20,2)
        u,v=np.meshgrid(np.arange(w),np.arange(h))
        bend=7*((u-w/2)/(w/2))**2*((v>380)&(v<445))
        return cv2.remap(gray,u.astype(np.float32),(v-bend).astype(np.float32),
                         cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=235)

    def mesh(self):
        u,v=np.meshgrid(np.linspace(0,1,33),np.linspace(0,1,33))
        identity={'rows':33,'columns':33,'grid':np.stack([u,v],-1).reshape(-1,2).tolist(),
                  'method':'original map'}
        return {**identity,'grid':np.stack([u+.025*np.sin(np.pi*v)*np.sin(np.pi*u),v],-1).reshape(-1,2).tolist(),
                'method':'accepted two-axis map','preserved_mesh':identity,
                'refinement_regions':[[.01,.01,.99,.99]]}

    def test_composition_levels_ink_without_sideways_shift_or_glyph_stretch(self):
        original=self.mesh()
        with tempfile.TemporaryDirectory(prefix='doctrack-ink-test-') as folder:
            path=Path(folder)/'page.png'
            cv2.imwrite(str(path),cv2.cvtColor(self.curved_page(),cv2.COLOR_GRAY2BGRA))
            result=candidate.refine(path,dict(original))
        proposals=[p for p in result['ink_guard_audit'] if p['accepted']]
        self.assertTrue(proposals)
        self.assertLess(proposals[0]['after_px'],.1)
        self.assertLess(proposals[0]['glyph_height_change_px'],.1)
        self.assertLess(proposals[0]['sideways_glyph_shift_px'],.01)
        self.assertLess(proposals[0]['protected_shift_px'],.1)
        self.assertEqual(result['preserved_mesh']['grid'],original['grid'])
        self.assertEqual(result['preserved_mesh']['preserved_mesh'],original['preserved_mesh'])

    def test_optional_failure_retains_the_complete_accepted_refinement_chain(self):
        original=self.mesh()
        with patch.object(candidate.accepted,'solve',return_value=original), \
             patch.object(candidate,'refine',side_effect=ValueError('failed ink proposal')):
            result=candidate.solve('unused.png')
        self.assertEqual(result['grid'],original['grid'])
        self.assertEqual(result['preserved_mesh'],original['preserved_mesh'])
        self.assertEqual(result['refinement_regions'],original['refinement_regions'])
        self.assertFalse(result['ink_guard_audit']['accepted'])


if __name__=='__main__':unittest.main()
