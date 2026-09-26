import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import cv2
import numpy as np
import residual_content_mesh as candidate
from content_column_rulers import correct_columns


class ContentResidualTests(unittest.TestCase):
    def identity(self):
        u,v = np.meshgrid(np.linspace(0,1,33),np.linspace(0,1,33))
        return {'rows':33,'columns':33,'grid':np.stack([u,v],-1).reshape(-1,2).tolist(),
                'method':'Accepted original'}

    def curved_line(self, position):
        h,w = 1280,1000
        gray = np.full((h,w),235,np.uint8)
        baseline = round(position*h)
        cv2.putText(gray,'WORKING TEXT AND LINES STAY FIXED',(80,baseline),
                    cv2.FONT_HERSHEY_SIMPLEX,1.3,20,2)
        u,v = np.meshgrid(np.arange(w),np.arange(h))
        bend = 12*((u-w/2)/(w/2))**2
        bent = cv2.remap(gray,u.astype(np.float32),(v-bend).astype(np.float32),cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT,borderValue=235)
        return cv2.cvtColor(bent,cv2.COLOR_GRAY2BGRA)

    def test_supported_residual_is_corrected_at_top_middle_and_bottom(self):
        with tempfile.TemporaryDirectory(prefix='doctrack-residual-test-') as folder:
            for location in (.18,.5,.82):
                path=Path(folder)/'page.png'
                cv2.imwrite(str(path),self.curved_line(location))
                result=candidate.refine(path,self.identity())
                accepted=[p for p in result['residual_audit']['proposals'] if p.get('accepted')]
                self.assertTrue(accepted, f'no correction at {location}: {result["residual_audit"]}')
                self.assertLess(accepted[0]['after_px'],.3)
                self.assertIn('preserved_mesh',result)

    def test_failed_optional_pass_retains_established_mesh(self):
        original=self.identity()
        with patch.object(candidate.prior,'solve',return_value=original), \
             patch.object(candidate,'refine',side_effect=ValueError('failed evidence')):
            result=candidate.solve('unused.png')
        self.assertEqual(result['grid'],original['grid'])
        self.assertEqual(result['method'],'Accepted original')
        self.assertFalse(result['residual_audit']['accepted'])

    def test_nearby_fragments_join_without_merging_adjacent_rows(self):
        grid=candidate.grid_of(self.identity())
        left=np.array([[.15,.30],[.45,.30]])
        right=np.array([[.48,.302],[.85,.302]])
        next_row=np.array([[.15,.32],[.85,.32]])
        rows=candidate.canonical_text_rows([left,right,next_row],grid)
        self.assertEqual(len(rows),2)
        self.assertAlmostEqual(np.ptp(rows[0][:,0]),.7)

    def test_coherent_paragraph_levels_without_moving_separate_established_text(self):
        h,w=1280,1000
        gray=np.full((h,w),235,np.uint8)
        for y in range(300,475,35):
            cv2.putText(gray,'WORKING TEXT AND LINES STAY FIXED',(80,y),
                        cv2.FONT_HERSHEY_SIMPLEX,1.2,20,2)
        cv2.putText(gray,'PRESERVED ESTABLISHED TEXT BELOW',(80,850),
                    cv2.FONT_HERSHEY_SIMPLEX,1.2,20,2)
        u,v=np.meshgrid(np.arange(w),np.arange(h))
        bend=10*((u-w/2)/(w/2))**2*((v>270)&(v<490))
        bent=cv2.remap(gray,u.astype(np.float32),(v-bend).astype(np.float32),
                       cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=235)
        with tempfile.TemporaryDirectory(prefix='doctrack-paragraph-test-') as folder:
            path=Path(folder)/'page.png'
            cv2.imwrite(str(path),cv2.cvtColor(bent,cv2.COLOR_GRAY2BGRA))
            result=candidate.refine(path,self.identity())
        groups=[p for p in result['residual_audit']['proposals']
                if p['kind']=='coherent_text_group' and p['accepted']]
        self.assertEqual(len(groups),1)
        self.assertEqual(groups[0]['rows'],5)
        self.assertLess(groups[0]['after_max_px'],.1)
        fixed=np.column_stack([np.linspace(.1,.8,33),np.full(33,850/(h-1))])
        np.testing.assert_allclose(candidate.prior.inverse(candidate.grid_of(result),fixed),fixed,
                                   atol=1e-12)

    def test_repeated_bent_margins_correct_both_sides_and_respect_crossed_rule(self):
        h,w=1280,1000
        gray=np.full((h,w),235,np.uint8)
        for y in range(160,920,60):
            cv2.putText(gray,'WORKING TEXT AND LINES STAY FIXED',(110,y),
                        cv2.FONT_HERSHEY_SIMPLEX,1.1,20,2)
        u,v=np.meshgrid(np.arange(w),np.arange(h))
        drift=10*np.sin((v-160)/760*np.pi)
        gray=cv2.remap(gray,(u-drift).astype(np.float32),v.astype(np.float32),
                       cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=235)
        grid=candidate.grid_of(self.identity())
        corrected,audit,regions=correct_columns(grid,gray,[],[],[],[],[],[])
        self.assertEqual(len([p for p in audit if p['accepted']]),2)
        self.assertTrue(candidate.prior.valid(corrected))
        self.assertTrue(regions)
        self.assertTrue(all(p['after_px']<.1 for p in audit))
        rule=np.column_stack([np.linspace(.08,.94,257),np.full(257,.3)])
        protected,audit,regions=correct_columns(grid,gray,[],[rule],[],[],[],[])
        self.assertFalse(any(p['accepted'] for p in audit))
        self.assertTrue(np.array_equal(protected,grid))
        self.assertFalse(regions)


if __name__=='__main__':
    unittest.main()
