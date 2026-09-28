import unittest
from unittest.mock import patch
import cv2
import numpy as np

import normal_row_candidate as candidate


class NormalRowTests(unittest.TestCase):
    def test_supported_continuation_keeps_independent_words_protected(self):
        mask=np.zeros((400,800),np.uint8)
        mask[80:150,178:183]=1
        mask[80:150,300:340]=1
        guide=np.c_[np.linspace(.3,.8,101),np.full(101,180/799)]
        real=np.array([[180.5/800,100.5/400],[180.5/800,110.5/400]])
        corrected=candidate.supported_rule_word_mask(mask,[real],[guide])
        self.assertEqual(corrected[115,180],0)
        self.assertEqual(corrected[105,180],1)
        self.assertEqual(corrected[105,320],1)
        self.assertEqual(corrected[80,180],1)

    def test_vertical_rule_preserves_whole_words_and_horizontal_normals(self):
        self.check_vertical_rule_preservation(preserve_neighbouring_rules=False)

    def test_transposed_neighbour_guard_preserves_words_and_fixed_normals(self):
        self.check_vertical_rule_preservation(preserve_neighbouring_rules=True)

    def check_vertical_rule_preservation(self, *, preserve_neighbouring_rules):
        gray=np.full((400,800),235,np.uint8)
        y=np.linspace(40,360,321)
        x=180+6*np.sin((y-40)/320*2*np.pi)
        guide=np.c_[y/399,x/799]
        cv2.polylines(gray,[np.rint(np.c_[x,y]).astype('int32')],False,90,1)
        cv2.putText(gray,'PRESERVED WORD',(205,160),cv2.FONT_HERSHEY_SIMPLEX,.6,20,2)
        hy=np.linspace(.08,.92,101)
        horizontal=[np.c_[hy,np.full(101,row)] for row in (.2,.5,.8)]
        vertical=[guide]+[np.c_[hy,np.full(101,col)] for col in (.7,.8)]
        u,v=np.meshgrid(np.linspace(0,1,33),np.linspace(0,1,33))
        old=np.stack([u,v],-1)
        base={'rows':33,'columns':33,'grid':old.reshape(-1,2).tolist(),
              'method':'Accepted','established_rulers':[
                  {'normal_axis':1,'points':q.tolist()} for q in horizontal]}
        yy,xx=np.where(gray[140:165,200:410]<100)
        ink=np.c_[(xx+200)/799,(yy+140)/399]
        measurement=(gray,np.full_like(gray,255),gray,old)
        def sections(image,*args,**kwargs):
            return horizontal if image.shape==gray.shape else vertical
        with patch.object(candidate.rows,'current_measurement',return_value=measurement), \
                patch.object(candidate,'measure_sections',side_effect=sections), \
                patch.object(candidate.rows.p.reference,'extract_rules',return_value=[]), \
                patch.object(candidate.rows.p.reference,'barcode_rulers',return_value=[]), \
                patch.object(candidate.rows.accepted.accepted,'circular_landmarks',return_value=[]):
            result=candidate.refine('unused.png',base,word_ink_groups=[ink],
                                    preserve_neighbouring_rules=preserve_neighbouring_rules)
        self.assertIsNot(result,base)
        self.assertIs(result['preserved_mesh'],base)
        self.assertEqual(result['established_rulers'][:-1],base['established_rulers'])
        self.assertEqual(result['established_rulers'][-1]['normal_axis'],0)
        grid=np.asarray(result['grid']).reshape(result['rows'],result['columns'],2)
        after=candidate.rows.p.inverse(grid,guide[:,::-1])
        self.assertLess(np.ptp(after[:,0])*800,.15)
        moved=candidate.rows.p.inverse(grid,ink)-ink
        self.assertLess(np.ptp(moved,axis=0).max()*800,.01)
        fixed=np.vstack(horizontal)
        self.assertLess(abs(candidate.rows.p.inverse(grid,fixed)[:,1]-fixed[:,1]).max()*400,.01)

    def test_missing_independent_word_evidence_returns_the_same_base(self):
        base={'method':'Accepted'}
        self.assertIs(candidate.refine('unused.png',base),base)


if __name__=='__main__':
    unittest.main()
