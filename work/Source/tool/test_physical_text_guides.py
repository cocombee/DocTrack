import unittest
import numpy as np
from physical_text_guides import (associated_rows,measured_text_rows,
                                  _anchors,_typographic_body_curve)


class PhysicalTextGuideTests(unittest.TestCase):
    def test_two_neighbouring_words_share_one_physical_line(self):
        w,h=540,260
        def word(left,right,top,bottom):
            quad=np.array([[left,top],[right,top],[right,bottom],[left,bottom]],float)
            return dict(text='word',confidence=.9,points=(quad/[w,h]).tolist())
        rows=associated_rows([word(80,120,80,105),word(125,172,81,106),
                              word(180,225,122,147)],w,h)
        self.assertEqual(sorted(map(len,rows)),[1,2])

    def test_one_tall_ocr_box_splits_two_actual_text_lines(self):
        h,w=260,540
        gray=np.full((h,w),240,np.uint8);safe=np.full_like(gray,255)
        for x in range(85,430,14):
            y=int(72+.00025*(x-250)**2)
            gray[y:y+12,x:x+5]=20
        for x in range(140,375,14):
            y=int(117+.0002*(x-250)**2)
            gray[y:y+12,x:x+5]=20
        before=gray.copy()
        boxes=[np.array([[left,60],[right,60],[right,150],[left,150]],float)/[w-1,h-1]
               for left,right in ((80,200),(205,330),(335,435))]
        lines=measured_text_rows(gray,safe,[boxes])
        self.assertEqual(len(lines),2)
        self.assertEqual([q['branch'] for q in lines],[0,1])
        self.assertLess(lines[0]['points'][:,1].mean(),lines[1]['points'][:,1].mean())
        self.assertTrue(all(q['agreement_px']<2.8 for q in lines))
        np.testing.assert_array_equal(gray,before)

    def test_printed_dashes_do_not_become_glyph_lines(self):
        h,w=260,540
        gray=np.full((h,w),240,np.uint8);safe=np.full_like(gray,255)
        for x in range(85,430,14):gray[83:85,x:x+7]=20
        boxes=[np.array([[left,60],[right,60],[right,110],[left,110]],float)/[w-1,h-1]
               for left,right in ((80,200),(205,330),(335,435))]
        self.assertEqual(measured_text_rows(gray,safe,[boxes]),[])

    def test_short_supported_word_gets_a_weak_guide(self):
        h,w=260,540
        gray=np.full((h,w),240,np.uint8);safe=np.full_like(gray,255)
        for x in range(85,181,17):gray[90:104,x:x+5]=20
        box=np.array([[80,83],[190,83],[190,113],[80,113]],float)/[w-1,h-1]
        guides=measured_text_rows(gray,safe,[[box]])
        self.assertEqual(len(guides),1)
        self.assertTrue(guides[0]['compact_word_guide'])
        self.assertLess(guides[0]['agreement_px'],1)

    def test_small_lower_mark_is_not_a_base_glyph_anchor(self):
        gray=np.full((1280,800),240,np.uint8)
        safe=np.full_like(gray,255)
        for x in range(100,701,30):
            gray[500:518,x:x+7]=20
        # A dot under one letter is real ink, yet it is not another base
        # character and must not supply a baseline sample.
        gray[522:526,410:414]=20
        anchors=_anchors(gray,safe,10)
        self.assertEqual(len(anchors),21)
        self.assertFalse(np.any((anchors[:,0]>405)&(anchors[:,0]<420)&
                                (anchors[:,1]>520)))
        self.assertEqual(int((gray[522:526,410:414]==20).sum()),16)

    def test_typographic_curve_keeps_base_glyphs_and_ignores_descender(self):
        # A single quadratic misses three adjacent base letters in a locally
        # curved row. A lowercase descender is only a lower outlier.
        x=np.linspace(130,690,24)
        baseline=1218-10*np.exp(-((x-410)/120)**4)
        y=baseline.copy()
        y[8]+=5
        points=np.c_[x,y,np.full_like(x,18)]
        curve=_typographic_body_curve(points,18,400)
        self.assertIsNotNone(curve)
        predicted=np.interp(x,curve['points'][:,0],curve['points'][:,1])
        self.assertLess(np.max(np.abs(predicted[10:14]-baseline[10:14])),2)
        self.assertLess(abs(predicted[8]-baseline[8]),2.5)
        self.assertEqual(curve['support'],24)


if __name__=='__main__':unittest.main()
