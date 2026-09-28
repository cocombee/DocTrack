import unittest
import numpy as np
from physical_barcode_guides import barcode_components


class PhysicalBarcodeGuideTests(unittest.TestCase):
    def test_recovers_bars_missed_by_line_seeds_without_synthesizing_gaps(self):
        gray = np.full((500, 500), 238, np.uint8)
        safe = np.full_like(gray, 255)
        widths = [3, 5, 4, 7, 3, 5, 6, 4, 8, 3, 4, 6, 5, 3, 7, 4]
        centers = []
        for i, width in enumerate(widths):
            x = 90 + i*18
            y = 180 + i//5
            gray[y:y+74,x:x+width] = 14
            centers.append(x+(width-1)/2)
        # A short neighbouring glyph and one unrelated rule are not bars.
        gray[189:209,385:391] = 14
        gray[20:120,30:34] = 14
        seeds = []
        for i in (0, 1, 3, 4, 6, 7, 8, 10, 11, 13, 14, 15):
            x = centers[i]/499
            y = np.linspace(180+i//5,253+i//5,74)/499
            seeds.append(np.c_[np.full(len(y),x),y])
        result = barcode_components(gray,safe,seeds)
        self.assertEqual(len(result),16)
        self.assertEqual([round(q['center_px'][0],1) for q in result],centers)
        for q in result:
            self.assertGreaterEqual(len(q['points']),74)
            self.assertTrue((q['ink'] >= 0).all() and (q['ink'] <= 1).all())

    def test_sparse_vertical_rules_are_not_a_barcode(self):
        gray = np.full((500,500),238,np.uint8)
        safe = np.full_like(gray,255)
        rules = [np.c_[np.full(74,x/499),np.linspace(180,253,74)/499]
                 for x in (50,110,170,230,290,350,410,470)]
        self.assertEqual(barcode_components(gray,safe,rules),[])


if __name__ == '__main__': unittest.main()
