import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

import receipt_content_mesh as candidate
import local_receipt_mesh as local


class ReceiptContentTests(unittest.TestCase):
    def test_local_failure_retains_preferred_original(self):
        original = {'rows': 33, 'columns': 33,
                    'grid': candidate.base.IDENTITY.reshape(-1, 2).tolist(),
                    'method': 'Shared mesh rulers'}
        for error in (ValueError('failed proposal'), KeyError('missing guide')):
            with patch.object(candidate.base, 'solve', return_value=dict(original)), \
                 patch.object(candidate, 'refine', side_effect=error):
                result = local.solve('unused.png')
            self.assertEqual(result['grid'], original['grid'])
            self.assertEqual(result['method'], original['method'])
            self.assertFalse(result['local_combination']['accepted'])

    def test_local_correction_requires_accepted_content_evidence(self):
        original = {'rows': 33, 'columns': 33,
                    'grid': candidate.base.IDENTITY.reshape(-1, 2).tolist(),
                    'method': 'Identity mesh · insufficient deformation evidence'}
        with patch.object(candidate.base, 'solve', return_value=original):
            self.assertIs(local.solve('unused.png'), original)

    def barcode(self):
        gray = np.full((640, 500), 235, np.uint8)
        for x in range(90, 390, 9):
            y = round(110 + 40 * ((x - 240) / 300)**2)
            cv2.rectangle(gray, (x, y), (x + 3, y + 65), 20, -1)
        return gray

    def test_barcode_envelopes_retain_observed_bow_before_correction(self):
        gray = self.barcode()
        bars = candidate.barcode_rulers(gray)
        guides = candidate.measured_envelopes(gray, bars)
        self.assertEqual(len(guides), 2)
        self.assertGreater(np.ptp(guides[0][:, 1])*640, 4)
        grid = candidate.ruler_mesh([(p, 4.) for p in guides],
                                    [(p, 4.) for p in bars])
        self.assertTrue(candidate.base.valid(grid, edge_extension=True))
        self.assertLess(max(np.std(candidate.base.inverse_points(grid, p)[:, 1])*640
                            for p in guides), .5)

    def test_unrelated_barcodes_do_not_form_one_envelope(self):
        bars = []
        for top in (.15, .6):
            for x in np.linspace(.2, .8, 20):
                bars.append(np.column_stack([np.full(17, x),
                                             np.linspace(top, top + .1, 17)]))
        self.assertEqual(candidate.measured_envelopes(self.barcode(), bars), [])

    def test_missed_dark_caption_rows_supply_independent_guides(self):
        gray = np.full((640, 800), 235, np.uint8)
        for y in (400, 445):
            cv2.putText(gray, 'ALL PRINTED TEXT STAYS STRAIGHT', (40, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, 20, 2)
        guides = candidate.supplemental_text(gray, [])
        self.assertGreaterEqual(len(guides), 2)
        self.assertEqual(candidate.supplemental_text(gray, guides), [])

    def test_invalid_refinement_returns_original_mesh(self):
        gray = self.barcode()
        for y in (300, 360, 420):
            cv2.line(gray, (40, y), (460, y), 20, 1)
        original = {'rows': 33, 'columns': 33,
                    'grid': candidate.base.IDENTITY.reshape(-1, 2).tolist(),
                    'method': 'Shared mesh rulers'}
        x = np.linspace(.2, .8, 33)
        captions = [np.column_stack([x, .7 + .04*x*x]),
                    np.column_stack([x, .8 + .04*x*x])]
        with tempfile.TemporaryDirectory(prefix='doctrack-receipt-test-') as folder:
            path = Path(folder)/'receipt.png'
            cv2.imwrite(str(path), cv2.cvtColor(gray, cv2.COLOR_GRAY2BGRA))
            with patch.object(candidate.base, 'solve', return_value=dict(original)), \
                 patch.object(candidate, 'supplemental_text', return_value=captions), \
                 patch.object(candidate, 'ruler_mesh', return_value=candidate.base.IDENTITY + .3):
                result = candidate.solve(path)
            with patch.object(candidate.base, 'solve', return_value=dict(original)), \
                 patch.object(candidate, 'refine', side_effect=ValueError('failed proposal')):
                failed = candidate.solve(path)
        self.assertEqual(result['grid'], original['grid'])
        self.assertEqual(result['method'], original['method'])
        self.assertFalse(result['content_refinement']['accepted'])
        self.assertEqual(failed['grid'], original['grid'])
        self.assertEqual(failed['method'], original['method'])
        self.assertEqual(failed['content_refinement']['error'], 'ValueError')


if __name__ == '__main__':
    unittest.main()
