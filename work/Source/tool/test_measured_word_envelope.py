import unittest
import cv2
import numpy as np
from measured_word_envelope import envelope_consensus


class MeasuredWordEnvelopeTests(unittest.TestCase):
    def word(self, slope=.06, curve=0, changing_height=False):
        gray = np.full((1280,1051),235,np.uint8)
        mask = np.zeros_like(gray)
        for x in range(100,261):
            y = round(200+slope*(x-180)+curve*((x-180)/80)**2)
            height = round(18+(x-100)*.12) if changing_height else 18
            # Connected blur; the profile must not rely on six isolated glyphs.
            mask[y:y+height,x] = 1
        gray[mask>0] = 130
        gray = cv2.GaussianBlur(gray,(0,0),1.8)
        cloud = np.c_[np.nonzero(mask)[1],np.nonzero(mask)[0]]/[1050,1279]
        return gray, cloud

    def test_connected_blurred_word_recovers_tilt_across_full_ink(self):
        gray,cloud = self.word()
        proof = envelope_consensus(gray,cloud,pixel_size=[1050,1279])
        self.assertIsNotNone(proof)
        self.assertGreaterEqual(len(proof['independent_profiles']),3)
        self.assertGreater(proof['coverage'],.75)
        self.assertAlmostEqual(proof['angle'],np.arctan(.06),delta=.005)

    def test_flat_and_curved_bodies_cannot_authorize_a_rigid_rotation(self):
        for options in [dict(slope=0),dict(slope=.06,curve=7),
                        dict(slope=.06,changing_height=True)]:
            gray,cloud = self.word(**options)
            self.assertIsNone(envelope_consensus(gray,cloud,pixel_size=[1050,1279]))

    def test_unowned_tilt_does_not_drive_the_owned_word(self):
        gray,cloud = self.word(slope=0)
        # An unrelated dark stroke must not become evidence through its blur.
        cv2.line(gray,(90,180),(270,190),20,3)
        self.assertIsNone(envelope_consensus(gray,cloud,pixel_size=[1050,1279]))

    def test_preflight_and_word_refinement_use_the_same_optional_evidence(self):
        from rigid_word_row import word_rotation_evidence,refine_word
        from unittest.mock import patch
        gray,cloud=self.word()
        self.assertIsNone(word_rotation_evidence(gray,cloud,pixel_size=[1050,1279]))
        proof=word_rotation_evidence(gray,cloud,pixel_size=[1050,1279],
                                    whole_word_envelope=True)
        self.assertIsNotNone(proof)
        old=np.zeros((2,2,2))
        with patch('rigid_word_row._refine',return_value=(old,{'accepted':False})) as solve:
            _,report=refine_word(old,gray,[cloud],0,[],pixel_size=[1050,1279],
                                 whole_word_envelope=True)
        self.assertEqual(report['independent_glyph_baseline'],proof)
        self.assertEqual(solve.call_count,1)

    def test_invalid_empty_and_unprinted_owners_decline(self):
        gray,cloud = self.word()
        self.assertIsNone(envelope_consensus(np.full_like(gray,235),cloud,pixel_size=[1050,1279]))
        for bad in [np.empty((0,2)),np.array([[np.nan,.3]]),np.array([[1.1,.3]])]:
            self.assertIsNone(envelope_consensus(gray,bad,pixel_size=[1050,1279]))
        self.assertIsNone(envelope_consensus(gray,cloud,pixel_size=[0,1279]))


if __name__ == '__main__':
    unittest.main()
