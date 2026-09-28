import tempfile
import unittest
from pathlib import Path
import cv2
import numpy as np
from native_word_evidence import request_words,validated_words,ink_groups,ink_observations,recover_faint_components,measurement_words,merge_crop_words,request_words_with_crops


WORD={'text':'Invoice','confidence':.9,
      'points':[[.1,.1],[.5,.1],[.5,.3],[.1,.3]]}


class NativeWordEvidenceTests(unittest.TestCase):
    def test_faint_recovery_captures_compact_fragment_and_excludes_nearby_rule(self):
        gray=np.full((400,400),230,np.uint8)
        gray[23:30,24:29]=80;gray[23:29,19:21]=85
        gray[17,15:55]=70
        word={**WORD,'points':(np.array([[22,20],[50,20],[50,40],[22,40]])/400).tolist()}
        before=ink_observations(gray,[word])
        after,audit=recover_faint_components(gray,before,[word],halo=3)
        self.assertEqual(len(audit['components']),1)
        self.assertEqual(audit['components'][0]['new_points'],12)
        x,y=np.rint(after[0]['ink']*399).astype(int).T
        self.assertFalse((y==17).any())
        old={tuple(q) for q in before[0]['ink']}
        self.assertTrue(old.issubset({tuple(q) for q in after[0]['ink']}))
        np.testing.assert_array_equal(after[0]['quad'],before[0]['quad'])

    def test_faint_recovery_keeps_ambiguous_fragment_unassigned(self):
        gray=np.full((400,400),230,np.uint8)
        gray[23:30,14:18]=80;gray[23:30,29:33]=80
        gray[23:29,22:24]=85
        words=[{**WORD,'points':(np.array([[12,20],[20,20],[20,40],[12,40]])/400).tolist()},
               {**WORD,'points':(np.array([[26,20],[35,20],[35,40],[26,40]])/400).tolist()}]
        before=ink_observations(gray,words)
        after,audit=recover_faint_components(gray,before,words,halo=3)
        self.assertEqual(audit['components'],[])
        for a,z in zip(before,after):np.testing.assert_array_equal(a['ink'],z['ink'])

    def test_optional_halo_captures_faint_edge_with_exact_native_pairing(self):
        gray=np.full((100,100),230,np.uint8);gray[23:29,19:21]=85
        word={**WORD,'points':[[.22,.2],[.5,.2],[.5,.4],[.22,.4]]}
        self.assertEqual(ink_observations(gray,[word]),[])
        records=ink_observations(gray,[word],localization_halo=3)
        self.assertEqual(len(records),1)
        np.testing.assert_array_equal(records[0]['quad'],
                            (np.asarray(word['points'])*100-.5)/99)
        x,y=np.rint(records[0]['ink']*99).astype(int).T
        self.assertTrue((gray[y,x]==85).all())
        for invalid in (0,6,2.5):
            with self.assertRaises(ValueError):ink_observations(gray,[word],localization_halo=invalid)
    def test_request_uses_exact_pixels_and_matching_atomic_response(self):
        import json
        gray=np.arange(200,dtype=np.uint8).reshape(10,20)
        with tempfile.TemporaryDirectory() as folder:
            def respond(event):
                raster=Path(folder)/event['raster']
                np.testing.assert_array_equal(cv2.imread(str(raster),0),gray)
                response=Path(folder)/event['response']
                temporary=response.with_suffix('.partial')
                temporary.write_text(json.dumps([WORD]))
                temporary.replace(response)
            self.assertEqual(request_words(gray,folder,emit=respond),[WORD])
            self.assertEqual(list(Path(folder).iterdir()),[])

    def test_timeout_and_invalid_response_supply_no_evidence(self):
        with tempfile.TemporaryDirectory() as folder:
            gray=np.zeros((10,20),np.uint8)
            self.assertEqual(request_words(gray,folder,timeout=.01,emit=lambda _:None),[])
            def invalid(event):(Path(folder)/event['response']).write_text('{partial')
            self.assertEqual(request_words(gray,folder,emit=invalid),[])

    def test_folded_or_outside_native_quads_cannot_be_evidence(self):
        folded={**WORD,'points':[[.1,.1],[.5,.3],[.5,.1],[.1,.3]]}
        outside={**WORD,'points':[[.1,.1],[1.1,.1],[1.1,.3],[.1,.3]]}
        self.assertEqual(validated_words([folded,outside,{'text':'bad'},WORD]),[WORD])
        self.assertEqual(validated_words({'words':[WORD]}),[])

    def test_ink_uses_the_mesh_pixel_basis_and_safe_stock(self):
        gray=np.full((40,80),230,np.uint8)
        gray[7:10,12:29]=80
        safe=np.full_like(gray,255);safe[7:10,12:18]=0
        group=ink_groups(gray,[WORD],safe)[0]
        x,y=np.rint(group*[79,39]).astype(int).T
        self.assertTrue((safe[y,x]>250).all())
        self.assertTrue((gray[y,x]==80).all())
        self.assertFalse((x<18).any())

    def test_native_pixel_centers_convert_to_actual_measurement_nodes(self):
        word={**WORD,'points':[[.5/80,.5/40],[79.5/80,.5/40],
                              [79.5/80,39.5/40],[.5/80,39.5/40]]}
        actual=measurement_words([word],80,40)[0]
        np.testing.assert_allclose(actual['points'],[[0,0],[1,0],[1,1],[0,1]],atol=1e-15)
        self.assertEqual(measurement_words([word],1,40),[])

    def test_crop_adds_missed_word_in_global_pixel_coordinates(self):
        result=merge_crop_words([],[WORD],[100,200,300,400],800,1000)
        expected=(np.asarray(WORD['points'])*[200,200]+[100,200])/[800,1000]
        np.testing.assert_array_equal(result[0]['points'],expected)
        duplicate=merge_crop_words(result,[WORD],[100,200,300,400],800,1000)
        self.assertEqual(duplicate,result)

    def test_clipped_crop_word_cannot_redefine_ownership(self):
        clipped={**WORD,'points':[[0,.1],[.5,.1],[.5,.3],[0,.3]]}
        self.assertEqual(merge_crop_words([WORD],[clipped],[0,0,200,200],800,1000),[WORD])
        with self.assertRaises(ValueError):merge_crop_words([],[],[-1,0,200,200],800,1000)

    def test_crop_requests_keep_source_pixels_unmodified(self):
        gray=np.arange(300*500,dtype=np.uint8).reshape(300,500)
        crops=[]
        def provider(image,folder,**kwargs):crops.append(image.copy());return []
        self.assertEqual(request_words_with_crops(gray,'unused',provider=provider),[])
        self.assertEqual(len(crops),5)
        np.testing.assert_array_equal(crops[0],gray)
        padding=round(48*500/1280)
        np.testing.assert_array_equal(crops[1],gray[:150+padding,:250+padding])

    def test_agreeing_crop_recovers_faint_ink_without_replacing_the_owner_quad(self):
        original={**WORD,'points':[[.2,.2],[.5,.2],[.5,.4],[.2,.4]]}
        enlarged={**WORD,'points':[[.15,.18],[.55,.18],[.55,.42],[.15,.42]]}
        merged=merge_crop_words([original],[enlarged],[0,0,800,1000],800,1000)
        self.assertEqual(merged[0]['points'],original['points'])
        self.assertEqual(merged[0]['ink_envelopes'],[enlarged['points']])
        self.assertEqual(validated_words(merged),merged)
        gray=np.full((100,80),230,np.uint8);gray[22:25,12:16]=50
        self.assertEqual(ink_groups(gray,[original]),[])
        self.assertTrue(ink_groups(gray,merged))

    def test_crop_enrichment_cannot_cross_another_independent_word(self):
        original={**WORD,'points':[[.2,.2],[.5,.2],[.5,.4],[.2,.4]]}
        adjacent={**WORD,'points':[[.52,.2],[.8,.2],[.8,.4],[.52,.4]]}
        enlarged={**WORD,'points':[[.15,.18],[.55,.18],[.55,.42],[.15,.42]]}
        merged=merge_crop_words([original,adjacent],[enlarged],[0,0,800,1000],800,1000)
        self.assertEqual(merged,[original,adjacent])


if __name__=='__main__':unittest.main()
