import unittest
import numpy as np
from text_column_alignment import aligned_columns,independent_owner_links,alignment_audit


def quad(x,y,width=.06,height=.014):
    return np.array([[x,y],[x+width,y],[x+width,y+height],[x,y+height]])


class TextColumnAlignmentTests(unittest.TestCase):
    def test_three_or_more_already_aligned_rows_supply_a_relation(self):
        words=[quad(.03,.7),quad(.0301,.72),quad(.0299,.74),quad(.03,.76),quad(.3,.72)]
        columns=aligned_columns(words,[1000,1280])
        self.assertEqual(columns,[[0,1,2,3]])
        self.assertEqual(independent_owner_links(columns,words,words,5),[[0,1,2,3]])

    def test_two_rows_drifting_columns_and_different_font_scales_are_uncertain(self):
        self.assertEqual(aligned_columns([quad(.03,.7),quad(.03,.72)],[1000,1280]),[])
        self.assertEqual(aligned_columns([quad(.03,.7),quad(.0304,.72),quad(.0308,.74)],[1000,1280]),[])
        self.assertEqual(aligned_columns([quad(.03,.7),quad(.03,.72),quad(.03,.75,height=.028)],[1000,1280]),[])

    def test_ambiguous_owner_is_not_silently_linked_to_a_column(self):
        words=[quad(.03,.7),quad(.03,.72),quad(.03,.74)]
        self.assertEqual(independent_owner_links([[0,1,2]],words,[*words,words[0]],4),[])

    def test_shape_preserving_but_different_translations_break_working_column(self):
        points=np.array([[.03,.7],[.03,.72],[.03,.74]])
        self.assertTrue(alignment_audit(points,points+[.005,0],[1000,1280])['accepted'])
        after=points.copy();after[2,0]-=.005
        report=alignment_audit(points,after,[1000,1280])
        self.assertFalse(report['accepted'])
        self.assertAlmostEqual(report['relative_alignment_change_px'],5)


if __name__=='__main__':unittest.main()
