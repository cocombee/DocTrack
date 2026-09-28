import unittest
import numpy as np
from independent_word_owners import word_groups


def box(x0,y0,x1,y1):
    return [[x0,y0],[x1,y0],[x1,y1],[x0,y1],[(x0+x1)/2,y0],[(x0+x1)/2,y1]]


class IndependentWordTests(unittest.TestCase):
    def test_nearby_words_keep_separate_owners_and_own_letters(self):
        boxes=np.asarray([box(.21,.30,.22,.31),box(.25,.30,.26,.31),
                          box(.20,.299,.23,.312),box(.24,.299,.27,.312)])
        groups=word_groups(boxes,2)
        self.assertEqual(groups,[[2,0],[3,1]])

    def test_unobserved_ink_is_still_owned_and_not_added_to_an_unrelated_word(self):
        boxes=np.asarray([box(.8,.6,.82,.62),box(.2,.3,.25,.32)])
        self.assertEqual(word_groups(boxes,1),[[1],[0]])

    def test_vertical_pass_groups_unobserved_letters_in_the_reading_direction(self):
        boxes=np.asarray([box(.10,.30,.11,.31),box(.125,.30,.135,.31),
                          box(.10,.35,.11,.36),box(.125,.35,.135,.36)])
        expected=word_groups(boxes,4)
        self.assertEqual(expected,[[0,1],[2,3]])
        self.assertEqual(word_groups(boxes[:,:,::-1],4,reading_axis=1),expected)
        self.assertNotEqual(word_groups(boxes[:,:,::-1],4),expected)


if __name__=='__main__':
    unittest.main()
