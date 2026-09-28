import unittest
import numpy as np
from ordered_constraint_solver import ordered_shifts, infeasible_cycle


class OrderedConstraintTests(unittest.TestCase):
    def test_refusal_witness_contains_only_real_edges_and_closes(self):
        bounds={(-1,0):(0,0),(0,1):(1,2),(-1,1):(0,0)}
        cycle=infeasible_cycle(bounds,2)
        self.assertLess(sum(q['maximum_relative_shift'] for q in cycle),0)
        for edge,following in zip(cycle,cycle[1:]+cycle[:1]):
            self.assertEqual(edge['to_owner'],following['from_owner'])
            a,b=edge['from_owner'],edge['to_owner']
            expected=bounds[(a,b)][1] if a<b else -bounds[(b,a)][0]
            self.assertEqual(edge['maximum_relative_shift'],expected)
        self.assertEqual(infeasible_cycle({(-1,0):(0,0),(0,1):(1,2)},2),[])

    def test_coupled_rigid_words_leave_an_unrelated_word_stationary(self):
        bounds={(-1,0):(0,0),(0,1):(1,2),(1,2):(0,0),(-1,2):(-2,2),(-1,3):(-2,2)}
        np.testing.assert_allclose(ordered_shifts(bounds,4,[3,2,1,0]),[0,1,1,0])

    def test_detects_an_infeasible_cycle_between_three_words(self):
        bounds={(-1,0):(-5,5),(-1,1):(-5,5),(-1,2):(-5,5),
                (0,1):(1,2),(1,2):(1,2),(0,2):(-1,0)}
        self.assertIsNone(ordered_shifts(bounds,3))

    def test_preserves_the_preferred_word_at_zero_when_a_neighbor_must_move(self):
        bounds={(-1,0):(-5,5),(-1,1):(-5,5),(0,1):(2,5)}
        np.testing.assert_allclose(ordered_shifts(bounds,2,[0,1]),[0,2])
        np.testing.assert_allclose(ordered_shifts(bounds,2,[1,0]),[-2,0])


if __name__=='__main__':
    unittest.main()
