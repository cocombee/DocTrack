import unittest
import numpy as np

from repeated_rule_mask import recover,whole_ruling_guides,restore_measured_ruling_tails
from printed_rule_sections import measure_sections


class RepeatedRuleMaskTests(unittest.TestCase):
    def fixture(self,*,ink=45,count=42):
        gray=np.full((400,800),240,np.uint8)
        positions=np.arange(60,60+17*count,17)
        for x in positions:
            gray[198:202,x:x+14]=ink
        safe=np.full_like(gray,255)
        heuristic=np.zeros_like(gray,np.uint8)
        for x in positions[positions<400]:heuristic[197:203,x-1:x+15]=1
        protected=np.zeros_like(heuristic)
        observed=np.flatnonzero(gray[200]<240)
        observed=observed[observed>=400]
        guide=np.c_[observed/799,np.full(len(observed),199.5/399)]
        return gray,safe,heuristic,protected,[guide]

    def test_two_profiles_recover_dashes_and_measure_real_previously_hidden_tail(self):
        gray,safe,heuristic,protected,guides=self.fixture()
        original=gray.copy()
        mask,evidence=recover(gray,safe,heuristic,protected,guides)
        self.assertTrue(evidence)
        self.assertFalse(mask[199,80])
        measured=measure_sections(gray,safe,word_mask=mask)
        self.assertTrue(any(g[0,0]*799<70 and g[-1,0]*799>740 for g in measured))
        np.testing.assert_array_equal(gray,original)
        self.assertTrue(all(not q['protected_material_changed'] and not q['ink_pixels_generated']
                            for q in evidence))

    def test_actual_native_or_prior_print_and_unsafe_material_stay_masked(self):
        gray,safe,heuristic,protected,guides=self.fixture()
        protected[196:204,78:96]=1
        mask,_=recover(gray,safe,heuristic,protected,guides)
        np.testing.assert_array_equal(mask[protected>0],np.ones(int((protected>0).sum())))
        safe[:]=0
        mask,evidence=recover(gray,safe,heuristic,protected,guides)
        np.testing.assert_array_equal(mask,heuristic|protected)
        self.assertFalse(evidence)

    def test_curved_seed_recovers_real_curved_ends_without_flattening_image(self):
        gray=np.full((400,800),240,np.uint8)
        positions=np.arange(60,760,17)
        def level(x):return 190+.00018*(x-400)**2
        heuristic=np.zeros_like(gray,np.uint8)
        for x in positions:
            for column in range(x,x+14):
                y=round(level(column));gray[y-2:y+2,column]=45
            y=round(level(x+7))
            if x<170 or x>650:heuristic[y-5:y+5,x-1:x+15]=1
        samples=np.arange(180,650)
        guide=np.c_[samples/799,level(samples)/399]
        original=gray.copy();safe=np.full_like(gray,255)
        mask,evidence=recover(gray,safe,heuristic,np.zeros_like(gray),[guide])
        self.assertTrue(evidence)
        for x in (80,730):self.assertFalse(mask[round(level(x)),x])
        np.testing.assert_array_equal(gray,original)
        self.assertTrue(all(q['search_follows_observed_curvature'] for q in evidence))

    def test_curved_recovery_never_clears_unsafe_halo_or_unobserved_far_stock(self):
        gray,safe,heuristic,protected,guides=self.fixture()
        safe[197,:]=0
        mask,evidence=recover(gray,safe,heuristic,protected,guides)
        np.testing.assert_array_equal(mask[197],heuristic[197])
        # A distant lookalike row has no measured continuation support.
        gray[250:254,60:760]=45;heuristic[249:255,59:761]=1
        mask,evidence=recover(gray,safe,heuristic,protected,guides)
        np.testing.assert_array_equal(mask[249:255,59:761],heuristic[249:255,59:761])

    def test_missing_long_seed_weak_single_profile_and_square_punctuation_abstain(self):
        gray,safe,heuristic,protected,guides=self.fixture()
        for seed in ([],[guides[0][:50]]):
            mask,evidence=recover(gray,safe,heuristic,protected,seed)
            np.testing.assert_array_equal(mask,heuristic)
            self.assertFalse(evidence)
        weak,safe,heuristic,protected,guides=self.fixture(ink=230)
        mask,evidence=recover(weak,safe,heuristic,protected,guides)
        self.assertFalse(evidence)
        np.testing.assert_array_equal(mask,heuristic)

        dots=np.full_like(gray,240)
        for x in range(60,760,18):dots[198:202,x:x+4]=45
        mask,evidence=recover(dots,safe,heuristic,protected,guides)
        self.assertFalse(evidence)
        np.testing.assert_array_equal(mask,heuristic)

    def test_whole_ruling_joins_real_dashes_without_filling_printed_spaces(self):
        gray,safe,heuristic,protected,_=self.fixture()
        observed=np.flatnonzero(gray[200]<240)
        parts=[np.c_[x/799,np.full(len(x),199.5/399)] for x in
               (observed[observed<350],observed[observed>370])]
        mask,evidence=recover(gray,safe,heuristic,protected,parts,whole_rulings=True)
        joined,reports=whole_ruling_guides(gray,safe,protected,parts,evidence)
        self.assertEqual(len(joined),1)
        self.assertTrue(reports)
        columns=np.rint(joined[0][:,0]*799).astype(int)
        self.assertTrue(np.all(gray[200,columns]<240))
        for part in parts:
            self.assertTrue(all(any(np.array_equal(q,p) for p in joined[0]) for q in part))
        self.assertFalse(reports[0]['missing_ink_generated'])

    def test_whole_ruling_never_joins_across_protected_or_unsafe_gap(self):
        for blocked in ('protected','unsafe'):
            gray,safe,heuristic,protected,_=self.fixture()
            observed=np.flatnonzero(gray[200]<240)
            parts=[np.c_[x/799,np.full(len(x),199.5/399)] for x in
                   (observed[observed<340],observed[observed>380])]
            if blocked=='protected':protected[190:210,345:378]=1
            else:safe[190:210,345:378]=0
            mask,evidence=recover(gray,safe,heuristic,protected,parts,whole_rulings=True)
            joined,reports=whole_ruling_guides(gray,safe,protected,parts,evidence)
            self.assertEqual(len(joined),2)
            self.assertTrue(all(len(r['original_sections'])==1 for r in reports))

    def tail_fixture(self):
        gray,safe,_,protected,_=self.fixture()
        observed=np.flatnonzero(gray[200]<240)
        reference=np.c_[observed/799,np.full(len(observed),199.5/399)]
        current=reference[(observed>170)&(observed<650)].copy()
        return gray,safe,protected,current,reference

    def test_measured_tails_retain_exact_samples_without_inventing_blank_ink(self):
        gray,safe,protected,current,reference=self.tail_fixture()
        original=gray.copy();old_current=current.copy();old_reference=reference.copy()
        joined,reports=restore_measured_ruling_tails(gray,safe,protected,[current],[reference])
        np.testing.assert_array_equal(joined[0],reference)
        np.testing.assert_array_equal(gray,original)
        np.testing.assert_array_equal(current,old_current)
        np.testing.assert_array_equal(reference,old_reference)
        self.assertEqual(len(reports),1)
        self.assertFalse(reports[0]['ink_or_points_generated'])
        self.assertTrue(reports[0]['current_ink_and_safe_stock_verified'])

    def test_tail_restore_abstains_entirely_on_glyph_unsafe_or_stale_ink(self):
        for obstruction in ('glyph','unsafe','stale'):
            gray,safe,protected,current,reference=self.tail_fixture()
            x,y=np.rint(reference[0]*[799,399]).astype(int)
            if obstruction=='glyph':protected[y,x]=1
            elif obstruction=='unsafe':safe[y,x]=0
            else:gray[y-4:y+5,x-2:x+3]=240
            joined,reports=restore_measured_ruling_tails(gray,safe,protected,[current],[reference])
            np.testing.assert_array_equal(joined[0],current)
            self.assertFalse(reports)

    def test_distant_ambiguous_or_unordered_reference_cannot_restore_tails(self):
        gray,safe,protected,current,reference=self.tail_fixture()
        distant=reference.copy();distant[:,1]+=.02
        for guides,refs in (([current],[distant]),([current,current.copy()],[reference]),
                            ([current],[reference[::-1]]),([current[::-1]],[reference])):
            joined,reports=restore_measured_ruling_tails(gray,safe,protected,guides,refs)
            self.assertFalse(reports)
            for after,before in zip(joined,guides):np.testing.assert_array_equal(after,before)

    def test_measured_tail_cannot_reconnect_across_protected_blank_gap(self):
        for obstruction in ('glyph','unsafe'):
            gray,safe,protected,current,reference=self.tail_fixture()
            # x=75 is real blank stock between the first two printed dashes.
            self.assertEqual(gray[200,75],240)
            if obstruction=='glyph':protected[200,75]=1
            else:safe[200,75]=0
            joined,reports=restore_measured_ruling_tails(gray,safe,protected,[current],[reference])
            np.testing.assert_array_equal(joined[0],current)
            self.assertFalse(reports)

    def test_measured_tail_restore_requires_paired_finite_rasters(self):
        gray,safe,protected,current,reference=self.tail_fixture()
        with self.assertRaises(ValueError):
            restore_measured_ruling_tails(gray,safe[:-1],protected,[current],[reference])
        gray=gray.astype(float);gray[0,0]=np.nan
        with self.assertRaises(ValueError):
            restore_measured_ruling_tails(gray,safe,protected,[current],[reference])


if __name__=='__main__':unittest.main()
