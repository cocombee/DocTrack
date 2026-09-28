import unittest
import numpy as np
from composite_mesh_sampling import MeshSampler,transpose_mesh
from column_word_origins import column_word_bindings,bound_column_audit


class ColumnWordOriginTests(unittest.TestCase):
    def fixture(self):
        x,y=np.meshgrid(np.arange(301)/300,np.arange(201)/200)
        grid=np.stack([x,y],axis=2)
        parent={'rows':201,'columns':301,'grid':grid.reshape(-1,2).tolist()}
        bodies=[]
        for center in (.2,.5,.8):
            xx,yy=np.meshgrid(np.linspace(.228,.28,7),np.linspace(center-.01,center+.01,5))
            bodies.append(np.c_[xx.ravel(),yy.ravel()])
        origins=[[.22,center] for center in (.2,.5,.8)]
        record={'normal_axis':0,'points':origins,'reference':origins}
        base={**parent,'preserved_mesh':parent,'refinement_regions':[[0,0,1,1]],
              'established_text_columns':[record],
              'retained_word_clouds':[b.tolist() for b in bodies]}
        return base,record,grid,bodies

    def test_blank_stock_can_stretch_while_the_complete_word_origins_stay_aligned(self):
        base,record,grid,bodies=self.fixture();query=grid.copy()
        bump=np.maximum(0,1-abs(query[:,:,0]-.22)/.006)
        query[:,:,0]-=.0008*bump*np.clip((query[:,:,1]-.6)/.1,0,1)
        result={**base,'grid':query.reshape(-1,2).tolist(),'coordinate_transform':True,
                'preserved_mesh':base}
        binding=column_word_bindings(base,[record],[1000,1000])[0]
        self.assertIsNotNone(binding)
        audit=bound_column_audit(MeshSampler(result),record,binding,[1000,1000])
        self.assertTrue(audit['accepted'],audit)
        self.assertLess(audit['relative_alignment_change_px'],.00001)
        from composite_mesh_inverse import inverse_retained
        material=inverse_retained(MeshSampler(result),np.asarray(record['points']))
        self.assertGreater(np.ptp((material-np.asarray(record['reference']))[:,0])*1000,.5)

    def test_distinct_rigid_word_translations_still_fail_the_original_column_limit(self):
        base,record,grid,bodies=self.fixture();query=grid.copy()
        query[:,:,0]-=.002*np.clip((query[:,:,1]-.6)/.1,0,1)
        result={**base,'grid':query.reshape(-1,2).tolist(),'coordinate_transform':True,
                'preserved_mesh':base}
        binding=column_word_bindings(base,[record],[1000,1000])[0]
        audit=bound_column_audit(MeshSampler(result),record,binding,[1000,1000])
        self.assertFalse(audit['accepted'])
        self.assertAlmostEqual(audit['relative_alignment_change_px'],2)
        self.assertLess(audit['maximum_word_shape_error_px'],.001)

    def test_ambiguity_and_wrong_registration_coordinates_cannot_replace_the_legacy_guard(self):
        base,record,grid,bodies=self.fixture()
        base['retained_word_clouds'].append(bodies[0].tolist())
        self.assertIsNone(column_word_bindings(base,[record],[1000,1000])[0])
        base['retained_word_clouds'].pop()
        wrong={**record,'points':(np.asarray(record['points'])+[.001,0]).tolist()}
        self.assertIsNone(column_word_bindings(base,[wrong],[1000,1000])[0])

    def test_an_unrelated_tiny_cloud_does_not_erase_other_complete_column_evidence(self):
        base,record,grid,bodies=self.fixture()
        base['retained_word_clouds'].append([[.8,.1]])
        self.assertIsNotNone(column_word_bindings(base,[record],[1000,1000])[0])
        base['retained_word_clouds'][0]=bodies[0][:10].tolist()
        self.assertIsNone(column_word_bindings(base,[record],[1000,1000])[0])

    def test_later_registration_does_not_reset_an_earlier_word_origin(self):
        base,record,grid,bodies=self.fixture();query=grid.copy()
        query[:,:,0]-=.002*np.clip((query[:,:,1]-.6)/.1,0,1)
        later={**base,'grid':query.reshape(-1,2).tolist(),'coordinate_transform':True,
               'preserved_mesh':base}
        binding=column_word_bindings(later,[record],[1000,1000])[0]
        audit=bound_column_audit(MeshSampler(later),record,binding,[1000,1000])
        self.assertFalse(audit['accepted'])
        self.assertAlmostEqual(audit['relative_alignment_change_px'],2)

    def test_all_physical_evidence_follows_vertical_transposition(self):
        base,record,grid,bodies=self.fixture()
        transposed=transpose_mesh(base);column=transposed['established_text_columns'][0]
        binding=column_word_bindings(transposed,[column],[1000,1000])[0]
        self.assertIsNotNone(binding)
        self.assertEqual(binding['normal_axis'],1)
        self.assertTrue(bound_column_audit(MeshSampler(transposed),column,binding,[1000,1000])['accepted'])

    def test_wrong_column_and_distorted_material_are_rejected(self):
        base,record,grid,bodies=self.fixture()
        binding=column_word_bindings(base,[record],[1000,1000])[0]
        other={**record,'normal_axis':1}
        with self.assertRaises(ValueError):bound_column_audit(MeshSampler(base),other,binding,[1000,1000])
        query=grid.copy();query[:,:,0]*=1.02
        result={**base,'grid':query.reshape(-1,2).tolist()}
        self.assertFalse(bound_column_audit(MeshSampler(result),record,binding,[1000,1000])['accepted'])

    def test_incomplete_or_changed_binding_cannot_move_the_reference_origin(self):
        import copy
        base,record,grid,bodies=self.fixture()
        binding=column_word_bindings(base,[record],[1000,1000])[0]
        changed=copy.deepcopy(binding)
        changed['words'][0]['reference_origin'][0]+=.001
        with self.assertRaises(ValueError):
            bound_column_audit(MeshSampler(base),record,changed,[1000,1000])
        binding['words'].pop()
        with self.assertRaises(ValueError):
            bound_column_audit(MeshSampler(base),record,binding,[1000,1000])

    def test_column_dispatch_keeps_legacy_default_and_explicit_ambiguous_fallback(self):
        from progressive_content_mesh import column_geometry_audit
        base,record,grid,bodies=self.fixture();query=grid.copy()
        bump=np.maximum(0,1-abs(query[:,:,0]-.22)/.006)
        query[:,:,0]-=.0008*bump*np.clip((query[:,:,1]-.6)/.1,0,1)
        result={**base,'grid':query.reshape(-1,2).tolist(),'coordinate_transform':True,
                'preserved_mesh':base}
        bindings=column_word_bindings(base,[record],[1000,1000])
        legacy=column_geometry_audit(result,[record],[1000,1000])
        self.assertFalse(legacy[0]['accepted'])
        self.assertEqual(legacy,column_geometry_audit(result,[record],[1000,1000],word_bindings=[None]))
        self.assertTrue(column_geometry_audit(result,[record],[1000,1000],word_bindings=bindings)[0]['accepted'])
        with self.assertRaises(ValueError):
            column_geometry_audit(result,[record],[1000,1000],word_bindings=[])


if __name__=='__main__':unittest.main()
