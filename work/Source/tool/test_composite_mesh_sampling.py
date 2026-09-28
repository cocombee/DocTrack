import unittest
import numpy as np
from local_receipt_mesh import sample
from local_mesh_storage import compact
from composite_mesh_sampling import MeshSampler,transpose_mesh


class CompositeMeshSamplingTests(unittest.TestCase):
    def test_unrecognized_print_keeps_paired_material_under_axis_transpose(self):
        mesh={'rows':2,'columns':2,'grid':[[0,0],[1,0],[0,1],[1,1]],
              'unrecognized_print_clouds':[[[.21,.31],[.22,.42]]],
              'unrecognized_print_references':[[[.2,.3],[.2,.4]]]}
        swapped=transpose_mesh(mesh)
        self.assertEqual(swapped['unrecognized_print_clouds'],[[[.31,.21],[.42,.22]]])
        self.assertEqual(swapped['unrecognized_print_references'],[[[.3,.2],[.4,.2]]])
        self.assertEqual(transpose_mesh(swapped),mesh)

    def test_transpose_keeps_the_complete_tangent_rule_role_ledger(self):
        role={'normal_axis':0,'component_indices':[2,3,4],
              'points':[[.2,.3],[.2,.4]],'source_points':[[.21,.31],[.22,.42]]}
        ledger={'roles':[role],'earlier_roles':{'roles':[role],'earlier_roles':None}}
        mesh={'rows':2,'columns':2,'grid':[[0,0],[1,0],[0,1],[1,1]],
              'tangent_free_rule_proxy_roles':ledger}
        swapped=transpose_mesh(mesh)
        self.assertEqual(swapped['tangent_free_rule_proxy_roles']['roles'][0]['normal_axis'],1)
        self.assertEqual(swapped['tangent_free_rule_proxy_roles']['roles'][0]['points'],[[.3,.2],[.4,.2]])
        self.assertEqual(transpose_mesh(swapped),mesh)
        self.assertEqual(mesh['tangent_free_rule_proxy_roles'],ledger)

    def test_transpose_preserves_windowed_composition_and_rule_coordinates(self):
        x,y=np.meshgrid(np.arange(17)/16,np.arange(33)/32)
        field=np.stack([x+.03*x*y,y],axis=2)
        parent={'rows':33,'columns':17,'grid':field.reshape(-1,2).tolist()}
        query=np.stack([x,y],axis=2);query[8:17,4:9,1]+=.005
        mesh=compact({'rows':33,'columns':17,'grid':query.reshape(-1,2).tolist(),
                      'coordinate_transform':True,'preserved_mesh':parent,
                      'refinement_regions':[[.25,.25,.5,.5]],
                      'established_rulers':[{'normal_axis':0,'points':[[.4,.2],[.4,.6]]}],
                      'established_text_rows':[{'points':[[.2,.3],[.4,.3]]}]})
        transposed=transpose_mesh(mesh)
        points=np.array([[.27,.32],[.44,.49],[.24,.3],[.8,.6]])
        np.testing.assert_allclose(MeshSampler(transposed).sample(points[:,::-1])[:,::-1],
                                   MeshSampler(mesh).sample(points),atol=1e-15,rtol=0)
        self.assertEqual(transposed['established_rulers'][0]['normal_axis'],1)
        self.assertEqual(transpose_mesh(transposed),mesh)

    def test_coordinate_warp_composes_without_resampling_its_parent(self):
        x,y=np.meshgrid(np.arange(17)/16,np.arange(33)/32)
        old=np.stack([x,y+.01*x*y],-1)
        parent={'rows':33,'columns':17,'grid':old.reshape(-1,2).tolist()}
        query=np.stack([x,y],-1);query[8:17,4:9,1]+=.005
        mesh={'rows':33,'columns':17,'grid':query.reshape(-1,2).tolist(),
              'coordinate_transform':True,'preserved_mesh':parent,
              'refinement_regions':[[.25,.25,.5,.5]]}
        points=np.array([[.27,.32],[.44,.49],[.24,.3],[.8,.6]])
        expected=sample(old,points)
        expected[:2]=sample(old,sample(query,points[:2]))
        sampler=MeshSampler(compact(mesh))
        np.testing.assert_allclose(sampler.sample(points),expected,atol=1e-15,rtol=0)
        unchanged={**mesh,'grid':np.stack([x,y],-1).reshape(-1,2).tolist()}
        np.testing.assert_array_equal(MeshSampler(unchanged).sample(points),sample(old,points))

    def test_recursive_storage_compacts_a_parent_of_an_already_compacted_child(self):
        x,y=np.meshgrid(np.arange(17)/16,np.arange(33)/32)
        grid=np.stack([x,y],-1).reshape(-1,2).tolist()
        initial={'rows':33,'columns':17,'grid':grid}
        parent={**initial,'preserved_mesh':initial,'refinement_regions':[[.25,.25,.5,.5]]}
        child=compact({**initial,'preserved_mesh':parent,'refinement_regions':[[.25,.25,.5,.5]]})
        self.assertNotIn('grid_window',child['preserved_mesh'])
        result=compact(child,recursive=True)
        self.assertIn('grid_window',result['preserved_mesh'])
        points=np.array([[.27,.32],[.44,.49],[.24,.3],[.8,.6]])
        np.testing.assert_array_equal(MeshSampler(result).sample(points),MeshSampler(child).sample(points))

    def test_coordinate_transform_requires_a_parent_and_boolean_flag(self):
        grid={'rows':2,'columns':2,'grid':[[0,0],[1,0],[0,1],[1,1]]}
        with self.assertRaises(ValueError):MeshSampler({**grid,'coordinate_transform':True})
        with self.assertRaises(ValueError):MeshSampler({**grid,'coordinate_transform':'true'})
    def test_local_window_and_retained_parent_keep_original_indices(self):
        x,y=np.meshgrid(np.linspace(0,1,17),np.linspace(0,1,33))
        old=np.stack([x,y+.01*x*y],-1)
        parent={'rows':33,'columns':17,'grid':old.reshape(-1,2).tolist()}
        changed=old.copy();changed[8:17,4:9,1]+=.005
        region=[.25,.25,.5,.5]
        full={'rows':33,'columns':17,'grid':changed.reshape(-1,2).tolist(),
              'preserved_mesh':parent,'refinement_regions':[region]}
        sampler=MeshSampler(compact(full))
        queries=np.array([[.27,.32],[.44,.49],[.24,.3],[.5,.5],[0,0],[1,1]])
        inside=(queries[:,0]>=.25)&(queries[:,0]<=.5)&(queries[:,1]>=.25)&(queries[:,1]<=.5)
        expected=sample(old,queries);expected[inside]=sample(changed,queries[inside])
        np.testing.assert_allclose(sampler.sample(queries),expected,atol=1e-15)
        dense=sampler.dense(65,33)
        np.testing.assert_allclose(dense[20,10],sampler.sample(np.array([[10/32,20/64]]))[0],atol=1e-15)

    def test_outside_a_new_patch_uses_the_exact_older_layer(self):
        x,y=np.meshgrid(np.linspace(0,1,9),np.linspace(0,1,9));old=np.stack([x,y],-1)
        parent={'rows':9,'columns':9,'grid':old.reshape(-1,2).tolist()}
        corrupted=old+.02
        mesh={'rows':9,'columns':9,'grid':corrupted.reshape(-1,2).tolist(),
              'preserved_mesh':parent,'refinement_regions':[[.25,.25,.5,.5]]}
        queries=np.array([[.1,.3],[.8,.6]])
        np.testing.assert_array_equal(MeshSampler(mesh).sample(queries),sample(old,queries))


if __name__=='__main__':unittest.main()
