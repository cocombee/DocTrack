import unittest
import numpy as np
from coordinate_stock import allocate_stock,minimum_cell_jacobian,limited_destination_density


class CoordinateStockTests(unittest.TestCase):
    def test_stock_with_sufficient_capacity_keeps_uniform_spacing(self):
        density=limited_destination_density([.3,1,2],[.1,.2,.1],.4)
        np.testing.assert_array_equal(density,np.ones(3))

    def test_only_low_capacity_stock_is_reallocated_and_shortage_declines(self):
        lengths=np.array([.1,.2,.1]);capacity=np.array([.21,.7,2])
        density=limited_destination_density(capacity,lengths,.8)
        self.assertAlmostEqual(density@lengths,.8)
        self.assertAlmostEqual(density[0],1)
        self.assertAlmostEqual(density[1],density[2])
        self.assertTrue((capacity/density>=.21-1e-12).all())
        with self.assertRaisesRegex(ValueError,'Insufficient actual capacity'):
            limited_destination_density([.21,.21],[.1,.1],.3)

    def test_blank_stock_absorbs_compression_and_ink_band_keeps_translation(self):
        u,v=np.meshgrid(np.arange(17)/16,np.arange(65)/64)
        old=np.stack([u,v+.09*np.sin(np.pi*v)],axis=2)
        knots=np.array([[0,0],[.375,.4],[.625,.65],[1,1]])
        q=np.stack([u,np.interp(v,knots[:,0],knots[:,1])],axis=2)
        result=allocate_stock(old,q,[knots]*17,np.ones(v.shape),.5)
        band=(v>=.375)&(v<=.625)
        np.testing.assert_allclose(result[:,:,1][band],v[band]+.025,atol=1e-15,rtol=0)
        np.testing.assert_array_equal(result[:,:,0],u)
        self.assertGreater(np.max(abs(result-q)),.001)
        self.assertTrue((np.diff(result[:,:,1],axis=0)>0).all())
        np.testing.assert_allclose(result[[0,-1],:,1],v[[0,-1]],atol=1e-15,rtol=0)

    def test_actual_retained_capacity_survives_an_aliased_fold(self):
        from composite_mesh_sampling import MeshSampler
        u,v=np.meshgrid(np.arange(17)/16,np.arange(17)/16)
        source=np.stack([u+10*v*v,v],axis=2)
        base={'rows':17,'columns':17,'grid':source.reshape(-1,2).tolist()}
        u,v=np.meshgrid(np.arange(3)/2,np.arange(3)/2)
        query=np.stack([u,v+.4*u*(1-v)],axis=2)
        parent=MeshSampler({'rows':3,'columns':3,'grid':query.reshape(-1,2).tolist(),
                            'coordinate_transform':True,'preserved_mesh':base,
                            'refinement_regions':[[0,0,1,1]]})
        alias=parent.dense(3,3)
        self.assertLess(minimum_cell_jacobian(alias).min(),0)
        u,v=np.meshgrid(np.arange(17)/16,np.arange(65)/64)
        knots=np.array([[0,0],[.375,.4],[.625,.65],[1,1]])
        proposed=np.stack([u,np.interp(v,knots[:,0],knots[:,1])],axis=2)
        result=allocate_stock(alias,proposed,[knots]*17,np.ones(v.shape),.5,parent=parent)
        band=(v>=.375)&(v<=.625)
        np.testing.assert_allclose(result[:,:,1][band],v[band]+.025,atol=1e-15,rtol=0)
        self.assertTrue((np.diff(result[:,:,1],axis=0)>0).all())

    def test_stronger_reserve_reallocates_blank_stock_without_moving_fixed_ink_endpoints(self):
        from composite_mesh_sampling import MeshSampler
        from normal_composition_guard import validate_normal_composition
        u,v=np.meshgrid(np.arange(9)/8,np.arange(65)/64)
        source_y=np.where(v<=.25,.3*v,.075+v-.25)
        old=np.stack([u,source_y],axis=2)
        parent=MeshSampler({'rows':65,'columns':9,'grid':old.reshape(-1,2).tolist()})
        knots=np.array([[0,0],[.6,.5],[.8,.7],[1,1]])
        proposed=np.stack([u,np.interp(v,knots[:,0],knots[:,1])],axis=2)
        ordinary=allocate_stock(old,proposed,[knots]*9,np.ones(v.shape),.7,parent=parent,capacity_radius=0)
        reserved=allocate_stock(old,proposed,[knots]*9,np.ones(v.shape),.7,parent=parent,capacity_radius=0,minimum_area=.28)
        self.assertGreater(abs(reserved-ordinary).max(),.01)
        band=(v>=.6)&(v<=.8)
        np.testing.assert_allclose(reserved[:,:,1][band],v[band]-.1,atol=1e-15,rtol=0)
        np.testing.assert_array_equal(reserved[:,:,0],u)
        np.testing.assert_allclose(reserved[[0,-1],:,1],v[[0,-1]],atol=1e-15,rtol=0)
        self.assertTrue(validate_normal_composition(parent,reserved,[[0,0,1,1]])['accepted'])
        with self.assertRaises(ValueError):
            allocate_stock(old,proposed,[knots]*9,np.ones(v.shape),.7,parent=parent,minimum_area=.2)

    def test_an_actual_folded_parent_cannot_supply_capacity(self):
        from composite_mesh_sampling import MeshSampler
        folded={'rows':2,'columns':2,'grid':[[0,0],[1,0],[0,1],[-.1,1]],
                'continuous_jacobian_audit':{'accepted':True,'continuous_lower_bound':10}}
        u,v=np.meshgrid(np.arange(5)/4,np.arange(9)/8);alias=np.stack([u,v],axis=2)
        knots=np.array([[0,0],[.375,.4],[.625,.65],[1,1]])
        with self.assertRaisesRegex(ValueError,'Invalid capacity'):
            allocate_stock(alias,alias,[knots]*5,np.ones(v.shape),.5,
                           parent=MeshSampler(folded),allow_unused_alias=True)

    def test_neighbour_span_preserves_area_through_a_diagonal_compressed_strip(self):
        from composite_mesh_sampling import MeshSampler
        from normal_composition_guard import validate_normal_composition
        u,v=np.meshgrid(np.arange(17)/16,np.arange(129)/128)
        old=np.stack([u,v-.78*np.clip(v-.28,0,2/128)],axis=2)
        parent=MeshSampler({'rows':129,'columns':17,'grid':old.reshape(-1,2).tolist()})
        u,v=np.meshgrid(np.arange(17)/16,np.arange(257)/256)
        query=np.stack([u,v],axis=2)
        knots=np.array([[0,0],[.55,.43],[.65,.53],[1,1]])
        columns=[None]*8+[knots]*9
        for col in range(8,17):
            query[:,col,1]=np.interp(v[:,col],knots[:,0],knots[:,1])
        ordinary=allocate_stock(old,query,columns,np.ones(v.shape),.6,parent=parent)
        guarded=allocate_stock(old,query,columns,np.ones(v.shape),.6,parent=parent,
                               include_neighbour_span=True)
        self.assertFalse(validate_normal_composition(parent,ordinary,[[0,0,1,1]])['accepted'])
        self.assertTrue(validate_normal_composition(parent,guarded,[[0,0,1,1]])['accepted'])
        np.testing.assert_array_equal(guarded[:,:8],np.stack([u,v],axis=2)[:,:8])
        band=(v[:,8]>=.55)&(v[:,8]<=.65)
        np.testing.assert_allclose(guarded[band,8:,1],v[band,8:]-.12,atol=1e-15,rtol=0)
        np.testing.assert_array_equal(guarded[:,:,0],u)
        with self.assertRaisesRegex(ValueError,'actual parent'):
            allocate_stock(old,query,columns,np.ones(v.shape),.6,include_neighbour_span=True)

    def test_non_normal_and_crossed_proposals_are_rejected(self):
        u,v=np.meshgrid(np.arange(5)/4,np.arange(9)/8);old=np.stack([u,v],axis=2)
        q=old.copy();q[:,:,0]+=.001
        with self.assertRaises(ValueError):allocate_stock(old,q,[None]*5,np.ones(v.shape),.5)
        knots=np.array([[0,0],[.7,.6],[.6,.7],[1,1]])
        with self.assertRaises(ValueError):allocate_stock(old,old,[knots]*5,np.ones(v.shape),.5)

    def test_capacity_checks_all_four_bilinear_cell_corners(self):
        grid=np.array([[[0,0],[1,0]],[[0,1],[-.1,1]]],dtype=float)
        self.assertLess(minimum_cell_jacobian(grid).min(),0)


if __name__=='__main__':unittest.main()
