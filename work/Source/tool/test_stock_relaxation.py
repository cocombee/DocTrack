import unittest
import numpy as np
from monotone_stock_relaxation import relax,relax_tangent_stock


class StockRelaxationTests(unittest.TestCase):
    def test_two_axis_stock_keeps_word_rule_and_normal_coordinate(self):
        x,y=np.meshgrid(np.linspace(0,1,101),np.linspace(0,1,151))
        candidate=np.stack([x+.03*np.sin(x*60)*np.sin(y*30),y],-1)
        box=np.asarray([[[.3,.4],[.5,.4],[.5,.5],[.3,.5]]])
        rule=np.c_[np.full(101,.7),np.linspace(.1,.9,101)]
        field=y+.003*np.sin(y*np.pi)
        result=relax_tangent_stock(candidate,field,box,[0],[rule],(.1,.1,.9,.9),8)
        np.testing.assert_array_equal(result[:,:,1],candidate[:,:,1])
        np.testing.assert_array_equal(result[60:76,30:51],candidate[60:76,30:51])
        np.testing.assert_array_equal(result[15:136,69:72],candidate[15:136,69:72])
        self.assertLess(np.linalg.norm(np.diff(result[85:95,15:25,0],n=2,axis=1)),
                        np.linalg.norm(np.diff(candidate[85:95,15:25,0],n=2,axis=1)))

    def test_rigid_words_and_rule_survive_stock_smoothing(self):
        y=np.linspace(0,1,1001)
        knots=[]
        field=np.empty((len(y),81))
        for col in range(81):
            points=[[0,0],[.3,.3],[.4,.4]]
            if 25<=col<=55:
                points.extend([[.44,.45],[.46,.47]])
            points.extend([[.55,.57],[.56,.58],[1,1]])
            points=np.asarray(points)
            knots.append(points)
            field[:,col]=np.interp(y,*points.T)
        result=relax(field,y,knots,8)
        self.assertGreater(np.diff(result,axis=0).min(),0)
        np.testing.assert_allclose(result[300:401],field[300:401],atol=1e-12)
        np.testing.assert_allclose(result[550:561],field[550:561],atol=1e-12)
        np.testing.assert_allclose(result[440:461,25:56],
                                   field[440:461,25:56],atol=1e-12)
        self.assertLess(np.ptp(np.diff(result[480],axis=0)),
                        np.ptp(np.diff(field[480],axis=0)))


if __name__=='__main__':
    unittest.main()
