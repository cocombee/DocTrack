import unittest
from types import SimpleNamespace
import cv2
import numpy as np
from mesh_ruler import paragraph_rulers, ruler_mesh
from dewarp_mesh import inverse_points, valid

class ParagraphRulerTests(unittest.TestCase):
    def make_column(self, bent):
        image=np.full((800,640),235,np.uint8)
        lines=[]
        for row,y in enumerate(range(100,681,24)):
            x=100+round(13*np.sin((y-100)/580*np.pi)) if bent else 100
            # Merged first words have different widths but the same margin.
            width=6+(row%4)*7
            cv2.rectangle(image,(x,y),(x+width,y+10),20,-1)
            for xx in range(x+45,x+235,18):
                cv2.rectangle(image,(xx,y),(xx+6,y+10),20,-1)
            _,_,stats,centers=cv2.connectedComponentsWithStats((image[y:y+11]<100).astype(np.uint8))
            p=centers[1:].copy();p[:,1]+=y
            p=p[np.argsort(p[:,0])]
            lines.append(SimpleNamespace(centers=p))
        return image,SimpleNamespace(lines=lines)

    def test_variable_word_width_does_not_invent_margin_bend(self):
        image,features=self.make_column(False)
        guides=paragraph_rulers(image,features)
        self.assertGreaterEqual(len(guides),1)
        self.assertLess(max(np.ptp(p[:,0]) for p in guides),1e-7)

    def test_curved_column_supplies_correctable_vertical_evidence(self):
        image,features=self.make_column(True)
        guides=paragraph_rulers(image,features)
        self.assertGreaterEqual(len(guides),1)
        grid=ruler_mesh([],[(p,.6) for p in guides])
        self.assertTrue(valid(grid,edge_extension=True))
        for p in guides:
            self.assertLess(np.std(inverse_points(grid,p)[:,0]),np.std(p[:,0])*.35)

    def test_sparse_caption_does_not_establish_paragraph_margin(self):
        image,features=self.make_column(True)
        features.lines=features.lines[:3]
        self.assertEqual(paragraph_rulers(image,features),[])

if __name__=='__main__':unittest.main()
