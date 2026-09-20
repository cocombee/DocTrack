"""Offline comparison: model sees original photo, before boundary rectification."""
from pathlib import Path
import json,sys
import cv2,numpy as np,torch
from dewarp_mesh import ROOT
sys.path.insert(0,str(ROOT/'UVDoc'))
from model import UVDocnet

def main():
    torch.set_num_threads(2)
    model=UVDocnet(num_filter=32,kernel_size=5)
    model.load_state_dict(torch.load(ROOT/'UVDoc'/'model'/'best_model.pkl',map_location='cpu',weights_only=True)['model_state']);model.eval()
    root=Path('../Scanner Integration/Original Surface Probe 47');root.mkdir(exist_ok=True)
    manifest=json.loads(Path('../Data/manifest.json').read_text())
    baseline={r['index']:r for r in json.loads(Path('../Scanner Integration/Mesh Regression Baseline 46/Report.json').read_text())}
    for index in [17,25,35,38,46]:
        source=cv2.imread(str(Path('../Data')/manifest[index-1]['image']))
        rgb=cv2.cvtColor(source,cv2.COLOR_BGR2RGB).astype(np.float32)/255
        with torch.inference_mode():grid,_=model(torch.from_numpy(cv2.resize(rgb,(488,712)).transpose(2,0,1)).unsqueeze(0))
        points=(grid[0].permute(1,2,0).numpy()+1)/2
        np.save(root/f'Image {index} Grid.npy',points)
        size=(baseline[index]['width'],baseline[index]['height'])
        mapping=cv2.resize(points,size,interpolation=cv2.INTER_LINEAR)*[source.shape[1]-1,source.shape[0]-1]
        output=cv2.remap(source,mapping[:,:,0].astype(np.float32),mapping[:,:,1].astype(np.float32),cv2.INTER_LINEAR)
        cv2.imwrite(str(root/f'Image {index}.png'),output)
        print(index,'rendered')

if __name__=='__main__':main()
