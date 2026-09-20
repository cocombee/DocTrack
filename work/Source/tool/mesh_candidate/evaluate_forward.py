from pathlib import Path
import sys,json,cv2,numpy as np
import mesh_ruler
from forward_ruler import forward_ruler_mesh
def proposal(horizontal,vertical=(),prior=None,n=33,prior_weight=.045):
 g=forward_ruler_mesh(horizontal,vertical,prior,n,prior_weight=.005)
 dx=np.diff(g,axis=1);dy=np.diff(g,axis=0)
 jac=[(a[:,:,0]*b[:,:,1]-a[:,:,1]*b[:,:,0])*(n-1)**2 for a in (dx[:-1],dx[1:]) for b in (dy[:,:-1],dy[:,1:])]
 print('proposal',float(g.min()),float(g.max()),float(min(j.min() for j in jac)),flush=True)
 np.save(out/f'Image {i} Forward.npy',g)
 return g
mesh_ruler.ruler_mesh=proposal
import dewarp_mesh
root=Path(__file__).resolve().parents[3]/'Scanner Integration';out=root/'Relative Ruler Probe 49';out.mkdir(exist_ok=True)
for i in [2,4,17,25,35,38,46]:
 p=root/'Full Resolution Candidate 49'/f'Image {i} Preliminary.png'
 result=dewarp_mesh.solve(p);(out/f'Image {i} Mesh.json').write_text(json.dumps(result))
 a=cv2.imread(str(p),-1);h,w=a.shape[:2];g=np.array(result['grid']).reshape(33,33,2).astype(np.float32)
 xx,yy=np.meshgrid(np.linspace(0,32,w),np.linspace(0,32,h));full=cv2.remap(g,xx.astype(np.float32),yy.astype(np.float32),cv2.INTER_LINEAR)
 image=cv2.remap(a,(full[:,:,0]*(w-1)).astype(np.float32),(full[:,:,1]*(h-1)).astype(np.float32),cv2.INTER_LINEAR)
 cv2.imwrite(str(out/f'Image {i}.png'),image)
 print(i,result['method'],result.get('baseline_error'),result.get('selected_error'),flush=True)
