"""Local gradient line refinement of pixel-only predictions; never reads labels."""
from pathlib import Path
import json,numpy as np,cv2
from PIL import Image,ImageOps
r=Path(__file__).resolve().parents[2];out=r/'Scanner Integration/Predictions Line Refine V1';out.mkdir(exist_ok=True)
for f in sorted((r/'Data/originals').rglob('*.jpg')):
 d=json.loads((r/'Scanner Integration/Predictions Dart ONNX Final'/(f.name+'.json')).read_text());q=np.array([[v['x'],v['y']] for v in d['outerBoundary']]);im=np.asarray(ImageOps.exif_transpose(Image.open(f)).convert('RGB'));h,w=im.shape[:2];scale=min(1,480/max(w,h));sw,sh=round(w*scale),round(h*scale);gray=cv2.cvtColor(cv2.resize(im,(sw,sh)),cv2.COLOR_RGB2GRAY).astype(np.float32);q=q*np.array([sw,sh]);lines=[]
 if len(q)!=4:continue
 for i,a in enumerate(q):
  b=q[(i+1)%4];v=b-a;length=np.linalg.norm(v);normal=np.array([-v[1],v[0]])/length
  clipped=(max(a[0],b[0])<1 or max(a[1],b[1])<1 or min(a[0],b[0])>sw-1 or min(a[1],b[1])>sh-1)
  best=-1e9;chosen=(a,b)
  for o1 in ([0] if clipped else range(-18,19,3)):
   for o2 in ([0] if clipped else range(-18,19,3)):
    aa=a+normal*o1;bb=b+normal*o2;t=np.linspace(.08,.92,32)[:,None];p=aa+(bb-aa)*t
    inside=p+normal*2;outside=p-normal*2
    get=lambda xy:cv2.remap(gray,xy[:,0].astype(np.float32),xy[:,1].astype(np.float32),cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE).ravel()
    delta=np.abs(get(inside)-get(outside));score=np.mean(np.minimum(delta,70))-.06*(o1*o1+o2*o2)
    if score>best:best=score;chosen=(aa,bb)
  lines.append(chosen)
 p=[]
 for i in range(4):
  a,b=lines[i-1];c,e=lines[i];u=b-a;v=e-c;matrix=np.column_stack([u,-v])
  if abs(np.linalg.det(matrix))<1:p.append(q[i]/[sw,sh]);continue
  t=np.linalg.solve(matrix,c-a)[0];point=a+t*u
  p.append(np.clip(point/[sw,sh],0,1))
 d['outerBoundary']=[{'x':float(x),'y':float(y),'visibility':'inferred','confidence':.5} for x,y in p];d['modelVersion']='offline-line-refinement-v1';d['timing_note']='refinement latency not included';(out/(f.name+'.json')).write_text(json.dumps(d))
