"""Read-only diagnostic for background leakage along accepted source edges."""
from pathlib import Path
import json,cv2,numpy as np
root=Path(__file__).resolve().parents[3]
manifest=json.loads((root/'Data/manifest.json').read_text())
for index in (4,17,25,28,35,38,46):
 e=manifest[index-1];im=cv2.imread(str(root/'Data'/e['image']));h,w=im.shape[:2]
 d=json.loads((root/'Scanner Integration/Predictions Physical V18'/f"{e['name']}.json").read_text())
 rows=[]
 for s in d['boundarySegments']:
  if s.get('isImageClip'):continue
  def p(k):return np.array([s[k]['x']*w,s[k]['y']*h])
  a,b=p('start'),p('end');c=p('control1') if s.get('control1') else a;z=p('control2') if s.get('control2') else b
  ts=np.linspace(.02,.98,max(10,round(np.linalg.norm(b-a))))
  q=(1-ts[:,None])**3*a+3*(1-ts[:,None])**2*ts[:,None]*c+3*(1-ts[:,None])*ts[:,None]**2*z+ts[:,None]**3*b
  tangent=np.gradient(q,axis=0);normal=np.column_stack([-tangent[:,1],tangent[:,0]]);normal/=np.linalg.norm(normal,axis=1)[:,None]
  offsets=np.arange(-6,6.01,.25)
  pts=q[:,None,:]+normal[:,None,:]*offsets[None,:,None]
  values=cv2.remap(im,pts[:,:,0].astype('float32'),pts[:,:,1].astype('float32'),cv2.INTER_LINEAR).astype(float)
  outside=values[:,offsets<-3].mean(1);inside=values[:,offsets>3].mean(1)
  stable=(values[:,offsets<-3].std(1).max(1)<12)&(values[:,offsets>3].std(1).max(1)<12)&(np.linalg.norm(inside-outside,axis=1)>75)
  gradient=np.linalg.norm(values[:,2:]-values[:,:-2],axis=2);edge=offsets[1:-1][gradient.argmax(1)]
  rows.append({'stable':int(stable.sum()),'samples':len(ts),'insetMedian':round(float(np.median(edge[stable])),2) if stable.any() else None,'inset90':round(float(np.percentile(edge[stable],90)),2) if stable.any() else None})
 print(index,rows)
