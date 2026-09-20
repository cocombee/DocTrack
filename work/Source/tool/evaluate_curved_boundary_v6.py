"""Image-only strip search for smooth physical boundaries around neural proposals."""
from pathlib import Path
import json,numpy as np,cv2
from PIL import Image,ImageOps
r=Path(__file__).resolve().parents[2];out=r/'Scanner Integration/Predictions Curved V6';out.mkdir(exist_ok=True)
for f in sorted((r/'Data/originals').rglob('*.jpg')):
 d=json.loads((r/'Scanner Integration/Predictions Dart ONNX V6'/(f.name+'.json')).read_text());im=np.asarray(ImageOps.exif_transpose(Image.open(f)).convert('RGB')).astype(np.float32);h,w=im.shape[:2];p=np.array([[v['x']*w,v['y']*h] for v in d['outerBoundary']]);result=[]
 for i,a in enumerate(p):
  b=p[(i+1)%len(p)];v=b-a;length=np.linalg.norm(v);n=np.array([-v[1],v[0]])/max(1,length);steps=max(3,int(length/5));radius=min(100,max(6,min(np.linalg.norm(np.roll(p,-1,axis=0)-p,axis=1))*.08));offsets=np.linspace(-radius,radius,49);xy=a+(b-a)*np.linspace(0,1,steps+1)[:,None];strip=xy[:,None,:]+offsets[None,:,None]*n
  sample=lambda delta:cv2.remap(im,(strip[:,:,0]+n[0]*delta).astype(np.float32),(strip[:,:,1]+n[1]*delta).astype(np.float32),cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
  inside=sample(4);outside=sample(-4);contrast=np.linalg.norm(inside-outside,axis=2)/np.sqrt(3);inside_far=sample(10);outside_far=sample(-10)
  # Thin internal marks have little sustained paper/background contrast.
  sustained=np.linalg.norm(inside_far-outside_far,axis=2)/np.sqrt(3);evidence=np.minimum(np.minimum(contrast,sustained),np.linalg.norm(sample(16)-sample(-16),axis=2)/np.sqrt(3))
  skin=lambda c:(c[:,:,0]>c[:,:,1]+15)&(c[:,:,1]>c[:,:,2]+10)
  occluded=(skin(inside_far)|skin(outside_far))[:,24]
  cost=-evidence/12+3*(offsets[None,:]/radius)**2;cost=np.broadcast_to(cost,(steps+1,49)).copy();cost[occluded]=1e12;cost[occluded,24]=0;prev=np.full(49,1e12);prev[24]=0;back=np.zeros((steps+1,49),np.int32)
  for t in range(1,steps+1):
   current=np.full(49,1e12)
   for state in range(49):
    if t==steps and state!=24:continue
    lo=max(0,state-3);hi=min(49,state+4);choices=prev[lo:hi]+.8*(np.arange(lo,hi)-state)**2;k=int(np.argmin(choices));current[state]=choices[k]+cost[t,state];back[t,state]=lo+k
   prev=current
  states=np.zeros(steps+1,np.int32);states[-1]=24
  for t in range(steps,0,-1):states[t-1]=back[t,states[t]]
  shifts=offsets[states];shifts[1:-1]=(shifts[:-2]+2*shifts[1:-1]+shifts[2:])/4
  result.extend((xy[:-1]+shifts[:-1,None]*n)/[w,h])
 d['outerBoundary']=[{'x':float(np.clip(x,0,1)),'y':float(np.clip(y,0,1)),'visibility':'inferred','confidence':.5} for x,y in result];d['modelVersion']='curved-strip-evaluation';d['timing_note']='curve refinement time not included';(out/(f.name+'.json')).write_text(json.dumps(d))
