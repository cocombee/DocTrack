"""Fit rounded corners only where the image supports an arc over a sharp corner."""
from pathlib import Path
import json,numpy as np,cv2
from PIL import Image,ImageOps
r=Path(__file__).resolve().parents[2];out=r/'Scanner Integration/Predictions Rounded V4';out.mkdir(exist_ok=True)
for f in sorted((r/'Data/originals').rglob('*.jpg')):
 d=json.loads((r/'Scanner Integration/Predictions Curved V9'/(f.name+'.json')).read_text());im=np.asarray(ImageOps.exif_transpose(Image.open(f)).convert('RGB')).astype(np.float32);h,w=im.shape[:2];q=np.array([[p['x']*w,p['y']*h] for p in d['corners']]);p=np.array([[v['x']*w,v['y']*h] for v in d['outerBoundary']]);roundings=[]
 def contrast(xy,normal):
  get=lambda v:cv2.remap(im,v[:,0].astype(np.float32),v[:,1].astype(np.float32),cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE).reshape(-1,3)
  inside=get(xy+normal*4);outside=get(xy-normal*4);skin=lambda c:(c[:,0]>c[:,1]+15)&(c[:,1]>c[:,2]+10)
  if np.mean(skin(inside)&skin(outside))>.2:return None
  return np.minimum(np.linalg.norm(inside-outside,axis=1)/np.sqrt(3),90).mean()
 for corner in q:
  i=int(np.argmin(np.linalg.norm(p-corner,axis=1)));best=.12;chosen=None
  for span in range(1,13):
   before=(i-span)%len(p);after=(i+span)%len(p);a=p[before];b=p[after];t=np.linspace(.05,.95,24)[:,None];xy=(1-t)**2*a+2*(1-t)*t*corner+t*t*b;v=2*(1-t)*(corner-a)+2*t*(b-corner);normal=np.column_stack([-v[:,1],v[:,0]])/np.linalg.norm(v,axis=1)[:,None]
   c=contrast(xy,normal)
   half=t<.5;line=np.where(half,a+(corner-a)*2*t,corner+(b-corner)*(2*t-1));tangent=np.where(half,corner-a,b-corner);n=np.column_stack([-tangent[:,1],tangent[:,0]])/np.linalg.norm(tangent,axis=1)[:,None];base=contrast(line,n)
   if c is None or base is None:continue
   gain=(c-base)/max(c,base,5)-.002*(np.linalg.norm(a-corner)+np.linalg.norm(b-corner))
   if gain>best:best=gain;chosen=(i,span,a,b,corner)
  if chosen:roundings.append(chosen)
 replacements={};skip=set()
 for i,span,a,b,c in roundings:
  start=(i-span)%len(p);end=(i+span)%len(p);t=np.linspace(0,1,2*span+2,endpoint=False)[:,None];replacements[start]=(1-t)**2*a+2*(1-t)*t*c+t*t*b
  skip.update((start+k)%len(p) for k in range(2*span))
 result=[]
 for i,v in enumerate(p):
  if i in replacements:result.extend(replacements[i])
  elif i not in skip:result.append(v)
 d['outerBoundary']=[{'x':float(x/w),'y':float(y/h),'visibility':'inferred','confidence':.5} for x,y in result];d['roundedCornerCount']=len(roundings);d['modelVersion']='rounded-corner-evaluation';(out/(f.name+'.json')).write_text(json.dumps(d))
print('Rounded corners',sum(json.loads(p.read_text()).get('roundedCornerCount',0) for p in out.glob('*.jpg.json')))
