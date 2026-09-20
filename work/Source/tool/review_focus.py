from pathlib import Path
from PIL import Image,ImageDraw
import json,numpy as np,sys
r=Path(__file__).resolve().parents[2];version=sys.argv[1] if len(sys.argv)>1 else 'Dart ONNX V8';m=json.loads((r/'Data/manifest.json').read_text());out=r/'Scanner Integration/Focus Review';out.mkdir(exist_ok=True)
for index in [5,6,7]:
 e=m[index-1];im=Image.open(r/'Data'/e['image']).convert('RGB');w,h=im.size;d=json.loads((r/f'Scanner Integration/Predictions {version}'/(e['name']+'.json')).read_text());q=np.array([[v['x']*w,v['y']*h] for v in d['outerBoundary']]);ref=json.loads((r/'Scanner Integration/Reference Baseline'/e['label']).read_text())['fullPolygon'];curve=[]
 for i,a in enumerate(ref):
  b=ref[(i+1)%len(ref)];A=np.array([a['x']*w,a['y']*h]);D=np.array([b['x']*w,b['y']*h]);B=np.array(a['hOut'])*[w,h] if a.get('smooth') else A;C=np.array(b['hIn'])*[w,h] if b.get('smooth') else D;t=np.linspace(0,1,80,endpoint=False)[:,None];curve.extend((1-t)**3*A+3*(1-t)**2*t*B+3*(1-t)*t*t*C+t**3*D)
 curve=np.array(curve);v=np.roll(q,-1,axis=0)-q;v/=np.maximum(np.linalg.norm(v,axis=1,keepdims=True),1e-9);normal=np.column_stack([-v[:,1],v[:,0]]);prev=np.roll(normal,1,axis=0);inset=q+4*(normal+prev)/np.maximum((1+np.sum(normal*prev,axis=1))[:,None],.1)
 draw=ImageDraw.Draw(im);draw.line([tuple(x) for x in curve]+[tuple(curve[0])],fill=(0,220,60),width=2);draw.line([tuple(x) for x in inset]+[tuple(inset[0])],fill=(0,160,255),width=2)
 lo=np.maximum(0,np.minimum(q.min(0),curve.min(0))-35);hi=np.minimum([w,h],np.maximum(q.max(0),curve.max(0))+35);im=im.crop((*lo.astype(int),*hi.astype(int)));im.save(out/f'Image {index} {version}.png')
