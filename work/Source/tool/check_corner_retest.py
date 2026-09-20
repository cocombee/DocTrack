from pathlib import Path
import json,hashlib,numpy as np
from PIL import Image,ImageDraw
r=Path('..');s=r/'Scanner Integration';out=s/'Corner Retest Build 22';m=json.loads((s/'Reference Baseline/manifest.json').read_text());rows=[];sheets=[]
for i,e in enumerate(m,1):
 name=e['name']+'.json';d=json.loads((out/'Inset Geometry'/name).read_text());p=np.array([[v['x']*d['width'],v['y']*d['height']] for v in d['points']]);a=p;b=np.roll(p,-1,axis=0);v=b-a;crossings=0
 def cross(x,y):return x[...,0]*y[...,1]-x[...,1]*y[...,0]
 for j in range(len(p)-2):
  k=np.arange(j+2,len(p));k=k[k!=len(p)-1] if j==0 else k
  if len(k)==0:continue
  c=a[k];dd=b[k];z=dd-c
  hit=(cross(v[j],c-a[j])*cross(v[j],dd-a[j]) < -1e-9)&(cross(z,a[j]-c)*cross(z,b[j]-c)<-1e-9)
  crossings+=int(hit.sum())
 old=json.loads((s/'Predictions Dart ONNX V14'/name).read_text());new=json.loads((s/'Predictions Corner V15 All'/name).read_text());rows.append({'index':i,'crossings':crossings,'physical_geometry_unchanged':old['boundarySegments']==new['boundarySegments'],'inset_anchors':len(d['segments'])})
 im=Image.open(r/'Data'/e['image']).convert('RGB');im.thumbnail((310,320));dr=ImageDraw.Draw(im);q=p/np.array([d['width'],d['height']])*np.array(im.size);dr.line([tuple(v) for v in q]+[tuple(q[0])],fill=(0,185,255),width=1);sheets.append((i,im,crossings))
for start in range(0,50,10):
 im=Image.new('RGB',(1600,720),(22,22,26));dr=ImageDraw.Draw(im)
 for k,(i,part,crossings) in enumerate(sheets[start:start+10]):
  x=k%5*320;y=k//5*360;im.paste(part,(x+(320-part.width)//2,y+30));dr.text((x+5,y+6),f'{i} crossings:{crossings}',fill='white')
 im.save(out/f'Inset Review {start+1}.jpg')
checks={};changed=[p for p,h in checks.items() if hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h]
report={'count':len(rows),'unchanged_reference_files':len(checks)-len(changed),'changed_reference_files':changed,'crossing_images':[x['index'] for x in rows if x['crossings']],'physical_geometry_unchanged_count':sum(v['physical_geometry_unchanged'] for v in rows),'images':rows}
(out/'Geometry Audit.json').write_text(json.dumps(report,indent=2));print({k:v for k,v in report.items() if k!='images'})
