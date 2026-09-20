from pathlib import Path
from PIL import Image,ImageDraw
import json,numpy as np,cv2
from boundary_regression import sample_boundary, reference_segments, errors
r=Path('..');m=json.loads((r/'Data/manifest.json').read_text());p=r/'Scanner Integration/Physical Review V18';p.mkdir(exist_ok=True);rows=[];panels=[]
for n,e in enumerate(m,1):
 d=json.loads((p/'Inset Geometry'/(e['name']+'.json')).read_text());ref=json.loads((r/'Data'/e['label']).read_text())['fullPolygon'];w,h=d['width'],d['height'];a=[]
 for i,v in enumerate(ref):
  z=ref[(i+1)%len(ref)];A=np.array([v['x'],v['y']]);B=np.array(v['hOut']) if v.get('smooth') else A;D=np.array([z['x'],z['y']]);C=np.array(z['hIn']) if z.get('smooth') else D;t=np.linspace(0,1,80,endpoint=False)[:,None];a.extend(((1-t)**3*A+3*(1-t)**2*t*B+3*(1-t)*t*t*C+t**3*D)*[w,h])
 a=sample_boundary(reference_segments(ref),w,h);b=sample_boundary(d['segments'],w,h);rows.append({'index':n,**errors(b,a),'anchors':len(d['segments'])})
 im=Image.open(r/'Data'/e['image']).convert('RGB');dr=ImageDraw.Draw(im);dr.line([tuple(v) for v in a]+[tuple(a[0])],fill=(0,240,70),width=max(1,round(w/800)));dr.line([tuple(v) for v in b]+[tuple(b[0])],fill=(0,170,255),width=max(1,round(w/800)))
 x0,y0=np.minimum(a.min(0),b.min(0))-15;x1,y1=np.maximum(a.max(0),b.max(0))+15;im=im.crop((max(0,x0),max(0,y0),min(w,x1),min(h,y1)));im.thumbnail((380,430));im.save(p/f'Comparison {n}.jpg');panels.append(im)
for start in range(0,50,10):
 im=Image.new('RGB',(2000,960),(22,22,26));dr=ImageDraw.Draw(im)
 for k,part in enumerate(panels[start:start+10]):
  x=k%5*400;y=k//5*480;im.paste(part,(x+(400-part.width)//2,y+30));dr.text((x+8,y+7),f"{start+k+1}: p95 {rows[start+k]['p95_px']:.1f}px",fill='white')
 im.save(p/f'Comparison Sheet {start+1}.jpg')
(p/'Reference Comparison.json').write_text(json.dumps(rows,indent=2));print(sorted(rows,key=lambda r:-r['p95_px'])[:15])
