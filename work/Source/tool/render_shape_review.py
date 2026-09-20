from pathlib import Path
from PIL import Image,ImageDraw
import json, numpy as np
r=Path(__file__).resolve().parents[2];p=r/'Scanner Integration/Predictions Shape V16';m=json.loads((r/'Data/manifest.json').read_text())
for n,e in enumerate(m,1):
 d=json.loads((p/(e['name']+'.json')).read_text());im=Image.open(r/'Data'/e['image']).convert('RGB');dr=ImageDraw.Draw(im);w,h=im.size
 for s in d['boundarySegments']:
  a=np.array(list(s['start'].values()));b=np.array(list(s.get('control1',s['start']).values()));c=np.array(list(s.get('control2',s['end']).values()));z=np.array(list(s['end'].values()));t=np.linspace(0,1,65)[:,None];q=((1-t)**3*a+3*(1-t)**2*t*b+3*(1-t)*t*t*c+t**3*z)*[w,h]
  dr.line([tuple(v) for v in q],fill=(0,170,255),width=max(1,round(w/650)));x,y=a*[w,h];rr=max(2,w/300);dr.ellipse((x-rr,y-rr,x+rr,y+rr),fill=(255,170,40))
 im.thumbnail((1400,1400));im.save(p/f'Image {n}.jpg')
