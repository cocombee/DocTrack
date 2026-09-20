from pathlib import Path
import json
from PIL import Image,ImageDraw
r=Path(__file__).resolve().parents[2];m=json.loads((r/'Data/manifest.json').read_text());review=r/'Outline Review/Remaining 47'
for start in range(0,50,10):
 canvas=Image.new('RGB',(1800,800),(28,28,30));draw=ImageDraw.Draw(canvas)
 for k,x in enumerate(m[start:start+10]):
  im=Image.open(r/'Data'/x['image']).convert('RGB');w,h=im.size;im.thumbnail((350,360));iw,ih=im.size
  xx=(k%5)*360+(360-iw)//2;yy=(k//5)*400+30
  d=json.loads((r/'Data'/x['label']).read_text());p=d.get('fullPolygon') or [dict(x=v[0],y=v[1]) for v in d['corners']]
  pts=[]
  for i,a in enumerate(p):
   b=p[(i+1)%len(p)];aa=(a['x']*iw,a['y']*ih);bb=(b['x']*iw,b['y']*ih)
   c=a.get('hOut',[a['x'],a['y']]) if a.get('smooth') else [a['x'],a['y']]
   e=b.get('hIn',[b['x'],b['y']]) if b.get('smooth') else [b['x'],b['y']]
   for t in [j/12 for j in range(12)]:pts.append(((1-t)**3*aa[0]+3*(1-t)**2*t*c[0]*iw+3*(1-t)*t*t*e[0]*iw+t**3*bb[0],(1-t)**3*aa[1]+3*(1-t)**2*t*c[1]*ih+3*(1-t)*t*t*e[1]*ih+t**3*bb[1]))
  dr=ImageDraw.Draw(im);dr.line(pts+[pts[0]],fill=(0,255,130),width=2)
  for a in p:
   q=(a['x']*iw,a['y']*ih);dr.ellipse((q[0]-2,q[1]-2,q[0]+2,q[1]+2),fill='white')
  canvas.paste(im,(xx,yy));draw.text(((k%5)*360+8,(k//5)*400+8),f"{start+k+1}. {len(p)} points",fill='white')
 canvas.save(review/f'Overlay {start+1}-{min(start+10,50)}.jpg')
