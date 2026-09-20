from pathlib import Path
from PIL import Image,ImageDraw
import json,numpy as np
r=Path(__file__).resolve().parents[2];out=r/'Outline Review/Precision Pass';m=json.loads((r/'Data/manifest.json').read_text())
for start in range(0,50,10):
 sheet=Image.new('RGB',(1400,1020),(20,20,24));draw=ImageDraw.Draw(sheet)
 for row,e in enumerate(m[start:start+10]):
  im=np.array(Image.open(r/'Data'/e['image']).convert('RGB'));h,w=im.shape[:2];wh=np.array([w,h]);d=json.loads((out/'Candidates'/Path(e['label']).name).read_text());p=d['fullPolygon'];pts=[]
  for i,A in enumerate(p):
   B=p[(i+1)%len(p)];a=np.array([A['x'],A['y']])*wh;b=np.array(A['hOut'])*wh;c=np.array(B['hIn'])*wh;dd=np.array([B['x'],B['y']])*wh;t=np.linspace(0,1,150,endpoint=False)[:,None];pts.extend((1-t)**3*a+3*(1-t)**2*t*b+3*(1-t)*t*t*c+t**3*dd)
  pts=np.array(pts);pts=np.vstack([pts,pts[0]]);dist=np.r_[0,np.cumsum(np.linalg.norm(np.diff(pts,axis=0),axis=1))];u=np.linspace(0,dist[-1],1400);q=np.c_[np.interp(u,dist,pts[:,0]),np.interp(u,dist,pts[:,1])];v=np.gradient(q,axis=0);nn=np.c_[-v[:,1],v[:,0]]/np.linalg.norm(v,axis=1)[:,None];xy=q[None,:,:]+np.arange(-20,20,.5)[:,None,None]*nn[None,:,:];xx=np.clip(xy[:,:,0].astype(int),0,w-1);yy=np.clip(xy[:,:,1].astype(int),0,h-1);strip=Image.fromarray(im[yy,xx]);dr=ImageDraw.Draw(strip);dr.line([(0,40),(1399,40)],fill=(0,220,100));sheet.paste(strip,(0,row*102+20));draw.text((5,row*102+3),str(start+row+1),fill='white')
 sheet.save(out/f'Perimeters {start+1}.jpg')
