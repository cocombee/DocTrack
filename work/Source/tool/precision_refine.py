from pathlib import Path
import json,numpy as np,shutil
from PIL import Image,ImageDraw
r=Path(__file__).resolve().parents[2];m=json.loads((r/'Data/manifest.json').read_text());out=r/'Outline Review/Precision Pass';out.mkdir(exist_ok=True);(out/'Before').mkdir(exist_ok=True);(out/'Candidates').mkdir(exist_ok=True)
# Exclude foreground hands and cropped intervals from edge evidence. Coordinates are normalized.
masks={7:[(0,.40,.09,.65)],10:[(.76,.48,1,.72)],15:[(0,.40,.20,.70)],18:[(.73,.65,1,1)],22:[(0,.29,.27,.44),(0,.89,.24,1)],26:[(0,.46,.20,.58)],28:[(0,.36,.16,.46)],29:[(0,0,.015,1)],37:[(.75,.76,.92,1)],41:[(0,0,.015,1),(0,.775,1,1)],43:[(.36,.43,.46,.67),(.60,.50,.68,.63)],44:[(.29,.44,.38,.55),(.83,.38,.92,.46)],45:[(.92,.41,1,.71)],48:[(0,0,1,.008),(0,.99,1,1)]}
def bilinear(im,xy):
 h,w=im.shape[:2];x=np.clip(xy[...,0],0,w-1.001);y=np.clip(xy[...,1],0,h-1.001);ix=x.astype(int);iy=y.astype(int);dx=(x-ix)[...,None];dy=(y-iy)[...,None]
 return (1-dy)*((1-dx)*im[iy,ix]+dx*im[iy,ix+1])+dy*((1-dx)*im[iy+1,ix]+dx*im[iy+1,ix+1])
def getpath(d,wh):
 p=d['fullPolygon'];return np.array([[v['x'],v['y']] for v in p])*wh,np.array([v.get('hOut',[v['x'],v['y']]) if v.get('smooth') else [v['x'],v['y']] for v in p])*wh,np.array([v.get('hIn',[v['x'],v['y']]) if v.get('smooth') else [v['x'],v['y']] for v in p])*wh
reports=[]
for idx,e in enumerate(m,1):
 path=r/'Data'/e['label'];dest=out/'Before'/path.name
 if not dest.exists():shutil.copy2(path,dest)
 d=json.loads(dest.read_text());im=Image.open(r/'Data'/e['image']).convert('RGB');wh=np.array(im.size);rgb=np.asarray(im,dtype=float);a,ho,hi=getpath(d,wh);N=len(a);base=np.vstack([a,ho,hi]);rows=[];targets=[];weights=[];observed=[];distances=[]
 for i in range(N):
  j=(i+1)%N;A,B,C,D=a[i],ho[i],hi[j],a[j];length=np.linalg.norm(D-A);num=max(8,min(170,int(length/3)));t=np.linspace(.025,.975,num);b=np.array([(1-t)**3,3*(1-t)**2*t,3*(1-t)*t*t,t**3]).T
  q=b@np.array([A,B,C,D]);v=3*(1-t[:,None])**2*(B-A)+6*(1-t[:,None])*t[:,None]*(C-B)+3*t[:,None]**2*(D-C);norm=np.linalg.norm(v,axis=1);normal=np.c_[-v[:,1],v[:,0]]/np.maximum(norm[:,None],1e-6)
  reach=max(9,min(45,max(wh)*.010));offset=np.arange(-reach,reach+1,.75);xy=q[:,None,:]+offset[None,:,None]*normal[:,None,:];step=max(1.2,min(3.0,max(wh)/900));inside=bilinear(rgb,xy+step*normal[:,None,:]);outside=bilinear(rgb,xy-step*normal[:,None,:]);delta=inside-outside
  # Prefer a coherent physical color transition near the expected boundary, not printed text.
  strength=np.sqrt((delta**2).mean(2));signed=delta.mean(2);score=strength*(.70+.30*(signed>0))-.16*(offset[None,:]+4)**2
  dp=score[0].copy();back=[];pen=.32*(offset[:,None]-offset[None,:])**2
  for z in range(1,num):
   costs=dp[:,None]-pen;k=costs.argmax(0);back.append(k);dp=score[z]+costs[k,np.arange(len(offset))]
  k=int(dp.argmax());seq=[k]
  for prev in reversed(back):k=int(prev[k]);seq.append(k)
  seq=np.array(seq[::-1]);chosen=offset[seq];st=strength[np.arange(num),seq]
  for z in range(num):
   qn=q[z]/wh
   hidden=any(x0<=qn[0]<=x1 and y0<=qn[1]<=y1 for x0,y0,x1,y1 in masks.get(idx,[]))
   # Weak or ambiguous transitions retain existing inferred geometry.
   weight=0 if hidden else np.clip((st[z]-10)/35,0,1)
   if abs(chosen[z]+4)>reach*.85:weight*=.25
   row=np.zeros(3*N);row[i]=b[z,0];row[N+i]=b[z,1];row[2*N+j]=b[z,2];row[j]=b[z,3]
   rows.append(row);targets.append(q[z]+(chosen[z]+4)*normal[z]);weights.append(weight);observed.append(q[z]+chosen[z]*normal[z]);distances.append(chosen[z]+4)
 X=np.array(rows);Y=np.array(targets);w=np.array(weights);old=base.copy()
 # Regularize handles more strongly on short corner arcs to retain stable tangent joins.
 reg=np.ones(3*N)*.9
 for i in range(N):
  reg[N+i]=reg[2*N+(i+1)%N]=max(.45,6/(1+np.linalg.norm(a[(i+1)%N]-a[i])/10))
 for iteration in range(3):
  Xw=X*np.sqrt(w[:,None]);Yw=Y*np.sqrt(w[:,None]);sol=np.linalg.lstsq(np.vstack([Xw,np.diag(np.sqrt(reg))]),np.vstack([Yw,np.sqrt(reg[:,None])*old]),rcond=None)[0]
  residual=np.linalg.norm(X@sol-Y,axis=1);w=np.array(weights)*np.minimum(1,3/np.maximum(residual,1e-6))
 sol=np.clip(sol,[0,0],wh);a2,ho2,hi2=np.split(sol,3)
 for i,p in enumerate(d['fullPolygon']):p.update(x=float(a2[i,0]/wh[0]),y=float(a2[i,1]/wh[1]),hOut=(ho2[i]/wh).tolist(),hIn=(hi2[i]/wh).tolist(),smooth=True,broken=True)
 d['edge_refined']='Source-resolution coherent RGB edge sampling, occlusion masks, robust regularized cubic fitting; 4px inward target. Pending close visual verification.';d['outline_review']=dict(d.get('outline_review',{}),status='precision candidate; pending visual verification');(out/'Candidates'/path.name).write_text(json.dumps(d,indent=2)+'\n')
 # Render candidate full perimeter at up to 1100 pixels, retaining source coordinate accuracy.
 thumb=im.copy();thumb.thumbnail((1100,1100));sc=np.array(thumb.size)/wh;draw=ImageDraw.Draw(thumb);pts=[]
 for i in range(N):
  j=(i+1)%N;t=np.linspace(0,1,35,endpoint=False)[:,None];pts.extend(((1-t)**3*a2[i]+3*(1-t)**2*t*ho2[i]+3*(1-t)*t*t*hi2[j]+t**3*a2[j])*sc)
 draw.line([tuple(q) for q in pts]+[tuple(pts[0])],fill=(0,220,100),width=2)
 for q in a2*sc:draw.ellipse((q[0]-2,q[1]-2,q[0]+2,q[1]+2),fill='white')
 thumb.save(out/f'{idx:02}.png');reports.append(dict(index=idx,mean_move_px=float(np.linalg.norm(a2-a,axis=1).mean()),max_move_px=float(np.linalg.norm(a2-a,axis=1).max()),supported_fraction=float((np.array(weights)>.2).mean())))
(out/'Fit Report.json').write_text(json.dumps(reports,indent=2)+'\n');print('50 candidates generated; original labels not modified.')
