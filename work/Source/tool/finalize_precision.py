from pathlib import Path
import json,numpy as np
r=Path(__file__).resolve().parents[2];out=r/'Outline Review/Precision Pass';m=json.loads((r/'Data/manifest.json').read_text())
# Stabilize very short corner arcs where independent fitting can reverse tiny handles.
for n in [4,43,44]:
 name=Path(m[n-1]['label']).name;p=out/'Candidates'/name;d=json.loads(p.read_text());old=json.loads((out/'Before'/name).read_text());a=d['fullPolygon'];b=old['fullPolygon'];N=len(a)
 for i in range(0,N,2):
  j=i+1;shift=np.mean([[a[k]['x']-b[k]['x'],a[k]['y']-b[k]['y']] for k in [i,j]],axis=0)
  for k in [i,j]:
   a[k]['x'],a[k]['y']=(np.array([b[k]['x'],b[k]['y']])+shift).tolist()
   for side in ['hIn','hOut']:a[k][side]=(np.array(b[k][side])+shift).tolist()
 p.write_text(json.dumps(d,indent=2)+'\n')
# Validate every cubic path before committing any label.
for n,e in enumerate(m,1):
 p=out/'Candidates'/Path(e['label']).name;d=json.loads(p.read_text());a=d['fullPolygon'];v=[]
 for i,A in enumerate(a):
  B=a[(i+1)%len(a)];ctrl=np.array([[A['x'],A['y']],A['hOut'],B['hIn'],[B['x'],B['y']]])
  assert np.isfinite(ctrl).all() and (ctrl>=0).all() and (ctrl<=1).all(),n
  t=np.linspace(0,1,40,endpoint=False)[:,None];v.extend((1-t)**3*ctrl[0]+3*(1-t)**2*t*ctrl[1]+3*(1-t)*t*t*ctrl[2]+t**3*ctrl[3])
 v=np.array(v);b=np.roll(v,-1,axis=0);cross=lambda x,y:x[...,0]*y[...,1]-x[...,1]*y[...,0]
 for i in range(len(v)):
  j=np.arange(i+2,len(v));j=j[j!=len(v)-1] if i==0 else j
  if not len(j):continue
  ab=b[i]-v[i];cd=b[j]-v[j];hit=(cross(ab,v[j]-v[i])*cross(ab,b[j]-v[i]) < -1e-16)&(cross(cd,v[i]-v[j])*cross(cd,b[i]-v[j]) < -1e-16)
  assert not hit.any(),('intersection',n,i)
print('All 50 candidate paths finite, in bounds, and non-self-intersecting.')
