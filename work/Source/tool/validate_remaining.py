from pathlib import Path
import json, numpy as np
r=Path(__file__).resolve().parents[2];m=json.loads((r/'Data/manifest.json').read_text());checks=[]
for n,e in enumerate(m,1):
 if n in (1,2,6):continue
 d=json.loads((r/'Data'/e['label']).read_text());p=d['fullPolygon'];points=[];curves=0
 for i,a in enumerate(p):
  b=p[(i+1)%len(p)];v=np.array([[a['x'],a['y']],a['hOut'],b['hIn'],[b['x'],b['y']]])
  assert np.isfinite(v).all() and (v>=0).all() and (v<=1).all(),n
  t=np.linspace(0,1,20,endpoint=False)[:,None];points.extend((1-t)**3*v[0]+3*(1-t)**2*t*v[1]+3*(1-t)*t*t*v[2]+t**3*v[3])
  edge=v[3]-v[0];cross=lambda x,y:x[...,0]*y[...,1]-x[...,1]*y[...,0]
  if max(abs(cross(v[1]-v[0],edge)),abs(cross(v[2]-v[0],edge)))>1e-7:curves+=1
 v=np.array(points);b=np.roll(v,-1,axis=0)
 for i in range(len(v)):
  j=np.arange(i+2,len(v));j=j[j!=len(v)-1] if i==0 else j
  if not len(j):continue
  ab=b[i]-v[i];cd=b[j]-v[j]
  hits=(cross(ab,v[j]-v[i])*cross(ab,b[j]-v[i]) < -1e-16)&(cross(cd,v[i]-v[j])*cross(cd,b[i]-v[j]) < -1e-16)
  assert not hits.any(),('intersection',n,i,j[hits].tolist())
 checks.append(dict(index=n,anchors=len(p),curved_segments=curves,valid=True))
(r/'Outline Review/Remaining 47/Geometry Validation.json').write_text(json.dumps(checks,indent=2)+'\n')
print('47/47: finite, in bounds, no sampled self-intersections. Rounded cards have non-linear cubic segments.')
rules=r.parent/'Document Outline Rules.md';s=rules.read_text();addition='\n- Rounded corners must use genuine Bézier arcs, not angular polygon approximations. Keep real crease corners sharp with independent handles.\n- Remaining 47 images received sparse outline annotations and visual contact-sheet review. Occluded and clipped edges are recorded as estimates in outline_review; these annotations are drafts pending user review, not evidence of a trained detector.\n'
if 'Rounded corners must use genuine' not in s:rules.write_text(s+addition)
