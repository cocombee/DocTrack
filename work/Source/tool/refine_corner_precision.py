from pathlib import Path
import json,numpy as np
r=Path(__file__).resolve().parents[2];out=r/'Scanner Integration/Predictions Corner Precision';out.mkdir(exist_ok=True)
for f in (r/'Scanner Integration/Predictions Dart ONNX V11').glob('*.jpg.json'):
 d=json.loads(f.read_text());wh=np.array([d['width'],d['height']]);p=np.array([[v['x'],v['y']] for v in d['outerBoundary']])*wh;q=np.array([[v['x'],v['y']] for v in d.get('corners') or []])*wh
 # Preserve the approved eight-segment rounded-card representation.
 if len(d.get('boundarySegments',[]))==8 and sum('control1'in v for v in d['boundarySegments'])==4:
  (out/f.name).write_text(json.dumps(d));continue
 changes=[]
 for corner in q:
  i=np.linalg.norm(p-corner,axis=1).argmin();lines=[]
  for direction in [-1,1]:
   near=[];j=(i+direction)%len(p)
   for k in range(len(p)-1):
    distance=np.linalg.norm(p[j]-corner)
    if 12<distance<42:near.append(p[j])
    if distance>=42:break
    j=(j+direction)%len(p)
   if len(near)<4:break
   near=np.array(near);center=near.mean(0);_,_,v=np.linalg.svd(near-center);lines.append((center,v[0]))
  if len(lines)!=2:continue
  a,u=lines[0];b,v=lines[1];det=np.linalg.det(np.column_stack([u,-v]))
  if abs(det)<.4:continue
  t=np.linalg.solve(np.column_stack([u,-v]),b-a)[0];target=a+t*u;delta=target-p[i]
  if np.linalg.norm(delta)>8:continue
  changes.append((int(i),delta))
 original=p.copy()
 for i,delta in changes:
  for direction in [-1,1]:
   j=i if direction==1 else (i-1)%len(p)
   for k in range(len(p)):
    distance=np.linalg.norm(original[j]-original[i])
    if distance>15:break
    p[j]+=delta*(1-distance/15)**2;j=(j+direction)%len(p)
 d['outerBoundary']=[{'x':float(v[0]/wh[0]),'y':float(v[1]/wh[1]),'visibility':'inferred','confidence':.5} for v in p];d['cornerAdjustments']=[{'index':i,'dx':float(v[0]),'dy':float(v[1])} for i,v in changes];(out/f.name).write_text(json.dumps(d))
 if f.name.startswith(('document_segmentation_tzzh7_001989','filtered_document_segmentation_000995')):print(f.name,d['cornerAdjustments'])
