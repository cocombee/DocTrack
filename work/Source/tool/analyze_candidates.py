from pathlib import Path
from PIL import Image,ImageDraw
import numpy as np,json
r=Path(__file__).resolve().parents[2];m=json.loads((r/'Scanner Integration/Reference Baseline/manifest.json').read_text());rows=[];oracle=[]
for n,e in enumerate(m,1):
 f=r/'Scanner Integration/Candidate Diagnostics V6'/(e['name']+'.json')
 if not f.exists():continue
 cand=json.loads(f.read_text());label=json.loads((r/'Scanner Integration/Reference Baseline'/e['label']).read_text());p=label['fullPolygon'];path=[]
 for i,a in enumerate(p):
  b=p[(i+1)%len(p)];A=np.array([a['x'],a['y']]);B=np.array(a['hOut']);C=np.array(b['hIn']);D=np.array([b['x'],b['y']]);t=np.linspace(0,1,20,endpoint=False)[:,None];path.extend((1-t)**3*A+3*(1-t)**2*t*B+3*(1-t)*t*t*C+t**3*D)
 mask=Image.new('1',(128,128));ImageDraw.Draw(mask).polygon([tuple(v*127) for v in path],fill=1);ref=np.array(mask);best=0
 for c in cand:
  mask=Image.new('1',(128,128));ImageDraw.Draw(mask).polygon([tuple(np.array(v)*127) for v in c['polygon']],fill=1);q=np.array(mask);iou=(q&ref).sum()/max(1,(q|ref).sum());c['iou']=float(iou);best=max(best,iou);rows.append({'index':n,'split':e['split'],**c})
 oracle.append((n,round(float(best),3),len(cand)))
(r/'Scanner Integration/Candidate Scores V6.json').write_text(json.dumps(rows));print('oracle',oracle);print('oracle >= .90',sum(v[1]>=.9 for v in oracle));
