"""Offline replay of image-only candidate scores and neural masks. No labels in selection."""
from pathlib import Path
import json,numpy as np
from PIL import Image,ImageDraw
r=Path(__file__).resolve().parents[2];out=r/'Scanner Integration/Predictions Ensemble V1';out.mkdir(exist_ok=True)
weights=np.array(json.loads((r/'Scanner Integration/Ranker Training.json').read_text())['weights'])
for f in (r/'Scanner Integration/Candidate Diagnostics V7').glob('*.json'):
 candidates=json.loads(f.read_text());neural=json.loads((r/'Scanner Integration/Predictions Mobile V1'/f.name).read_text());mask=np.load(r/'Scanner Integration/FairScan Evaluation'/(f.name[:-5]+'.npy'))>.5
 best=None;bestscore=-1
 for c in candidates:
  x=np.array(c['features']);z=np.array([1,*x,*[x[i]*x[j] for i in range(11) for j in range(i,11)]]);rank=float(z@weights)
  raster=Image.new('1',(256,256));ImageDraw.Draw(raster).polygon([tuple(np.array(p)*255) for p in c['polygon']],fill=1);p=np.asarray(raster);overlap=float((p&mask).sum()/max(1,(p|mask).sum()));score=.65*overlap+.35*rank
  if score>bestscore:bestscore=score;best=c['polygon']
 p=np.array([[v['x'],v['y']] for v in neural['outerBoundary']]);raster=Image.new('1',(256,256))
 if len(p)>2:
  ImageDraw.Draw(raster).polygon([tuple(v*255) for v in p],fill=1);q=np.asarray(raster);overlap=float((q&mask).sum()/max(1,(q|mask).sum()))
  if .65*overlap+.35*.8>bestscore:best=p
 if best is not None:neural['outerBoundary']=[{'x':float(p[0]),'y':float(p[1]),'visibility':'inferred','confidence':.5} for p in best]
 neural['elapsedMs']+=json.loads((r/'Scanner Integration/Predictions V8'/f.name).read_text())['elapsedMs'];neural['timing_note']='Sum of separately measured candidate and neural runs, excludes replay ranking; not end-to-end latency';neural['modelVersion']='offline-mask-candidate-ensemble-v1';neural['reasonCodes']=['OFFLINE_REPLAY_NOT_INTEGRATED'];(out/f.name).write_text(json.dumps(neural))
