"""Evaluation only: source-pixel boundary error after the requested 4px inset.
Reference JSON is never read by the detector. IoU alone cannot verify precision.
"""
from pathlib import Path
import json,sys,numpy as np
version=sys.argv[1] if len(sys.argv)>1 else 'Dart ONNX V6'
r=Path(__file__).resolve().parents[2];baseline=r/'Scanner Integration/Reference Baseline';manifest=json.loads((baseline/'manifest.json').read_text());reports=[]
def inset(p,d):
 area=np.sum(p[:,0]*np.roll(p[:,1],-1)-p[:,1]*np.roll(p[:,0],-1));sign=1 if area>0 else -1;v=np.roll(p,-1,axis=0)-p;v=v/np.maximum(np.linalg.norm(v,axis=1,keepdims=True),1e-9);n=sign*np.column_stack([-v[:,1],v[:,0]]);previous=np.roll(n,1,axis=0);den=1+np.sum(previous*n,axis=1);return p+d*(previous+n)/np.maximum(den[:,None],.1)
def sample(p):
 return np.concatenate([a+(b-a)*np.linspace(0,1,max(4,int(np.linalg.norm(b-a)/3)),endpoint=False)[:,None] for a,b in zip(p,np.roll(p,-1,axis=0))])
def distance(points,path):
 a=path;b=np.roll(path,-1,axis=0);v=b-a;den=np.maximum(np.sum(v*v,axis=1),1e-12);out=[]
 for start in range(0,len(points),128):
  q=points[start:start+128,None,:]-a[None];t=np.clip(np.sum(q*v,axis=2)/den,0,1);delta=q-t[:,:,None]*v;out.extend(np.sqrt(np.min(np.sum(delta*delta,axis=2),axis=1)).tolist())
 return np.array(out)
for i,e in enumerate(manifest,1):
 d=json.loads((r/f'Scanner Integration/Predictions {version}'/(e['name']+'.json')).read_text());wh=np.array([d['width'],d['height']]);p=np.array([[v['x'],v['y']] for v in d['outerBoundary']])*wh
 ref=json.loads((baseline/e['label']).read_text())['fullPolygon'];curve=[]
 for j,a in enumerate(ref):
  b=ref[(j+1)%len(ref)];A=np.array([a['x'],a['y']])*wh;D=np.array([b['x'],b['y']])*wh;B=np.array(a['hOut'])*wh if a.get('smooth') else A;C=np.array(b['hIn'])*wh if b.get('smooth') else D;n=max(8,int((np.linalg.norm(B-A)+np.linalg.norm(C-B)+np.linalg.norm(D-C))/2));t=np.linspace(0,1,n,endpoint=False)[:,None];curve.extend((1-t)**3*A+3*(1-t)**2*t*B+3*(1-t)*t*t*C+t**3*D)
 curve=np.array(curve);prediction=inset(p,4);errors=np.concatenate([distance(sample(prediction),curve),distance(curve,prediction)]);reports.append({'index':i,'name':e['name'],'mean_source_px':float(errors.mean()),'p95_source_px':float(np.percentile(errors,95)),'max_source_px':float(errors.max()),'within_4px_p95':bool(np.percentile(errors,95)<=4)})
summary={'count':len(reports),'p95_within_4_source_pixels':sum(v['within_4px_p95'] for v in reports),'comparison':'Symmetric distance to saved reference curves, after4px source-pixel inset of predicted polygon','limitations':'References are manually inferred and not all individually approved; this is a regression diagnostic, not proof of hidden-edge truth.'}
(r/f'Scanner Integration/Precision {version}.json').write_text(json.dumps({'summary':summary,'images':reports},indent=2));print(summary);print('largest p95',[(v['index'],round(v['p95_source_px'],1)) for v in sorted(reports,key=lambda v:v['p95_source_px'],reverse=True)[:10]])
