"""Image-only orientation consensus; references are used only by separate scoring."""
from pathlib import Path
exec((Path(__file__).parent/'evaluate_recenter.py').read_text().split("out=r/")[0])
out=r/'Scanner Integration/Predictions Rotation Consensus';out.mkdir(exist_ok=True)
for f in sorted((r/'Data/originals').rglob('*.jpg')):
 d=json.loads((r/'Scanner Integration/Predictions Dart ONNX V6'/(f.name+'.json')).read_text());q=np.array([[v['x'],v['y']] for v in d['outerBoundary']]);im=np.asarray(ImageOps.exif_transpose(Image.open(f)).convert('RGB'))[:,:,::-1];h,w=im.shape[:2];lo=q.min(0);hi=q.max(0);center=(lo+hi)/2;size=np.maximum(hi-lo,.15)*1.6;lo=np.clip(center-size/2,0,1);hi=np.clip(center+size/2,0,1);x0,y0=(lo*[w,h]).astype(int);x1,y1=(hi*[w,h]).astype(int);crop=im[y0:y1,x0:x1];proposals=[]
 for turns in range(4):
  a=np.rot90(crop,turns).copy();hp=heat(a);v=dict(zip([x.name for x in pm.get_outputs()],pm.run(None,{'img':tensor(a)})));pp=v['points'].reshape(4,2).tolist() if float(v['has_obj'].ravel()[0])>.8 else []
  for raw in [hp,pp]:
   if len(raw)!=4 or not all(.025<x<.975 and .025<y<.975 for x,y in raw):continue
   p=np.array(raw)
   if turns==1:p=np.column_stack([1-p[:,1],p[:,0]])
   elif turns==2:p=1-p
   elif turns==3:p=np.column_stack([p[:,1],1-p[:,0]])
   p=(p*[x1-x0,y1-y0]+[x0,y0])/[w,h];p=min([np.roll(p,k,axis=0) for k in range(4)],key=lambda z:np.linalg.norm(z-q));proposals.append(p)
 if len(proposals)>=3:
  estimates=np.array(proposals);median=np.median(estimates,axis=0);d['outerBoundary']=[{'x':float(x),'y':float(y),'visibility':'inferred','confidence':.5} for x,y in median]
 d['modelVersion']='rotation-consensus-evaluation';d['timing_note']='consensus time not included';d['candidateCount']=len(proposals);(out/(f.name+'.json')).write_text(json.dumps(d))
