"""Refine a crop-rescued prediction by recentering on its image-derived extent."""
from pathlib import Path
import json,numpy as np,cv2,onnxruntime as ort
from PIL import Image,ImageOps
r=Path(__file__).resolve().parents[2];opts=ort.SessionOptions();opts.intra_op_num_threads=2;folder=r/'Scanner Integration/DocAligner Evaluation';hm=ort.InferenceSession(str(folder/'lcnet100.onnx'),sess_options=opts,providers=['CPUExecutionProvider']);pm=ort.InferenceSession(str(folder/'lcnet050.onnx'),sess_options=opts,providers=['CPUExecutionProvider'])
def tensor(a):return cv2.resize(a,(256,256)).transpose(2,0,1).astype(np.float32)[None]/255
def heat(a):
 maps=hm.run(None,{'img':tensor(a)})[0][0];p=[]
 for m in maps:
  peak=float(m.max())
  if peak<.05:return []
  binary=(m>min(.3,max(.05,peak*.5))).astype(np.uint8);cs,_=cv2.findContours(binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
  if not cs:return []
  z=cv2.moments(max(cs,key=cv2.contourArea))
  if z['m00']<=0:return []
  p.append([(z['m10']/z['m00']+.5)/m.shape[1],(z['m01']/z['m00']+.5)/m.shape[0]])
 return p
out=r/'Scanner Integration/Predictions Recenter V1';out.mkdir(exist_ok=True)
for f in sorted((r/'Data/originals').rglob('*.jpg')):
 d=json.loads((r/'Scanner Integration/Predictions Dart ONNX V5'/(f.name+'.json')).read_text());q=np.array([[v['x'],v['y']] for v in d['outerBoundary']]);reason=' '.join(d.get('reasonCodes',[]))
 if 'crop:' in reason and 'crop:1.0' not in reason and len(q)==4:
  im=np.asarray(ImageOps.exif_transpose(Image.open(f)).convert('RGB'))[:,:,::-1];h,w=im.shape[:2];low=q.min(0);high=q.max(0);center=(low+high)/2;size=np.maximum(high-low,.15)*1.6;low=np.clip(center-size/2,0,1);high=np.clip(center+size/2,0,1);x0,y0=(low*[w,h]).astype(int);x1,y1=(high*[w,h]).astype(int);crop=im[y0:y1,x0:x1]
  for turns in range(4):
   a=np.rot90(crop,turns).copy();p=heat(a)
   if len(p)!=4:
    v=dict(zip([x.name for x in pm.get_outputs()],pm.run(None,{'img':tensor(a)})))
    if float(v['has_obj'].ravel()[0])>.5:p=v['points'].reshape(4,2).tolist()
   if len(p)==4 and all(.03<x<.97 and .03<y<.97 for x,y in p):
    if turns==1:p=[[1-y,x] for x,y in p]
    elif turns==2:p=[[1-x,1-y] for x,y in p]
    elif turns==3:p=[[y,1-x] for x,y in p]
    q=[[(x*(x1-x0)+x0)/w,(y*(y1-y0)+y0)/h] for x,y in p];d['outerBoundary']=[{'x':float(x),'y':float(y),'visibility':'inferred','confidence':.5} for x,y in q];d['recenter']=True;break
 d['modelVersion']='recenter-evaluation';d['timing_note']='recenter time not included';(out/(f.name+'.json')).write_text(json.dumps(d))
