"""Pixel-only cascade: heatmap, point regression, then center crop rescue."""
from pathlib import Path
import json,time
import numpy as np,cv2,onnxruntime as ort
from PIL import Image,ImageOps
r=Path(__file__).resolve().parents[2];opts=ort.SessionOptions();opts.intra_op_num_threads=2
folder=r/'Scanner Integration/DocAligner Evaluation';hm=ort.InferenceSession(str(folder/'lcnet100.onnx'),sess_options=opts,providers=['CPUExecutionProvider']);pm=ort.InferenceSession(str(folder/'lcnet050.onnx'),sess_options=opts,providers=['CPUExecutionProvider'])
def tensor(a):return cv2.resize(a,(256,256)).transpose(2,0,1).astype(np.float32)[None]/255
def heat(a):
 maps=hm.run(None,{'img':tensor(a)})[0][0];p=[]
 for m in maps:
  peak=float(m.max())
  if peak<.05:return []
  binary=(m>min(.3,max(.05,peak*.5))).astype(np.uint8);cs,_=cv2.findContours(binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
  if not cs:return []
  moment=cv2.moments(max(cs,key=cv2.contourArea))
  if moment['m00']<=0:return []
  p.append([moment['m10']/moment['m00']/m.shape[1],moment['m01']/moment['m00']/m.shape[0]])
 return p
out=r/'Scanner Integration/Predictions Cascade V3';out.mkdir(exist_ok=True)
for f in sorted((r/'Data/originals').rglob('*.jpg')):
 start=time.perf_counter();im=ImageOps.exif_transpose(Image.open(f)).convert('RGB');a=np.asarray(im)[:,:,::-1];h,w=a.shape[:2];p=heat(a);method='heatmap'
 if len(p)!=4:
  result=dict(zip([v.name for v in pm.get_outputs()],pm.run(None,{'img':tensor(a)})))
  if float(result['has_obj'].ravel()[0])>.5:p=result['points'].reshape(4,2).tolist();method='point'
 if len(p)!=4:
  for fraction in [.65,.4]:
   x0=int(w*(1-fraction)/2);y0=int(h*(1-fraction)/2);cw=w-2*x0;ch=h-2*y0;q=heat(a[y0:y0+ch,x0:x0+cw])
   if len(q)==4 and all(.025<x<.975 and .025<y<.975 for x,y in q):p=[[(x*cw+x0)/w,(y*ch+y0)/h] for x,y in q];method='center-crop';break
 if len(p)!=4:
  for turns in [1,2,3]:
   rotated=np.rot90(a,turns).copy();q=heat(rotated)
   if len(q)!=4:
    result=dict(zip([v.name for v in pm.get_outputs()],pm.run(None,{'img':tensor(rotated)})))
    if float(result['has_obj'].ravel()[0])>.5:q=result['points'].reshape(4,2).tolist()
   if len(q)==4:
    if turns==1:p=[[1-y,x] for x,y in q]
    elif turns==2:p=[[1-x,1-y] for x,y in q]
    else:p=[[y,1-x] for x,y in q]
    method='rotation-rescue';break
 if len(p)!=4:
  for fraction in [.65,.4]:
   x0=int(w*(1-fraction)/2);y0=int(h*(1-fraction)/2);cw=w-2*x0;ch=h-2*y0
   for turns in [1,2,3]:
    crop=np.rot90(a[y0:y0+ch,x0:x0+cw],turns).copy();q=heat(crop)
    if len(q)!=4:
     result=dict(zip([v.name for v in pm.get_outputs()],pm.run(None,{'img':tensor(crop)})))
     if float(result['has_obj'].ravel()[0])>.5:q=result['points'].reshape(4,2).tolist()
    if len(q)==4 and all(.025<x<.975 and .025<y<.975 for x,y in q):
     if turns==1:q=[[1-y,x] for x,y in q]
     elif turns==2:q=[[1-x,1-y] for x,y in q]
     else:q=[[y,1-x] for x,y in q]
     p=[[(x*cw+x0)/w,(y*ch+y0)/h] for x,y in q];method='crop-rotation-rescue';break
   if len(p)==4:break
 points=[{'x':float(x),'y':float(y),'visibility':'inferred','confidence':.5} for x,y in p]
 data={'width':w,'height':h,'status':'uncertain' if len(p)==4 else 'notFound','outerBoundary':points,'confidence':.5,'modelVersion':'docaligner-cascade-evaluation','method':method,'elapsedMs':round((time.perf_counter()-start)*1000)};(out/(f.name+'.json')).write_text(json.dumps(data))
