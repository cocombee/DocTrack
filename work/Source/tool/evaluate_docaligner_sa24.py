"""Official DocAligner LCNet heatmap preprocessing, local pixel-only evaluation."""
from pathlib import Path
import json,time,hashlib
import numpy as np,cv2,onnxruntime as ort
from PIL import Image,ImageOps
r=Path(__file__).resolve().parents[2];folder=r/'Scanner Integration/DocAligner Evaluation';model=folder/'fastvit_sa24.onnx'
opts=ort.SessionOptions();opts.intra_op_num_threads=2;opts.inter_op_num_threads=1
session=ort.InferenceSession(str(model),sess_options=opts,providers=['CPUExecutionProvider']);print([(v.name,v.shape) for v in session.get_inputs()],[(v.name,v.shape) for v in session.get_outputs()],flush=True)
out=r/'Scanner Integration/Predictions DocAligner SA24';out.mkdir(exist_ok=True)
for f in sorted((r/'Data/originals').rglob('*.jpg')):
 start=time.perf_counter();im=ImageOps.exif_transpose(Image.open(f)).convert('RGB');w,h=im.size
 x=cv2.resize(np.asarray(im)[:,:,::-1],(256,256)).transpose(2,0,1).astype(np.float32)[None]/255
 result=session.run(None,{session.get_inputs()[0].name:x});heatmaps=result[0][0];points=[]
 for heat in heatmaps:
  heat=cv2.resize(heat,(w,h));heat[heat<min(.3,max(.05,float(heat.max())*.5))]=0;heat=(heat*255).astype(np.uint8);_,binary=cv2.threshold(heat,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
  contours,_=cv2.findContours(binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
  if not contours:continue
  m=cv2.moments(max(contours,key=cv2.contourArea))
  if m['m00']>0:points.append({'x':m['m10']/m['m00']/w,'y':m['m01']/m['m00']/h,'visibility':'inferred','confidence':.5})
 data={'schemaVersion':1,'width':w,'height':h,'status':'uncertain' if len(points)==4 else 'notFound','outerBoundary':points if len(points)==4 else [],'confidence':.5,'modelVersion':'docaligner-fastvit-sa24-evaluation','preprocessingVersion':'upright-bgr-256-unit-nchw','reasonCodes':['EVALUATION_ONLY'],'creaseLines':[],'elapsedMs':round((time.perf_counter()-start)*1000)}
 (out/(f.name+'.json')).write_text(json.dumps(data));print(f.name,data['elapsedMs'],flush=True)
(folder/'SA24 Provenance.json').write_text(json.dumps({'url':'https://drive.google.com/uc?id=14vUH77v6yGg7zFctUgcT6BzV5Iisg4Dl','upstream':'https://github.com/DocsaidLab/DocAligner','version':'1.1.1','sha256':hashlib.sha256(model.read_bytes()).hexdigest(),'code_license':'Apache-2.0','weights_license':'Needs explicit scope verification before distribution','status':'isolated evaluation only'},indent=2))
