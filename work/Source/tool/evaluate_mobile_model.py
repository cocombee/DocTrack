"""Standalone, local-only model evaluation. Labels are not inference inputs."""
from pathlib import Path
import json,time
import numpy as np,cv2
from PIL import Image,ImageOps
from ai_edge_litert.interpreter import Interpreter
root=Path(__file__).resolve().parents[2]
model=root/'Scanner Integration/FairScan Evaluation/fairscan-segmentation-model.tflite'
interpreter=Interpreter(model_path=str(model),num_threads=2);interpreter.allocate_tensors()
a=interpreter.get_input_details()[0];b=interpreter.get_output_details()[0]
print('INPUT',a['shape'],a['dtype'],'OUTPUT',b['shape'],b['dtype'],flush=True)
out=root/'Scanner Integration/Predictions Mobile V1';out.mkdir(exist_ok=True)
for f in sorted((root/'Data/originals').rglob('*.jpg')):
 start=time.perf_counter();im=ImageOps.exif_transpose(Image.open(f)).convert('RGB');w,h=im.size
 ih,iw=map(int,a['shape'][1:3]);x=np.asarray(im.resize((iw,ih),Image.Resampling.BILINEAR),dtype=np.float32)/127.5-1
 interpreter.set_tensor(a['index'],x[None]);interpreter.invoke();mask=interpreter.get_tensor(b['index']).squeeze();np.save(root/'Scanner Integration/FairScan Evaluation'/(f.name+'.npy'),mask);mask=(mask>.5).astype(np.uint8)*255
 contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
 points=[]
 if contours:
  c=max(contours,key=cv2.contourArea);c=cv2.convexHull(c);c=cv2.approxPolyDP(c,1.0,True).reshape(-1,2)
  points=[{'x':float(v[0]/iw),'y':float(v[1]/ih),'visibility':'inferred','confidence':0.5} for v in c]
 data={'schemaVersion':1,'width':w,'height':h,'status':'uncertain','outerBoundary':points,'confidence':0.5,'modelVersion':'fairscan-v1.2.0-evaluation','preprocessingVersion':'upright-rgb-bilinear-minus1plus1','reasonCodes':['EVALUATION_ONLY'],'creaseLines':[],'elapsedMs':round((time.perf_counter()-start)*1000)}
 (out/(f.name+'.json')).write_text(json.dumps(data));print(f.name,data['elapsedMs'],flush=True)
