from pathlib import Path
import json,time
import numpy as np,cv2,onnxruntime as ort
from PIL import Image,ImageOps
r=Path(__file__).resolve().parents[2];model=r/'Scanner Integration/DocAligner Evaluation/lcnet050.onnx'
opts=ort.SessionOptions();opts.intra_op_num_threads=2;session=ort.InferenceSession(str(model),sess_options=opts,providers=['CPUExecutionProvider']);names=[v.name for v in session.get_outputs()];print(names,flush=True)
out=r/'Scanner Integration/Predictions Point V1';out.mkdir(exist_ok=True)
for f in sorted((r/'Data/originals').rglob('*.jpg')):
 start=time.perf_counter();im=ImageOps.exif_transpose(Image.open(f)).convert('RGB');w,h=im.size;x=cv2.resize(np.asarray(im)[:,:,::-1],(256,256)).transpose(2,0,1).astype(np.float32)[None]/255
 result=dict(zip(names,session.run(None,{session.get_inputs()[0].name:x})));q=result['points'].reshape(4,2);confidence=float(result['has_obj'].ravel()[0]);points=[{'x':float(v[0]),'y':float(v[1]),'visibility':'inferred','confidence':confidence} for v in q] if confidence>.5 else []
 data={'width':w,'height':h,'status':'uncertain' if points else 'notFound','outerBoundary':points,'confidence':confidence,'modelVersion':'docaligner-lcnet050-evaluation','elapsedMs':round((time.perf_counter()-start)*1000)};(out/(f.name+'.json')).write_text(json.dumps(data))
