from pathlib import Path
import json,time
import numpy as np,cv2
from PIL import Image,ImageOps,ImageDraw
from ai_edge_litert.interpreter import Interpreter
root=Path('..');out=root/'Scanner Integration/Segmentation Evaluation';m=json.loads((root/'Data/manifest.json').read_text())
model=Interpreter(model_path=str(out/'fairscan-segmentation-model.tflite'),num_threads=4);model.allocate_tensors();a=model.get_input_details()[0];b=model.get_output_details()[0];print(a['shape'],a['dtype'],b['shape'],flush=True)
rows=[];panels=[]
for i,e in enumerate(m,1):
 im=ImageOps.exif_transpose(Image.open(root/'Data'/e['image'])).convert('RGB');w,h=im.size
 data=np.array(im.resize((256,256)),np.float32)/255;data=(data-np.array([.485,.456,.406],np.float32))/np.array([.229,.224,.225],np.float32)
 start=time.perf_counter();model.set_tensor(a['index'],data[None]);model.invoke();logits=model.get_tensor(b['index']).squeeze();elapsed=time.perf_counter()-start
 mask=(logits>0).astype(np.uint8)*255;Image.fromarray(mask).save(out/f'Mask {i}.png')
 contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
 contour=max(contours,key=cv2.contourArea).reshape(-1,2)/256 if contours else np.zeros((0,2))
 rows.append({'index':i,'name':e['name'],'seconds':elapsed,'polygon':contour.tolist()})
 dr=ImageDraw.Draw(im);points=[(float(x*w),float(y*h)) for x,y in contour]
 if points:dr.line(points+[points[0]],fill=(255,0,100),width=max(2,round(w/400)))
 im.thumbnail((380,420));im.save(out/f'Image {i}.jpg');panels.append(im)
for start in range(0,50,10):
 sheet=Image.new('RGB',(2000,920),(22,22,26));dr=ImageDraw.Draw(sheet)
 for k,im in enumerate(panels[start:start+10]):
  x=k%5*400;y=k//5*460;sheet.paste(im,(x+(400-im.width)//2,y+28));dr.text((x+8,y+7),str(start+k+1),fill='white')
 sheet.save(out/f'Sheet {start+1}.jpg')
(out/'Results.json').write_text(json.dumps(rows,indent=2));print('Evaluated',len(rows),'mean ms',1000*np.mean([r['seconds'] for r in rows]))
