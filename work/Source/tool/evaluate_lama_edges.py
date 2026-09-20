"""Local, evaluation-only LaMa edge fill. Never used by the app automatically."""
from pathlib import Path
import cv2,numpy as np,json,time,hashlib
ROOT=Path(__file__).resolve().parents[2]/'Scanner Integration'
folder=ROOT/'Dewarping Evaluation'/'LaMa'
cv2.setNumThreads(2)
net=cv2.dnn.readNetFromONNX(str(folder/'inpainting_lama_2025jan.onnx'))
rows=[]
for name in ('Image 2','Pattern'):
    if name=='Pattern':
        y,x=np.mgrid[:512,:512]
        rgb=np.stack([225+12*np.sin(x/12+y/18),230+10*np.sin(x/12+y/18),235+8*np.sin(x/12+y/18)],-1).astype(np.uint8)
    else:
        rgba=cv2.imread(str(ROOT/'Flattening Preview V2'/f'{name}.png'),cv2.IMREAD_UNCHANGED)
        rgb=rgba[:,:,:3]
    h,w=rgb.shape[:2];mask=np.zeros((h,w),np.uint8)
    mask[:max(3,h//100),:]=255
    mask[:h//20,:w//15]=255
    damaged=rgb.copy();damaged[mask>0]=0
    start=time.perf_counter()
    net.setInput(cv2.dnn.blobFromImage(damaged,1/255,(512,512),swapRB=False),'image')
    net.setInput((cv2.dnn.blobFromImage(mask,1,(512,512))>0).astype(np.float32),'mask')
    predicted=net.forward()[0].transpose(1,2,0)
    predicted=cv2.resize(predicted,(w,h)).clip(0,255).astype(np.uint8)
    # Exact outside-mask compositing is mandatory, even during evaluation.
    out=rgb.copy();out[mask>0]=predicted[mask>0]
    assert np.array_equal(out[mask==0],rgb[mask==0])
    cv2.imwrite(str(folder/f'{name} Missing.png'),damaged)
    cv2.imwrite(str(folder/f'{name} Mask.png'),mask)
    cv2.imwrite(str(folder/f'{name} Filled.png'),out)
    rows.append({'name':name,'seconds':time.perf_counter()-start,
      'masked_mae':float(np.abs(out[mask>0].astype(float)-rgb[mask>0]).mean()),
      'outside_mask_unchanged':True,
      'automatic_use':False,'text_safety':'Not guaranteed by model; evaluation only'})
(folder/'Evaluation.json').write_text(json.dumps(rows,indent=2))
(folder/'Provenance.json').write_text(json.dumps({
 'upstream':'https://github.com/advimman/lama',
 'model_source':'https://huggingface.co/opencv/inpainting_lama',
 'model_sha256':hashlib.sha256((folder/'inpainting_lama_2025jan.onnx').read_bytes()).hexdigest(),
 'status':'local evaluation only; not integrated',
 'license_file':'LICENSE'},indent=2))
print(json.dumps(rows))
