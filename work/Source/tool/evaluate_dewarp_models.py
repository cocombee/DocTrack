"""Local comparison only. No candidate modifies accepted detection geometry."""
from pathlib import Path
import sys,json,time
import cv2,numpy as np
root=Path(__file__).resolve().parents[2]/'Scanner Integration'
out=root/'Dewarping Evaluation'/'Results';out.mkdir(exist_ok=True)
sys.path.insert(0,str(root/'Dewarping Evaluation'/'UVDoc'))
import torch
from model import UVDocnet
torch.set_num_threads(2)
model=UVDocnet(num_filter=32,kernel_size=5)
checkpoint=torch.load(root/'Dewarping Evaluation'/'UVDoc'/'model'/'best_model.pkl',map_location='cpu',weights_only=True)
model.load_state_dict(checkpoint['model_state']);model.eval()
report=[]
for n in [17,25,28,29,35,43]:
    source=cv2.imread(str(root/'Flattening Preview V1'/f'Image {n}.png'),cv2.IMREAD_UNCHANGED)
    alpha=source[:,:,3:4]/255
    rgb=cv2.cvtColor(source[:,:,:3],cv2.COLOR_BGR2RGB)*alpha+255*(1-alpha)
    inp=torch.from_numpy(cv2.resize(rgb.astype(np.float32)/255,(488,712)).transpose(2,0,1)).unsqueeze(0)
    start=time.time()
    with torch.inference_mode(): grid,_=model(inp)
    coarse=grid[0].permute(1,2,0).numpy()
    h,w=source.shape[:2];dense=cv2.resize(coarse,(w,h))
    result=cv2.remap(source,((dense[:,:,0]+1)*(w-1)/2).astype(np.float32),((dense[:,:,1]+1)*(h-1)/2).astype(np.float32),cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
    cv2.imwrite(str(out/f'UVDoc {n}.png'),result)
    np.save(out/f'UVDoc {n} Mesh.npy',coarse)
    report.append({'image':n,'seconds':time.time()-start})
    print(report[-1],flush=True)
(root/'Dewarping Evaluation'/'UVDoc Evaluation.json').write_text(json.dumps(report,indent=2))
# Text-and-line model: compare one curved paper, keeping input alpha in rendering.
sys.path.insert(0,str(root/'Dewarping Evaluation'/'Text Line Dewarp'/'src'))
import dewarping.dewarp as module
from dewarping.options import make_options
for n in [28,25]:
    source=cv2.imread(str(root/'Flattening Preview V1'/f'Image {n}.png'),cv2.IMREAD_UNCHANGED)
    mask=source[:,:,3];alpha=mask[:,:,None]/255
    bgr=(source[:,:,:3]*alpha+255*(1-alpha)).astype(np.uint8)
    original_render=module._render
    def render(img,*args,**kwargs):return original_render(source,*args,**kwargs)
    module._render=render
    start=time.time()
    result=module.dewarp_image(bgr,debug=True,verbose=True,opts=make_options('fast',proc_max_side=800,max_nfev=80,n_outlier_iter=1,use_page_boundary=False))
    module._render=original_render
    cv2.imwrite(str(out/f'Text Lines {n}.png'),result.rectified)
    if result.debug_image is not None:cv2.imwrite(str(out/f'Text Evidence {n}.png'),result.debug_image)
    print({'text_model_image':n,'seconds':time.time()-start},flush=True)
