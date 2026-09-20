"""Isolated DocRes geometry evaluation. No learned RGB output is used."""
from pathlib import Path
import sys,json,time,hashlib,importlib.util,os
import numpy as np,cv2,torch
ROOT=Path(__file__).resolve().parents[2]/'Scanner Integration'
REPO=ROOT/'Dewarping Evaluation'/'DocRes'
spec=importlib.util.spec_from_file_location('docres_restormer',REPO/'models'/'restormer_arch.py')
arch=importlib.util.module_from_spec(spec);spec.loader.exec_module(arch)
sys.path.insert(0,str(Path(__file__).resolve().parent/'mesh_v49'))
from dewarp_mesh import calibrate_corners,valid,line_error,IDENTITY,extract_text_features
from mesh_ruler import text_rulers
cv2.setNumThreads(2);torch.set_num_threads(2)
model=arch.Restormer(inp_channels=6,out_channels=3,dim=48,num_blocks=[2,3,3,4],num_refinement_blocks=4,heads=[1,2,4,8],ffn_expansion_factor=2.66,bias=False,LayerNorm_type='WithBias',dual_pixel_task=True)
checkpoint=REPO/'checkpoints'/'docres.pkl'
state=torch.load(checkpoint,map_location='cpu',weights_only=True)['model_state']
model.load_state_dict({k.removeprefix('module.'):v for k,v in state.items()});model.eval()
padding=float(os.environ.get('DOCRES_PADDING','0'))
source_dir=os.environ.get('DOCRES_SOURCE_DIR')
folder=ROOT/('DocRes Source Geometry Evaluation 49' if source_dir else ('DocRes Padded Geometry Evaluation 49' if padding else 'DocRes Geometry Evaluation 49'));folder.mkdir(exist_ok=True)
base=np.stack(np.meshgrid(np.arange(256),np.arange(256)),-1).astype(np.float32)/256
rows=[]
for index in [2,4,17,25,35,38,46]:
    source=(Path(source_dir)/f'Image {index} Isolated.png') if source_dir else ROOT/'Full Resolution Candidate 49'/f'Image {index} Preliminary.png'
    rgba=cv2.imread(str(source),cv2.IMREAD_UNCHANGED)
    original=rgba.copy()
    if padding:
        ph,pw=[round(n*padding) for n in rgba.shape[:2]]
        rgba=cv2.copyMakeBorder(rgba,ph,ph,pw,pw,cv2.BORDER_CONSTANT,value=(0,0,0,0))
    h,w=rgba.shape[:2]
    bgr=rgba[:,:,:3].copy();mask=rgba[:,:,3];bgr[mask==0]=0
    rgb=cv2.resize(bgr,(256,256)).astype(np.float32)/255
    prompt=np.dstack([base,cv2.resize(mask,(256,256)).astype(np.float32)/255])
    inp=torch.from_numpy(np.dstack([rgb,prompt]).transpose(2,0,1)).unsqueeze(0)
    start=time.perf_counter()
    with torch.inference_mode(): predicted=model(inp)[0,:2].permute(1,2,0).numpy()+base
    for _ in range(15):predicted=cv2.blur(predicted,(3,3),borderType=cv2.BORDER_REPLICATE)
    grid=calibrate_corners(predicted)
    # Pixel-coordinate conventions match upstream remap, but all colour comes
    # exclusively from the original isolated document.
    for name,g in [('Raw',predicted),('Calibrated',grid)]:
        target=rgba if name=='Raw' else original
        th,tw=target.shape[:2]
        xx,yy=np.meshgrid(np.linspace(0,g.shape[1]-1,original.shape[1]),np.linspace(0,g.shape[0]-1,original.shape[0]))
        full=cv2.remap(g.astype(np.float32),xx.astype(np.float32),yy.astype(np.float32),cv2.INTER_LINEAR)
        result=cv2.remap(target,(full[:,:,0]*tw).astype(np.float32),(full[:,:,1]*th).astype(np.float32),cv2.INTER_LINEAR)
        cv2.imwrite(str(folder/f'Image {index} {name}.png'),result)
    mask=original[:,:,3]
    gray=cv2.cvtColor(original[:,:,:3],cv2.COLOR_BGR2GRAY);gray[mask==0]=255
    horizontal=text_rulers(gray,extract_text_features(gray,mask=mask));vertical=text_rulers(gray.T,extract_text_features(gray.T,mask=mask.T))
    def metric(g):
        trans=np.transpose(g,(1,0,2))[:,:,::-1]
        return [line_error(g,horizontal),line_error(trans,vertical)]
    row={'raw_range':[float(predicted.min()),float(predicted.max())],'input_range':[float(inp.min()),float(inp.max())],'index':index,'seconds':time.perf_counter()-start,'valid':valid(grid,edge_extension=True),'before':metric(IDENTITY),'after':metric(grid),'horizontal':len(horizontal),'vertical':len(vertical)}
    rows.append(row);print(json.dumps(row),flush=True)
    (folder/f'Image {index} Mesh.json').write_text(json.dumps({'rows':33,'columns':33,'grid':grid.reshape(-1,2).tolist(),'method':'DocRes geometry evaluation'}))
(folder/'Report.json').write_text(json.dumps(rows,indent=2))
(folder/'Provenance.json').write_text(json.dumps({'repo':'https://github.com/ZZZHANG-jx/DocRes','checkpoint':'https://huggingface.co/spaces/qubvel-hf/documents-restoration/resolve/main/checkpoints/docres.pkl','sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),'license':'MIT; upstream LICENSE retained','torch_load':'weights_only=True','mode':'geometry only, not integrated','source_mask':'existing accepted alpha; MBD not used'},indent=2))
