"""Local, narrow-band paper-material matte experiment. Never alters geometry."""
from pathlib import Path
import cv2,numpy as np,json
ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'Data';EVAL=ROOT/'Scanner Integration'
cv2.setNumThreads(2);cv2.setRNGSeed(0)

def paper_matte(image,segments):
    h,w=image.shape[:2];points=[]
    for s in segments:
        a=np.array([s['start']['x'],s['start']['y']]);b=np.array([s['end']['x'],s['end']['y']])
        if s.get('control1') and s.get('control2'):
            c=np.array([s['control1']['x'],s['control1']['y']]);d=np.array([s['control2']['x'],s['control2']['y']])
            for t in np.linspace(0,1,65)[:-1]:points.append((1-t)**3*a+3*(1-t)**2*t*c+3*(1-t)*t*t*d+t**3*b)
        else:points.append(a)
    polygon=np.zeros((h,w),np.uint8);cv2.fillPoly(polygon,[(np.array(points)*[w,h]-.5).round().astype(np.int32)],1)
    width=int(np.clip(round(min(h,w)*.01),4,10));kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(width*2+1,width*2+1))
    inner=cv2.erode(polygon,kernel);outer=cv2.dilate(polygon,kernel)
    labels=np.where(polygon,cv2.GC_PR_FGD,cv2.GC_PR_BGD).astype(np.uint8)
    labels[inner>0]=cv2.GC_FGD;labels[outer==0]=cv2.GC_BGD
    if not (labels==cv2.GC_BGD).any() or not inner.any():return image.copy(),0
    cv2.grabCut(image[:,:,:3].copy(),labels,None,np.zeros((1,65),np.float64),np.zeros((1,65),np.float64),4,cv2.GC_INIT_WITH_MASK)
    background=((labels==cv2.GC_BGD)|(labels==cv2.GC_PR_BGD)).astype(np.uint8)
    _,components=cv2.connectedComponents(background,connectivity=4)
    external=np.unique(components[(polygon==0)&(background>0)])
    removed=(polygon>0)&(inner==0)&np.isin(components,external)&(background>0)
    out=image.copy();out[removed,3]=0
    assert np.array_equal(out[inner>0],image[inner>0])
    assert np.array_equal(out[:,:,:3],image[:,:,:3])
    return out,int(removed.sum())

if __name__=='__main__':
    folder=EVAL/'Material Matte Source 49';folder.mkdir(exist_ok=True);rows=[]
    manifest=json.loads((DATA/'manifest.json').read_text())
    for i in [4,35]:
        e=manifest[i-1];source=EVAL/'Decoded Source Evidence 49'/f'{e["name"]}.png'
        image=cv2.imread(str(source),cv2.IMREAD_UNCHANGED)
        if image.shape[2]==3:image=np.dstack([image,np.full(image.shape[:2],255,np.uint8)])
        d=json.loads((EVAL/'Predictions Physical V18'/f'{e["name"]}.json').read_text())
        out,n=paper_matte(image,d['boundarySegments'])
        cv2.imwrite(str(folder/f'{e["name"]}.png'),out)
        rows.append({'index':i,'removed_pixels':n,'RGB_unchanged':True,'interior_locked':True})
    (folder/'Report.json').write_text(json.dumps(rows,indent=2));print(json.dumps(rows))
