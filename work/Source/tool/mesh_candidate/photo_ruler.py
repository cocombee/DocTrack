"""Experimental colour-boundary refinement inside an established photo panel."""
import cv2,numpy as np,os

def refine_photo_frames(bgr,horizontal,vertical):
    if len(horizontal)!=2 or len(vertical)!=2:return None
    h,w=bgr.shape[:2];lab=cv2.cvtColor(bgr,cv2.COLOR_BGR2LAB).astype(np.float32)
    chroma=cv2.GaussianBlur(lab[:,:,1:],(3,3),0)
    radius=max(8,round(min(h,w)*.035));offsets=np.arange(-round(radius*.3),radius+1,dtype=float)
    result=[]
    for guide,axis,direction in [(horizontal[0],1,1),(horizontal[1],1,-1),(vertical[0],0,1),(vertical[1],0,-1)]:
        xy=guide*[w-1,h-1];normal=np.zeros(2);normal[axis]=direction
        ref=xy-normal*max(8,radius*.6)
        stock=cv2.remap(chroma,ref[:,0].astype(np.float32).reshape(-1,1),ref[:,1].astype(np.float32).reshape(-1,1),cv2.INTER_LINEAR)[:,0,:]
        samples=xy[:,None,:]+normal*offsets[None,:,None]
        values=cv2.remap(chroma,samples[:,:,0].astype(np.float32),samples[:,:,1].astype(np.float32),cv2.INTER_LINEAR)
        strength=np.linalg.norm(values-stock[:,None,:],axis=2)
        picked=np.full(len(xy),np.nan)
        for i,row in enumerate(strength):
            if np.median(row[:3])>6:continue
            for j in range(3,len(offsets)-4):
                if np.min(row[j:j+4])>8 and np.max(row[j-3:j-1])<7:
                    picked[i]=offsets[j];break
        good=np.isfinite(picked)
        if os.environ.get('DOCTRACK_PHOTO_DEBUG'):print('edge',axis,direction,'support',good.mean(),'maxchroma',np.percentile(strength.max(1),[10,50,90]).tolist())
        if good.mean()<.7:
            result.append(None);continue
        ids=np.arange(len(xy));smooth=np.interp(ids,ids[good],picked[good])
        smooth=np.array([np.median(smooth[max(0,i-2):i+3]) for i in ids])
        if np.percentile(abs(picked[good]-smooth[good]),90)>3:
            result.append(None);continue
        # A long unsupported interval cannot be invented through image texture.
        gaps=np.diff(np.r_[-1,ids[good],len(xy)])
        if gaps.max()>len(xy)*.18:
            result.append(None);continue
        result.append((xy+normal*smooth[:,None])/[w-1,h-1])
    if not any(p is not None for p in result):return None
    original=horizontal+vertical
    refined=[old if new is None else new for old,new in zip(original,result)]
    return refined[:2],refined[2:]
