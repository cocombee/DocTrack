"""Experimental text-band rulers, independent of broken glyph components."""
import cv2,numpy as np

def density_rulers(gray):
    h,w=gray.shape
    binary=cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,max(31,(min(h,w)//20)|1),15)
    _,_,stats,_=cv2.connectedComponentsWithStats(binary)
    glyph=stats[(stats[:,4]>8)&(stats[:,3]>3)&(stats[:,3]<h*.05)&(stats[:,2]<w*.08)]
    if len(glyph)<40:return []
    height=float(np.average(glyph[:,3],weights=glyph[:,4]))
    density=cv2.GaussianBlur(binary.astype(np.float32)/255,(0,0),sigmaX=max(3,height*.8),sigmaY=max(1,height*.27))
    # A text band needs whitespace above and below; broad shaded folds do not.
    offset=max(3,round(height))
    signal=density-(np.roll(density,offset,0)+np.roll(density,-offset,0))*.5
    profile=np.maximum(signal[:,int(.05*w):int(.95*w)],0).mean(1)
    seeds=[]
    for y in np.argsort(profile)[::-1]:
        if profile[y]<.015:break
        if y<h*.03 or y>h*.97 or any(abs(y-s)<height*1.1 for s in seeds):continue
        seeds.append(int(y))
    result=[]
    x0,x1=int(.03*w),int(.97*w)
    for seed in seeds:
        lo=max(offset,int(seed-height*1.8));hi=min(h-offset,int(seed+height*1.8))
        if hi-lo<5:continue
        energy=signal[lo:hi,x0:x1]-.00025*np.abs(np.arange(lo,hi)[:,None]-seed)
        cost=-energy[:,0];back=np.zeros(energy.shape,np.int16)
        for x in range(1,energy.shape[1]):
            options=np.stack([np.r_[cost[0]+1,cost[:-1]+.004],cost,np.r_[cost[1:]+.004,cost[-1]+1]])
            choice=np.argmin(options,0);back[:,x]=np.arange(len(cost))+choice-1
            cost=np.min(options,0)-energy[:,x]
        path=np.zeros(energy.shape[1],np.int32);path[-1]=np.argmin(cost)
        for x in range(len(path)-1,0,-1):path[x-1]=back[path[x],x].clip(0,len(cost)-1)
        xx=np.arange(x0,x1);yy=path+lo
        strong=signal[yy,xx]>.03
        if strong.mean()<.32:continue
        left,right=np.percentile(xx[strong],[2,98])
        if right-left<w*.30:continue
        # Discard a path that switches between adjacent text bands.
        sample=np.linspace(left,right,49);trace=np.interp(sample,xx,yy)
        if np.percentile(abs(np.diff(trace)),95)>height*.6:continue
        points=np.column_stack([sample,trace])/[w-1,h-1]
        if any(np.median(abs(np.interp(sample,r[:,0]*(w-1),r[:,1]*(h-1))-trace))<height*.6 for r in result):continue
        result.append(points)
    return sorted(result,key=lambda r:r[:,1].mean())
