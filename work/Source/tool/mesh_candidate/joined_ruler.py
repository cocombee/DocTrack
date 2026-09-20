"""Reconnect separated text fragments only when their baseline tangents agree."""
import cv2
import numpy as np


def joined_text_rulers(gray, features):
    h,w=gray.shape
    binary=cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,max(31,(min(h,w)//20)|1),15)
    _,_,stats,centers=cv2.connectedComponentsWithStats(binary)
    keep=(stats[:,4]>=8)&(stats[:,3]>=3)&(stats[:,3]<h*.05)&(stats[:,2]<w*.2)
    stats,centers=stats[keep],centers[keep]
    if not len(centers):return []
    fragments=[]
    for line in features.lines:
        p=line.centers
        if len(p)<4 or np.ptp(p[:,0])<w*.07:continue
        ids=np.argmin(np.sum((p[:,None,:]-centers[None,:,:])**2,axis=2),axis=1)
        good=np.linalg.norm(p-centers[ids],axis=1)<2
        ids=ids[good];p=p[good]
        if len(p)<4:continue
        height=float(np.median(stats[ids,3]));x=p[:,0];y=stats[ids,1]+stats[ids,3]-1
        # A fragment's direction is more reliable than a quadratic extrapolation.
        design=np.column_stack([x-x.mean(),np.ones(len(x))]);weight=np.ones(len(x))
        for _ in range(5):
            fit=np.linalg.lstsq(design*weight[:,None],y*weight,rcond=None)[0]
            err=y-design@fit;weight=np.minimum(1,.12*height/(np.abs(err)+1e-6))
        clean=np.abs(err)<.25*height
        if clean.mean()<.7 or abs(fit[0])>.35:continue
        xx=np.linspace(x.min(),x.max(),17);yy=(xx-x.mean())*fit[0]+fit[1]
        fragments.append((np.column_stack([xx,yy]),height,fit[0],len(p)))
    return join_fragments(fragments,w,h)


def join_fragments(fragments,w,h):
    edges=[]
    for i,(a,ha,sa,na) in enumerate(fragments):
        for j,(b,hb,sb,nb) in enumerate(fragments):
            gap=b[0,0]-a[-1,0]
            if not 0<gap<min(.22*w,10*max(ha,hb)):continue
            if max(ha,hb)>1.5*min(ha,hb) or abs(sa-sb)>.18:continue
            dy=b[0,1]-a[-1,1]
            error=max(abs(dy-sa*gap),abs(dy-sb*gap))
            if error>.4*min(ha,hb):continue
            edges.append((error/min(ha,hb)+gap/w,i,j))
    successor={};predecessor={}
    for _,i,j in sorted(edges):
        if i in successor or j in predecessor:continue
        successor[i]=j;predecessor[j]=i
    result=[]
    for i in successor:
        if i in predecessor:continue
        chain=[i]
        while chain[-1] in successor:chain.append(successor[chain[-1]])
        p=np.concatenate([fragments[k][0] for k in chain])
        if np.ptp(p[:,0])<.25*w or sum(fragments[k][3] for k in chain)<12:continue
        # Keep observed piecewise bends; do not force a single global parabola.
        result.append(p/[w-1,h-1])
    return result
