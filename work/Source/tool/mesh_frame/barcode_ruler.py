"""Repeated parallel barcode bars constrain geometry without decoding content."""
import cv2
import numpy as np


def barcode_rulers(gray):
    h,w=gray.shape
    block=max(31,(min(h,w)//20)|1)
    binary=cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,block,15)
    count,labels,stats,centers=cv2.connectedComponentsWithStats(binary)
    ids=[i for i in range(1,count) if stats[i,3]>=max(12,h*.012) and stats[i,3]<h*.22
         and stats[i,3]>3*stats[i,2] and stats[i,4]>.25*stats[i,2]*stats[i,3]]
    used=set();result=[]
    for seed in ids:
        if seed in used:continue
        x,y,bw,bh,_=stats[seed];group=[]
        for i in ids:
            xx,yy,ww,hh,_=stats[i]
            overlap=min(y+bh,yy+hh)-max(y,yy)
            if overlap>.75*max(bh,hh) and abs(centers[i,0]-centers[seed,0])<w*.8:group.append(i)
        if len(group)<12:continue
        group.sort(key=lambda i:centers[i,0]);runs=[[]]
        for i in group:
            if runs[-1] and stats[i,0]-(stats[runs[-1][-1],0]+stats[runs[-1][-1],2])>bh:runs.append([])
            runs[-1].append(i)
        group=next((r for r in runs if seed in r),[])
        if len(group)<12:continue
        left=min(stats[i,0] for i in group);right=max(stats[i,0]+stats[i,2] for i in group)
        top=max(stats[i,1] for i in group);bottom=min(stats[i,1]+stats[i,3] for i in group)
        if right-left<w*.15 or bottom-top<bh*.7:continue
        region=labels[top:bottom,left:right]
        # Ordinary text must not qualify merely because some letters are tall.
        foreground=np.count_nonzero(region)
        if foreground==0 or np.count_nonzero(np.isin(region,group))/foreground<.85:continue
        used.update(group)
        # Sample the centre of each real bar, retaining any photographed bend.
        for i in group:
            xx,yy,ww,hh,_=stats[i];ys=np.linspace(yy+hh*.08,yy+hh*.92,17)
            p=[]
            for row in ys:
                hits=np.flatnonzero(labels[round(row),xx:xx+ww]==i)
                if len(hits):p.append([xx+float(hits.mean()),row])
            if len(p)>=14:result.append(np.asarray(p)/[w-1,h-1])
    return result

def barcode_envelopes(gray):
    """Return coherent top/bottom barcode envelope guides for horizontal straightening."""
    h,w=gray.shape
    bars=barcode_rulers(gray)
    if len(bars)<12:return []
    top=np.array([[p[:,0].mean(),p[:,1].min()] for p in bars])
    bot=np.array([[p[:,0].mean(),p[:,1].max()] for p in bars])
    lo,hi=np.quantile(top[:,0],[.05,.95])
    keep=(top[:,0]>=lo)&(top[:,0]<=hi)
    if keep.sum()<8:return []
    x=np.linspace(top[keep,0].min(),top[keep,0].max(),33)
    return [np.column_stack([x,np.full(33,np.median(top[keep,1]) )]),np.column_stack([x,np.full(33,np.median(bot[keep,1]) )])]
