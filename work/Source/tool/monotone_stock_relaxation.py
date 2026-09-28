"""Smooth stock between rigid ink bands without changing their constraints."""
import cv2
import numpy as np


def relax(field, coordinates, column_knots, iterations=12):
    result=field.copy()
    plans=[]
    for knots in column_knots:
        if knots is None:
            plans.append(None)
            continue
        dest,source=knots.T
        slices=[]
        for i in range(len(dest)-1):
            if abs((source[i+1]-dest[i+1])-(source[i]-dest[i]))<1e-10:
                continue
            first,last=np.searchsorted(coordinates,[dest[i],dest[i+1]])
            slices.append((i,slice(first,last)))
        plans.append((dest,source,np.interp(coordinates,dest,source),slices))
    for _ in range(iterations):
        smooth=cv2.GaussianBlur(result,(0,1),sigmaX=1.5,
                                borderType=cv2.BORDER_REPLICATE)
        for col,plan in enumerate(plans):
            if plan is None:
                continue
            dest,source,baseline,slices=plan
            projected=baseline.copy()
            boundary=np.interp(dest,coordinates,smooth[:,col])
            for i,inside in slices:
                # Equal endpoint displacement is a complete rigid band.
                # All other intervals contain only unconstrained stock.
                span=boundary[i+1]-boundary[i]
                if span<=0:
                    continue
                fraction=(smooth[inside,col]-boundary[i])/span
                projected[inside]=source[i]+fraction*(source[i+1]-source[i])
            result[:,col]=projected
    return result


def relax_tangent_stock(candidate,field,boxes,shifts,traces,region,iterations):
    rows,cols=field.shape
    free=np.zeros(field.shape,np.uint8)
    x0,y0,x1,y1=region
    free[max(0,int(np.ceil(y0*(rows-1)))):min(rows,int(y1*(rows-1))+1),
         max(0,int(np.ceil(x0*(cols-1)))):min(cols,int(x1*(cols-1))+1)]=1
    for box,shift in zip(boxes,shifts):
        lo,hi=box.min(0),box.max(0)
        left=max(0,int(np.floor(lo[0]*(cols-1)))-2)
        right=min(cols,int(np.ceil(hi[0]*(cols-1)))+3)
        top=max(0,int(np.floor((lo[1]+shift)*(rows-1)))-2)
        bottom=min(rows,int(np.ceil((hi[1]+shift)*(rows-1)))+3)
        free[top:bottom,left:right]=0
    y=np.linspace(0,1,rows)
    for trace in traces:
        destinations=[]
        for x,sy in trace:
            xx=np.clip(x*(cols-1),0,cols-1-1e-9)
            col=int(xx);fraction=xx-col
            forward=field[:,col]*(1-fraction)+field[:,col+1]*fraction
            destinations.append([x,np.interp(sy,forward,y)])
        cv2.polylines(free,[np.rint(np.asarray(destinations)*[cols-1,rows-1]).astype(np.int32)],
                      False,0,5)
    result=candidate.copy()
    weight=free*np.clip(abs(field-y[:,None])*(rows-1),0,1)
    dx,dy=np.diff(candidate,axis=1),np.diff(candidate,axis=0)
    quality=np.minimum.reduce([(a[:,:,0]*b[:,:,1]-a[:,:,1]*b[:,:,0])*(rows-1)*(cols-1)
        for a in (dx[:-1],dx[1:]) for b in (dy[:,:-1],dy[:,1:])])
    repair=np.zeros(field.shape,np.uint8)
    failed=(quality<=.2).astype(np.uint8)
    repair[:-1,:-1]|=failed;repair[1:,:-1]|=failed
    repair[:-1,1:]|=failed;repair[1:,1:]|=failed
    repair=cv2.dilate(repair,np.ones((13,13),np.uint8))
    weight*=repair
    for _ in range(iterations):
        smooth=cv2.GaussianBlur(result[:,:,0],(0,0),sigmaX=1.5,sigmaY=1.5,
                                borderType=cv2.BORDER_REPLICATE)
        result[:,:,0]+=weight*(smooth-result[:,:,0])
    return result
