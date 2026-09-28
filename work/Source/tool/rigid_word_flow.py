"""Integrate a bounded rigid rotation through surrounding unprinted stock.

The moving ink remains a rigid body. Other word envelopes and complete rule
traces have zero velocity. Callers must still validate the sampled mesh and
all independent ink; integration alone is not an acceptance certificate.
"""
import cv2
import numpy as np


def smooth(value):
    value=np.clip(value,0,1)
    return value*value*(3-2*value)


def backward(points, owned, fixed, guards, center, angle, pixel_size,
             *,padding=32,steps=32,constraint_padding=2,body_padding=.5,
             fixed_ink=()):
    scale=np.asarray(pixel_size,dtype=float)
    original=np.asarray(points,dtype=float)
    initial=original*scale
    result=initial.copy()
    owned=np.asarray(owned,dtype=float)*scale
    fixed=[np.asarray(box,dtype=float)*scale for box in fixed]
    guards=[np.asarray(g,dtype=float)*scale for g in guards]
    fixed_ink=[np.asarray(g,dtype=float)*scale for g in fixed_ink]
    center=np.asarray(center,dtype=float)
    if (not np.isfinite(result).all() or not np.isfinite(owned).all() or
            steps<8 or padding<=0 or constraint_padding<=0 or body_padding<0 or abs(angle)>.1):
        raise ValueError('Invalid local rigid flow')
    c,s=np.cos(angle),np.sin(angle)
    rotation=np.array([[c,-s],[s,c]])
    end=(owned-center)@rotation+center
    low=np.minimum(owned.min(0),end.min(0))-padding-3
    high=np.maximum(owned.max(0),end.max(0))+padding+3
    fixed=[box for box in fixed if np.all(box.max(0)>=low) and np.all(box.min(0)<=high)]
    # A subpixel raster provides a smooth distance from complete curved rules.
    factor=4
    origin=np.floor(low)-2
    shape=np.ceil((high-origin+2)*factor).astype(int)[::-1]
    blocked=np.zeros(shape,np.uint8)
    for ink in fixed_ink:
        if ink.ndim!=2 or ink.shape[1]!=2 or not np.isfinite(ink).all():
            raise ValueError('Invalid fixed glyph geometry')
        xy=np.rint((ink-origin)*factor).astype(int)
        good=np.all((xy>=0)&(xy<shape[::-1]),axis=1)
        # Retain the measured pixel bodies and a subpixel perimeter, without
        # treating the empty corners of a slanted word box as printed ink.
        for x,y in np.unique(xy[good],axis=0):
            cv2.circle(blocked,(int(x),int(y)),3,1,-1)
    for guide in guards:
        if not (np.all(guide.max(0)>=low) and np.all(guide.min(0)<=high)):
            continue
        cv2.polylines(blocked,[np.rint((guide-origin)*factor).astype(np.int32)],
                      False,1,3)
    distance=(cv2.distanceTransform(1-blocked,cv2.DIST_L2,cv2.DIST_MASK_PRECISE)/factor
              if blocked.any() else None)

    def velocity(query,time):
        c,s=np.cos(angle*time),np.sin(angle*time)
        body=(owned-center)@np.array([[c,-s],[s,c]])+center
        a,z=body.min(0)-body_padding,body.max(0)+body_padding
        delta=np.maximum(np.maximum(a-query,query-z),0)
        weight=np.prod(smooth(1-delta/padding),axis=1)
        for box in fixed:
            a,z=box.min(0),box.max(0)
            outside=np.maximum(np.maximum(a-query,query-z),0)
            weight*=smooth(np.linalg.norm(outside,axis=1)/constraint_padding)
        if distance is not None:
            xy=(query-origin)*factor
            base=np.floor(xy).astype(int);fraction=xy-base
            x,y=base.T;good=(x>=0)&(y>=0)&(x<distance.shape[1]-1)&(y<distance.shape[0]-1)
            gap=np.full(len(query),padding,dtype=float)
            a,b=fraction[good].T;xx,yy=x[good],y[good]
            # Do not use cv2.remap on an N-by-one query: a real dense mesh
            # can exceed OpenCV's signed-short destination dimension limit.
            gap[good]=(distance[yy,xx]*(1-a)*(1-b)+distance[yy,xx+1]*a*(1-b)+
                       distance[yy+1,xx]*(1-a)*b+distance[yy+1,xx+1]*a*b)
            weight*=smooth(gap/constraint_padding)
        relative=query-center
        return np.c_[angle*relative[:,1],-angle*relative[:,0]]*weight[:,None]

    dt=-1/steps
    for step in range(steps):
        t=1-step/steps
        a=velocity(result,t)
        b=velocity(result+dt*a/2,t+dt/2)
        c=velocity(result+dt*b/2,t+dt/2)
        d=velocity(result+dt*c,t+dt)
        result+=dt*(a+2*b+2*c+d)/6
    # Keep zero-motion coordinates bit-for-bit, including fractional rule
    # samples whose multiplication and division would otherwise round once.
    return original+(result-initial)/scale
