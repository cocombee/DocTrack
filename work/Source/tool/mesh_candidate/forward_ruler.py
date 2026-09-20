"""Experimental relative-baseline fit in observed image coordinates."""
import numpy as np
from mesh_ruler import _fit_displacement

def forward_ruler_mesh(horizontal,vertical=(),prior=None,n=33,prior_weight=.005):
    u,v=np.meshgrid(np.linspace(0,1,n),np.linspace(0,1,n));identity=np.stack([u,v],-1)
    forward=identity.copy()
    def terms(p):
        xy=np.clip(p*(n-1),0,n-1-1e-6);ix,iy=xy.astype(int);fx,fy=xy-[ix,iy]
        return [iy*n+ix,iy*n+ix+1,(iy+1)*n+ix,(iy+1)*n+ix+1],[(1-fx)*(1-fy),fx*(1-fy),(1-fx)*fy,fx*fy]
    for axis,guides in ((1,horizontal),(0,vertical)):
        if not guides:continue
        rr=[];cc=[];vals=[];rhs=[]
        def eq(ids,weights,target,weight):
            r=len(rhs);rr.extend([r]*len(ids));cc.extend(ids);vals.extend(np.asarray(weights)*weight);rhs.append(target*weight)
        for points,confidence in guides:
            reference=points[len(points)//2];ri,rw=terms(reference)
            for p in points:
                pi,pw=terms(p)
                eq(pi+ri,pw+[-z for z in rw],reference[axis]-p[axis],confidence*np.sqrt(33/len(points)))
        for y in range(n):
            for x in range(n):
                i=y*n+x
                eq([i],[1],0,prior_weight)
                if 0<x<n-1:eq([i-1,i,i+1],[1,-2,1],0,.8)
                if 0<y<n-1:eq([i-n,i,i+n],[1,-2,1],0,.8)
        # Relative line constraints alone admit an affine collapse of the
        # corrected axis. Preserve its overall scale and discourage local
        # compression while allowing shear across the orthogonal axis.
        ends=list(range(n)) if axis==1 else [y*n for y in range(n)]
        opposite=[i+n*(n-1) for i in ends] if axis==1 else [i+n-1 for i in ends]
        eq(ends+opposite,[-1/n]*n+[1/n]*n,0,8.)
        for y in range(n):
            for x in range(n):
                i=y*n+x
                if axis==1 and y+1<n:eq([i,i+n],[-1,1],0,.3)
                if axis==0 and x+1<n:eq([i,i+1],[-1,1],0,.3)
        delta=_fit_displacement(rr,cc,vals,rhs,np.zeros(n*n),n)
        forward[:,:,axis]+=delta.reshape(n,n)
    # Invert the forward coordinate field to the renderer's backward map.
    def sample(field,p):
        xy=p*(n-1);lo=np.clip(np.floor(xy).astype(int),0,n-2);f=xy-lo;x=lo[:,0];y=lo[:,1];a=f[:,0,None];b=f[:,1,None]
        return (1-a)*(1-b)*field[y,x]+a*(1-b)*field[y,x+1]+(1-a)*b*field[y+1,x]+a*b*field[y+1,x+1]
    gx=np.gradient(forward,1/(n-1),axis=1);gy=np.gradient(forward,1/(n-1),axis=0)
    target=identity.reshape(-1,2);guess=target.copy()
    for _ in range(16):
        delta=sample(forward,guess)-target;a=sample(gx,guess);b=sample(gy,guess);det=a[:,0]*b[:,1]-a[:,1]*b[:,0]
        if np.any(abs(det)<1e-8):return identity
        guess-=np.stack([(b[:,1]*delta[:,0]-b[:,0]*delta[:,1])/det,(-a[:,1]*delta[:,0]+a[:,0]*delta[:,1])/det],1)
        if not np.isfinite(guess).all() or np.max(abs(guess))>3:return identity
    grid=guess.reshape(n,n,2)
    return grid
