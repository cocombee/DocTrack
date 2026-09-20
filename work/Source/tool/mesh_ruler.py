"""Straight-line constraints on one smooth backward mesh; no pixel synthesis."""
import cv2
import numpy as np


def text_rulers(gray, features):
    """Estimate baselines, excluding accents, descenders and mixed text rows."""
    h,w=gray.shape
    binary=cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,max(31,(min(h,w)//20)|1),15)
    _,_,stats,centers=cv2.connectedComponentsWithStats(binary)
    keep=(stats[:,4]>=8)&(stats[:,3]>=3)&(stats[:,3]<h*.05)&(stats[:,2]<w*.2)
    stats,centers=stats[keep],centers[keep]
    result=[]
    if not len(centers):return result
    for line in features.high_confidence_lines:
        p=line.centers
        if len(p)<10 or np.ptp(p[:,0])<w*.35:continue
        ids=np.argmin(np.sum((p[:,None,:]-centers[None,:,:])**2,axis=2),axis=1)
        valid=np.linalg.norm(p-centers[ids],axis=1)<2
        p=np.column_stack([p[valid,0],stats[ids[valid],1]+stats[ids[valid],3]-1])
        if len(p)<10:continue
        x=p[:,0]/(w-1);y=p[:,1]/(h-1)
        design=np.column_stack([x*x,x,np.ones(len(x))]);weight=np.ones(len(x))
        for _ in range(5):
            c=np.linalg.lstsq(design*weight[:,None],y*weight,rcond=None)[0]
            err=y-design@c
            weight=np.minimum(1,1.5/(np.abs(err)*(h-1)+1e-6))
        clean=np.abs(err)*(h-1)<2.5
        if clean.mean()<.7 or np.std(err[clean])*(h-1)>1.5:continue
        # A mixed row or decorative arc is not a reliable paragraph baseline.
        xx=np.linspace(x.min(),x.max(),33);yy=np.polyval(c,xx)
        if np.ptp(yy)>.035 or np.max(np.abs(2*c[0]*xx+c[1]))>.15:continue
        result.append(np.column_stack([xx,yy]))
    return result


def ruler_mesh(horizontal, vertical=(), prior=None, n=33):
    """Least-squares ruler constraints plus bending and displacement penalties.

    Each guide is (observed points, confidence). Separate x/y equations share
    the same lattice. Weak priors preserve areas with no reliable evidence.
    """
    u,v=np.meshgrid(np.linspace(0,1,n),np.linspace(0,1,n))
    identity=np.stack([u,v],-1)
    base=identity if prior is None else prior
    out=base.copy()
    for axis,guides in ((1,horizontal),(0,vertical)):
        if not guides:continue
        rr=[];cc=[];values=[];rhs=[]
        def equation(indices,weights,target,importance):
            row=len(rhs);rr.extend([row]*len(indices));cc.extend(indices)
            values.extend(np.array(weights)*importance);rhs.append(target*importance)
        for points,confidence in guides:
            target=float(np.mean(points[:,axis]))
            for p in points:
                q=p.copy();q[axis]=target
                xy=np.clip(q*(n-1),0,n-1-1e-6);ix,iy=xy.astype(int);fx,fy=xy-[ix,iy]
                equation([iy*n+ix,iy*n+ix+1,(iy+1)*n+ix,(iy+1)*n+ix+1],
                    [(1-fx)*(1-fy),fx*(1-fy),(1-fx)*fy,fx*fy],p[axis]-target,
                    confidence*np.sqrt(33/len(points)))
        for y in range(n):
            for x in range(n):
                i=y*n+x
                equation([i],[1],base[y,x,axis]-identity[y,x,axis],.045)
                if 0<x<n-1:equation([i-1,i,i+1],[1,-2,1],0,.8)
                if 0<y<n-1:equation([i-n,i,i+n],[1,-2,1],0,.8)
        # Matrix-free preconditioned conjugate gradients: lightweight local
        # solve, without adding a runtime package or dense matrix allocation.
        rr=np.asarray(rr);cc=np.asarray(cc);values=np.asarray(values);rhs=np.asarray(rhs)
        def normal(v):
            rows=np.bincount(rr,weights=values*v[cc],minlength=len(rhs))
            return np.bincount(cc,weights=values*rows[rr],minlength=n*n)
        b=np.bincount(cc,weights=values*rhs[rr],minlength=n*n)
        diagonal=np.bincount(cc,weights=values**2,minlength=n*n)
        delta=(base[:,:,axis]-identity[:,:,axis]).reshape(-1).copy()
        residual=b-normal(delta);z=residual/diagonal;direction=z.copy();rz=residual@z
        for _ in range(500):
            if np.linalg.norm(residual)<1e-8:break
            nd=normal(direction);step=rz/max(direction@nd,1e-20)
            delta+=step*direction;residual-=step*nd
            z=residual/diagonal;next_rz=residual@z
            direction=z+(next_rz/max(rz,1e-20))*direction;rz=next_rz
        out[:,:,axis]=identity[:,:,axis]+delta.reshape(n,n)
    return out


def circular_landmarks(gray):
    """Closed, already near-circular marks are shape-preservation evidence.

    An arbitrary ellipse alone cannot establish an originally circular shape.
    """
    h,w=gray.shape
    edges=cv2.Canny(gray,70,160)
    contours,_=cv2.findContours(edges,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
    result=[]
    for contour in contours:
        if len(contour)<24:continue
        area=abs(cv2.contourArea(contour))
        if not max(80,h*w*.0004)<area<h*w*.035:continue
        (cx,cy),(a,b),angle=cv2.fitEllipse(contour)
        if min(a,b)/max(a,b)<.94:continue
        points=contour.reshape(-1,2).astype(float)
        radius=(a+b)/4
        if np.std(np.linalg.norm(points-[cx,cy],axis=1))/radius>.06:continue
        if any(np.linalg.norm(np.mean(p,axis=0)*[w-1,h-1]-[cx,cy])<radius*.5 for p in result):continue
        result.append(points[::max(1,len(points)//48)]/[w-1,h-1])
    return result
