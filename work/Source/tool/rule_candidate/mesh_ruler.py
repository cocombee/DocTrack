"""Straight-line constraints on one smooth backward mesh; no pixel synthesis."""
import cv2
import numpy as np


def text_rulers(gray, features):
    """Estimate baselines, excluding accents, descenders and mixed text rows."""
    h,w=gray.shape
    pixel_scale=max(1.,max(h,w)/640)
    binary=cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,max(31,(min(h,w)//20)|1),15)
    _,_,stats,centers=cv2.connectedComponentsWithStats(binary)
    keep=(stats[:,4]>=8)&(stats[:,3]>=3)&(stats[:,3]<h*.05)&(stats[:,2]<w*.2)
    stats,centers=stats[keep],centers[keep]
    result=[]
    if not len(centers):return result
    for line in features.lines:
        p=line.centers
        if len(p)<8 or np.ptp(p[:,0])<w*.12:continue
        ids=np.argmin(np.sum((p[:,None,:]-centers[None,:,:])**2,axis=2),axis=1)
        valid=np.linalg.norm(p-centers[ids],axis=1)<2
        p=np.column_stack([p[valid,0],stats[ids[valid],1]+stats[ids[valid],3]-1])
        if len(p)<8:continue
        x=p[:,0]/(w-1);y=p[:,1]/(h-1)
        design=np.column_stack([x*x,x,np.ones(len(x))]);weight=np.ones(len(x))
        for _ in range(5):
            c=np.linalg.lstsq(design*weight[:,None],y*weight,rcond=None)[0]
            err=y-design@c
            weight=np.minimum(1,1.5*pixel_scale/(np.abs(err)*(h-1)+1e-6))
        clean=np.abs(err)*(h-1)<2.5*pixel_scale
        if clean.mean()<.7 or np.std(err[clean])*(h-1)>1.5*pixel_scale:continue
        # A mixed row or decorative arc is not a reliable paragraph baseline.
        xx=np.linspace(x.min(),x.max(),33);yy=np.polyval(c,xx)
        strong=len(p)>=10 and np.ptp(x)>.2 and clean.mean()>=.85
        if np.ptp(yy)>(.10 if strong else .035) or np.max(np.abs(2*c[0]*xx+c[1]))>(.4 if strong else .15):continue
        result.append(np.column_stack([xx,yy]))
    return result


def ruler_mesh(horizontal, vertical=(), prior=None, n=33, prior_weight=.045):
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
                equation([i],[1],base[y,x,axis]-identity[y,x,axis],prior_weight)
                if 0<x<n-1:equation([i-1,i,i+1],[1,-2,1],0,.8)
                if 0<y<n-1:equation([i-n,i,i+n],[1,-2,1],0,.8)
        delta=_fit_displacement(rr,cc,values,rhs,
            (base[:,:,axis]-identity[:,:,axis]).reshape(-1),n)
        # Bound unsupported exterior extrapolation in the solve itself. Hard
        # clamping a finished mesh would flatten its derivative at the edge.
        for _ in range(3):
            positions=identity[:,:,axis].reshape(-1)+delta
            outside=np.flatnonzero((positions<-.024)|(positions>1.024))
            if not len(outside):break
            for i in outside:
                target=np.clip(positions[i],-.024,1.024)-identity[:,:,axis].reshape(-1)[i]
                equation([int(i)],[1],target,4.)
            delta=_fit_displacement(rr,cc,values,rhs,delta,n)
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


def paragraph_rulers(gray, features):
    """Repeated left margins in text columns supply local vertical evidence."""
    h,w=gray.shape
    binary=cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,max(31,(min(h,w)//20)|1),15)
    _,_,stats,centers=cv2.connectedComponentsWithStats(binary)
    keep=(stats[:,4]>=8)&(stats[:,3]>=3)&(stats[:,3]<h*.05)&(stats[:,2]<w*.2)
    stats,centers=stats[keep],centers[keep]
    if not len(centers):return []
    starts=[]
    for line in features.lines:
        p=line.centers
        if len(p)<8 or np.ptp(p[:,0])<w*.10:continue
        # A merged word's centroid can sit far inside a perfectly aligned
        # column. Use its ink boundary, not its centroid, as margin evidence.
        first=np.argmin(np.sum((centers-p[0])**2,axis=1))
        if np.linalg.norm(centers[first]-p[0])>2:continue
        starts.append([stats[first,0]/(w-1),np.median(p[:,1])/(h-1)])
    starts=np.asarray(starts)
    if len(starts)<5:return []
    starts=starts[np.argsort(starts[:,1])]
    chains=[]
    for p in starts:
        choices=[(abs(p[0]-c[-1][0]),i) for i,c in enumerate(chains)
            if .006<p[1]-c[-1][1]<.045 and abs(p[0]-c[-1][0])<.018]
        if choices:chains[min(choices)[1]].append(p)
        else:chains.append([p])
    result=[]
    for chain in chains:
        p=np.array(chain)
        if len(p)<6 or np.ptp(p[:,1])<.10:continue
        # Repeated margins must agree smoothly; indents/new columns end a chain.
        t=p[:,1];c=np.polyfit(t,p[:,0],2);residual=np.abs(np.polyval(c,t)-p[:,0])
        if np.percentile(residual,85)>.004 or np.ptp(p[:,0])>.07:continue
        tt=np.linspace(t.min(),t.max(),33)
        result.append(np.column_stack([np.polyval(c,tt),tt]))
    return result

def _fit_displacement(rr,cc,values,rhs,initial,n):
    rr=np.asarray(rr);cc=np.asarray(cc);values=np.asarray(values);rhs=np.asarray(rhs)
    def normal(v):
        rows=np.bincount(rr,weights=values*v[cc],minlength=len(rhs))
        return np.bincount(cc,weights=values*rows[rr],minlength=n*n)
    b=np.bincount(cc,weights=values*rhs[rr],minlength=n*n)
    diagonal=np.bincount(cc,weights=values**2,minlength=n*n)
    delta=initial.copy()
    residual=b-normal(delta);z=residual/diagonal;direction=z.copy();rz=residual@z
    for _ in range(500):
        if np.linalg.norm(residual)<1e-8:break
        nd=normal(direction);step=rz/max(direction@nd,1e-20)
        delta+=step*direction;residual-=step*nd
        z=residual/diagonal;next_rz=residual@z
        direction=z+(next_rz/max(rz,1e-20))*direction;rz=next_rz
    return delta

