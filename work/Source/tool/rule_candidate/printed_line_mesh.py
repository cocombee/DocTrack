"""Conservative printed-rule geometry; reads pixels but never synthesizes them."""
import cv2
import numpy as np

def _extract_rules_at_scale(gray):
    h,w=gray.shape
    binary=cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,31,12)
    segments=cv2.HoughLinesP(binary,1,np.pi/720,threshold=max(35,w//7),
        minLineLength=w*.35,maxLineGap=max(8,w//22))
    if segments is None:return []
    seeds=[]
    for x0,y0,x1,y1 in sorted(segments.reshape(-1,4), key=lambda s: -abs(int(s[2])-int(s[0]))):
        if abs(x1-x0)<w*.35:continue
        slope=(y1-y0)/(x1-x0);intercept=y0-slope*x0
        mid=intercept+slope*w*.5
        if abs(slope)>.35 or not h*.025<mid<h*.975:continue
        if any(abs(mid-m)<h*.015 for m,_,_,_,_ in seeds):continue
        seeds.append((mid,slope,intercept,min(x0,x1),max(x0,x1)))
    rules=[]
    # Follow each thin printed rule through bends rather than extrapolating a
    # straight Hough segment across a fold into a neighbouring text row.
    blur=cv2.GaussianBlur(gray.astype(np.float32),(0,0),3)
    contrast=np.maximum(0,blur-gray)
    # A rule is thin and has clear stock immediately above and below.
    # Penalize text before tracing, rather than accepting a smooth path through it.
    blank=np.ones(gray.shape,np.float32)
    for offset in (-4,-3,3,4):
        blank*=1-np.roll(binary,offset,axis=0).astype(np.float32)/255
    signal=cv2.GaussianBlur(contrast*blank,(9,1),0)
    for mid,slope,intercept,left_seed,right_seed in seeds:
        # Do not extrapolate a partial table rule into unrelated image content.
        x0=max(int(w*.025),int(left_seed-w*.015))
        x1=min(int(w*.975),int(right_seed+w*.015))
        radius=min(h*.08,max(h*.025,abs(slope)*w*.5+5))
        lo=max(5,int(mid-radius));hi=min(h-5,int(mid+radius))
        if hi-lo<5:continue
        expected=intercept+slope*np.arange(x0,x1)
        energy=signal[lo:hi,x0:x1]-.7*np.abs(np.arange(lo,hi)[:,None]-expected[None,:])
        cost=-np.minimum(energy[:,0],60)
        back=np.zeros_like(energy,dtype=np.int16)
        for x in range(1,energy.shape[1]):
            options=np.stack([np.r_[cost[0]+5,cost[:-1]+2],
                cost,np.r_[cost[1:]+2,cost[-1]+5]])
            choice=np.argmin(options,axis=0)
            back[:,x]=np.arange(len(cost))+choice-1
            cost=np.min(options,axis=0)-np.minimum(energy[:,x],60)
        path=np.zeros(energy.shape[1],dtype=np.int32);path[-1]=np.argmin(cost)
        for x in range(len(path)-1,0,-1):path[x-1]=back[path[x],x].clip(0,len(cost)-1)
        xx=np.arange(x0,x1);yy=path+lo
        # Measure the actual thin ridge rather than the blurred tracking score.
        points=[];clear=[];thickness=[]
        for x,y in zip(xx,yy):
            hits=np.flatnonzero(binary[y-2:y+3,x])+y-2
            if not len(hits):continue
            groups=np.split(hits,np.where(np.diff(hits)>1)[0]+1)
            group=min(groups,key=lambda g:abs(np.mean(g)-y))
            if len(group)>max(3,h*.006):continue
            yp=float(np.mean(group));points.append([x,yp]);thickness.append(len(group))
            yi=round(yp)
            clear.append(4<=yi<h-4 and np.mean(binary[[yi-4,yi-3,yi+3,yi+4],x])<32)
        if len(points)<w*.27 or np.mean(clear)<.88 or np.median(thickness)>max(2,max(h,w)/320):continue
        pts=np.array(points)
        # Local medians retain a real fold junction; a single cubic cannot.
        # Shorter table cells and partial rules are useful evidence too;
        # do not demand ink all the way to both image margins.
        left,right=np.percentile(pts[:,0],[1,99])
        if right-left<w*.35 or len(pts)/(right-left)<.45:continue
        centers=np.linspace(left,right,33)
        heights=np.array([np.median(pts[np.abs(pts[:,0]-x)<max(3,w/100),1])
            if np.any(np.abs(pts[:,0]-x)<max(3,w/100)) else np.nan for x in centers])
        if not np.isfinite(heights).all():continue
        residual=np.abs(pts[:,1]-np.interp(pts[:,0],centers,heights))
        if np.percentile(residual,85)>1.4:continue
        rule=np.column_stack([centers,heights])/np.array([w-1,h-1])
        if not any(abs(np.mean(rule[:,1])-np.mean(r[:,1]))<.015 for r in rules):rules.append(rule)
    # A trace that merges with or crosses another parallel rule has probably
    # jumped to neighbouring ink. Do not let it fold the fitted mesh.
    rejected=set()
    for i,a in enumerate(rules):
        for j,b in enumerate(rules[:i]):
            lo=max(a[0,0],b[0,0]);hi=min(a[-1,0],b[-1,0])
            if hi-lo<.25:continue
            xx=np.linspace(lo,hi,65)
            gap=np.interp(xx,a[:,0],a[:,1])-np.interp(xx,b[:,0],b[:,1])
            if np.min(np.abs(gap))<.003 and np.max(np.abs(gap))>.012:
                va,vb=np.ptp(a[:,1]),np.ptp(b[:,1])
                if va>vb*2:rejected.add(i)
                elif vb>va*2:rejected.add(j)
                else:rejected.update([i,j])
    result=sorted([r for i,r in enumerate(rules) if i not in rejected],key=lambda p:np.mean(p[:,1]))
    if len(result)==2:
        a,b=result
        lo=max(a[0,0],b[0,0]);hi=min(a[-1,0],b[-1,0])
        if hi-lo>.35:
            xx=np.linspace(lo,hi,65)
            ay=np.interp(xx,a[:,0],a[:,1]);by=np.interp(xx,b[:,0],b[:,1])
            # Two opposite bends alone cannot establish a coherent surface.
            # This commonly happens when tracing separate table cells.
            if min(np.std(ay),np.std(by))>.001 and np.corrcoef(ay,by)[0,1]<0:
                return []
    return result

def rule_mesh(rules,n=33):
    if len(rules)<2:return None
    u,v=np.meshgrid(np.linspace(0,1,n),np.linspace(0,1,n))
    grid=np.stack([u,v],-1)
    targets=np.array([0]+[np.mean(p[:,1]) for p in rules]+[1])
    # A nearby final rule describes the bottom text's surface too. Forcing
    # displacement to zero at the crop edge would bend that text back again.
    # Continue only when two independent rules agree on the bend direction.
    lower=False
    if targets[-2]>.85:
        xx=np.linspace(0,1,n)
        a=np.interp(xx,rules[-2][:,0],rules[-2][:,1])-targets[-3]
        b=np.interp(xx,rules[-1][:,0],rules[-1][:,1])-targets[-2]
        lower=(np.std(a)>.001 and np.std(b)>.001 and
            np.corrcoef(a,b)[0,1]>.8 and np.max(np.abs(a-b))<.02)
    for x in range(n):
        source=[0]+[np.interp(u[0,x],p[:,0],p[:,1]) for p in rules]+[1]
        if lower:source[-1]=1+source[-2]-targets[-2]
        if np.min(np.diff(source))<=0:return None
        grid[:,x,1]=np.interp(v[:,x],targets,source)
    return grid


def extract_rules(gray):
    """Use both scales: faint rules may become too thick when enlarged."""
    rules=_extract_rules_at_scale(gray)
    h,w=gray.shape
    if max(h,w)>800:
        scale=640/max(h,w)
        smaller=cv2.resize(gray,(round(w*scale),round(h*scale)),interpolation=cv2.INTER_AREA)
        for guide in _extract_rules_at_scale(smaller):
            if any(abs(np.mean(guide[:,1])-np.mean(other[:,1]))<.012 for other in rules):continue
            rules.append(guide)
    return sorted(rules,key=lambda p:np.mean(p[:,1]))
