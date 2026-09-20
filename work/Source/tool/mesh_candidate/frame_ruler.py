"""Conservative rectangular printed-panel evidence, independent of text rows."""
import cv2
import numpy as np


def _edge_profile(gray, seeds, positions, vertical, polarity):
    work=gray if vertical else gray.T
    h,w=work.shape
    # Compare stock on opposite sides of the proposed boundary. A photograph's
    # internal texture alone must not establish a frame edge.
    smooth=cv2.GaussianBlur(work.astype(np.float32),(3,7),0)
    gradient=np.maximum(0,polarity*(np.roll(smooth,3,axis=1)-np.roll(smooth,-3,axis=1)))
    radius=max(6,round(min(h,w)*.015))
    lo=max(4,int(min(seeds))-radius);hi=min(w-4,int(max(seeds))+radius+1)
    if hi<=lo:return None
    xx=np.arange(lo,hi)
    energy=[]
    for y,seed in zip(positions,seeds):
        y=int(round(y))
        stock=smooth[y,int(np.clip(seed-polarity*(radius+12),0,w-1))]
        outside=smooth[y,np.clip(xx-polarity*6,0,w-1)]
        score=gradient[y,lo:hi]-.15*np.abs(xx-seed)-2*np.maximum(0,np.abs(outside-stock)-8)
        score[np.abs(xx-seed)>radius]=-1000
        energy.append(score)
    energy=np.asarray(energy)
    step=max(2,round(np.median(np.diff(positions))*.55))
    shifts=np.arange(-step,step+1)
    cost=-energy[0];back=[]
    for row in range(1,len(positions)):
        options=[]
        for shift in shifts:
            moved=np.roll(cost,shift)+.3*shift*shift
            if shift>0:moved[:shift]=1e6
            if shift<0:moved[shift:]=1e6
            options.append(moved)
        options=np.asarray(options);choice=np.argmin(options,axis=0)
        back.append(np.arange(len(xx))-shifts[choice])
        cost=np.min(options,axis=0)-energy[row]
    path=np.empty(len(positions),int);path[-1]=np.argmin(cost)
    for row in range(len(positions)-1,0,-1):path[row-1]=back[row-1][path[row]]
    observed=xx[path].astype(float)
    strength=gradient[np.rint(positions).astype(int),xx[path]]
    if np.median(strength)<14 or np.mean(strength>8)<.8:return None
    points=np.column_stack([observed,positions])
    if not vertical:points=points[:,::-1]
    return points/np.array([gray.shape[1]-1,gray.shape[0]-1])


def frame_rulers(gray):
    h,w=gray.shape
    threshold,_=cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    threshold=max(threshold,float(np.percentile(gray,90))-30)
    binary=(gray<threshold).astype(np.uint8)*255
    k=max(3,round(min(h,w)*.008)|1)
    binary=cv2.morphologyEx(binary,cv2.MORPH_OPEN,np.ones((k,k),np.uint8))
    contours,_=cv2.findContours(binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    horizontal=[];vertical=[]
    for c in contours:
        area=cv2.contourArea(c)
        if not h*w*.1<area<h*w*.65:continue
        x,y,bw,bh=cv2.boundingRect(c)
        if min(bw/w,bh/h)<.2 or x<w*.025 or y<h*.025 or x+bw>w*.975 or y+bh>h*.975:continue
        if area/(bw*bh)<.70 or area/max(1,cv2.contourArea(cv2.convexHull(c)))<.82:continue
        polygon=cv2.approxPolyDP(c,cv2.arcLength(c,True)*.006,True).reshape(-1,2).astype(float)
        a=np.roll(polygon,1,axis=0)-polygon;b=np.roll(polygon,-1,axis=0)-polygon
        la=np.linalg.norm(a,axis=1);lb=np.linalg.norm(b,axis=1)
        angles=np.degrees(np.arccos(np.clip(np.sum(a*b,axis=1)/(la*lb),-1,1)))
        sharp=(angles>=65)&(angles<=115)&(np.minimum(la,lb)>min(bw,bh)*.1)
        if np.count_nonzero(sharp)!=4:continue
        filled=np.zeros((h,w),np.uint8);cv2.drawContours(filled,[c],-1,255,-1)
        yy=np.linspace(y+bh*.025,y+bh*.975,97)
        left=[];right=[]
        for row in yy:
            hits=np.flatnonzero(filled[round(row)])
            if not len(hits):break
            left.append(hits[0]);right.append(hits[-1])
        if len(left)!=len(yy):continue
        # Opposite sides must describe one broad panel, not an isolated object.
        widths=np.array(right)-left
        if np.percentile(widths,10)<bw*.6:continue
        l=_edge_profile(gray,left,yy,True,1);r=_edge_profile(gray,right,yy,True,-1)
        if l is None or r is None:continue
        xx=np.linspace(max(left)+bw*.03,min(right)-bw*.03,97)
        if xx[-1]-xx[0]<bw*.55:continue
        top=[];bottom=[]
        for col in xx:
            hits=np.flatnonzero(filled[:,round(col)])
            if not len(hits):break
            top.append(hits[0]);bottom.append(hits[-1])
        if len(top)!=len(xx):continue
        t=_edge_profile(gray,top,xx,False,1);b=_edge_profile(gray,bottom,xx,False,-1)
        if t is None or b is None:continue
        # Four coherent sides are required. An ellipse or natural cutout has
        # strongly changing opposing separation and cannot qualify as a panel.
        if np.ptp(widths)>bw*.28 or np.ptp(np.array(bottom)-top)>bh*.2:continue
        horizontal.extend([t,b]);vertical.extend([l,r])
    return horizontal,vertical


def band_rulers(bgr, paragraph_guides):
    """A uniform printed band needs matching bends in two separate columns."""
    if len(paragraph_guides)<2:return []
    h,w=bgr.shape[:2]
    hsv=cv2.cvtColor(bgr,cv2.COLOR_BGR2HSV)
    mask=((hsv[:,:,1]>65)&(hsv[:,:,2]>40)).astype(np.uint8)*255
    contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    guides=[]
    for c in contours:
        area=cv2.contourArea(c);x,y,bw,bh=cv2.boundingRect(c)
        if area<h*w*.02 or bh<h*.65 or not w*.018<bw<w*.15:continue
        region=np.zeros((h,w),np.uint8);cv2.drawContours(region,[c],-1,255,-1)
        hue=hsv[:,:,0][(region>0)&(mask>0)].astype(float)
        if not len(hue) or np.percentile(np.abs(hue-np.median(hue)),90)>8:continue
        samples=[]
        for yy in range(y+3,y+bh-3):
            xx=np.flatnonzero(region[yy]&mask[yy])
            if len(xx)>bw*.5:samples.append([xx[0],xx[-1],yy])
        samples=np.asarray(samples,float)
        if len(samples)<bh*.85:continue
        widths=samples[:,1]-samples[:,0]
        if np.std(widths)/np.mean(widths)>.12:continue
        center=samples[:,:2].mean(1)/(w-1);yy=samples[:,2]/(h-1)
        matches=[]
        for p in paragraph_guides:
            if p[0,1]<yy[0] or p[-1,1]>yy[-1]:continue
            band=np.interp(p[:,1],yy,center)
            if min(np.std(band),np.std(p[:,0]))<.0003:continue
            if np.corrcoef(band,p[:,0])[0,1]<.6:continue
            px=float(np.mean(p[:,0]))
            if all(abs(px-other)>.05 for other in matches):matches.append(px)
        if len(matches)<2:continue
        target_y=np.linspace(yy[0],yy[-1],97)
        for side in (0,1):
            edge=np.array([np.median(samples[np.abs(yy-v)<.006,side]) for v in target_y])/(w-1)
            guides.append(np.column_stack([edge,target_y]))
    return guides
