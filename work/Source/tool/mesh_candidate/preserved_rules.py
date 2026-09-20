"""Recover printed rules from the already accepted warp before replacing it."""
import cv2,numpy as np
from printed_line_mesh import extract_rules

def sample_grid(grid,points):
    n=grid.shape[0];xy=points*(n-1);lo=np.clip(np.floor(xy).astype(int),0,n-2);f=xy-lo;x=lo[:,0];y=lo[:,1];a=f[:,0,None];b=f[:,1,None]
    return (1-a)*(1-b)*grid[y,x]+a*(1-b)*grid[y,x+1]+(1-a)*b*grid[y+1,x]+a*b*grid[y+1,x+1]

def solid_rules(gray):
    # Existing straight borders can be thicker than the reconstruction
    # detector's thin-rule limit. Long horizontal opening cannot follow text.
    h,w=gray.shape
    binary=cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,31,12)
    opened=cv2.morphologyEx(binary,cv2.MORPH_OPEN,np.ones((1,max(20,round(w*.2))),np.uint8))
    _,labels,stats,_=cv2.connectedComponentsWithStats(opened)
    result=[]
    for index,(x,y,ww,hh,area) in enumerate(stats):
        if index==0 or ww<w*.3 or hh>max(8,h*.015) or not h*.025<y<h*.975:continue
        xx=np.linspace(x+1,x+ww-2,33);yy=[]
        for px in xx:
            hits=np.flatnonzero(labels[y:y+hh,round(px)]==index)
            yy.append(y+np.mean(hits) if len(hits) else np.nan)
        if np.isfinite(yy).all():result.append(np.column_stack([xx,yy])/[w-1,h-1])
    return result

def accepted_output_rules(gray,accepted):
    h,w=gray.shape;scale=min(1.,640/max(h,w));oh,ow=round(h*scale),round(w*scale)
    x,y=np.meshgrid(np.linspace(0,1,ow),np.linspace(0,1,oh));p=np.stack([x,y],-1).reshape(-1,2)
    back=sample_grid(accepted,p).reshape(oh,ow,2)
    previous=cv2.remap(gray,(back[:,:,0]*(w-1)).astype(np.float32),(back[:,:,1]*(h-1)).astype(np.float32),cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=255)
    observed=[]
    for guides,axis in ((extract_rules(previous)+solid_rules(previous),1),([p[:,::-1] for p in extract_rules(previous.T)+solid_rules(previous.T)],0)):
        for points in guides:
            observed.append((sample_grid(accepted,points),float(np.std(points[:,axis])),axis))
    return observed

def preserves_output_rules(candidate,observed,inverse_points,tolerance=.0003):
    return all(np.std(inverse_points(candidate,source)[:,axis])<=before+tolerance for source,before,axis in observed)
