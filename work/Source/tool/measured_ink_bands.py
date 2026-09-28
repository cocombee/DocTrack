"""Rigid measured pixel footprints; empty word-box corners are stock."""
import numpy as np
import cv2


def pixel_footprint(points, width, height):
    points = np.asarray(points, dtype=float)
    if (points.ndim != 2 or points.shape[1] != 2 or not len(points)
            or not np.isfinite(points).all() or (points < 0).any()
            or (points > 1).any() or min(width, height) < 2):
        raise ValueError('Invalid measured ink footprint')
    radius = .5 / np.array([width - 1, height - 1])
    offsets = np.array([[0, 0], [-1, -1], [-1, 1], [1, -1], [1, 1]]) * radius
    return np.unique(np.clip((points[:, None] + offsets).reshape(-1, 2), 0, 1), axis=0)


def column_bands(clouds, columns, tangent_padding, normal_padding):
    """Include every footprint near each knot, retaining its complete owner."""
    if (columns < 2 or not np.isfinite([tangent_padding,normal_padding]).all()
            or min(tangent_padding, normal_padding) < 0):
        raise ValueError('Invalid measured ink band lattice')
    low = np.full((len(clouds), columns), np.inf)
    high = np.full_like(low, -np.inf)
    for index, cloud in enumerate(clouds):
        cloud = np.asarray(cloud, dtype=float)
        if cloud.ndim != 2 or cloud.shape[1] != 2 or not np.isfinite(cloud).all():
            raise ValueError('Invalid measured ink cloud')
        first = np.maximum(0, np.ceil((cloud[:, 0] - tangent_padding) * (columns - 1)).astype(int))
        last = np.minimum(columns - 1, np.floor((cloud[:, 0] + tangent_padding) * (columns - 1)).astype(int))
        for offset in range(int(np.max(last - first, initial=-1)) + 1):
            active = first + offset <= last
            cols = first[active] + offset
            np.minimum.at(low[index], cols, cloud[active, 1] - normal_padding)
            np.maximum.at(high[index], cols, cloud[active, 1] + normal_padding)
    return low, high


def rule_ink_footprints(gray,safe,horizontal,vertical,groups):
 """Measure connected ruler ink while retaining independent word ownership.

 Trace centers alone miss blurred stroke edges. Only observed contrast ink
 touching a supplied rule center is selected; blank stock and word footprints
 never become rule owners. Scratch masks stay contiguous for transposed input.
 """
 gray=np.asarray(gray);safe=np.asarray(safe)
 if (gray.ndim!=2 or min(gray.shape)<2 or safe.shape!=gray.shape or
     not np.isfinite(gray).all() or not np.isfinite(safe).all()):
  raise ValueError('Invalid measured rule raster')
 groups=[np.asarray(g,dtype=float) for g in groups]
 if any(g.ndim!=2 or g.shape[1]!=2 or not len(g) or
        not np.isfinite(g).all() or (g<0).any() or (g>1).any() for g in groups):
  raise ValueError('Invalid independent word footprint')
 h,w=gray.shape;scale=max(h,w)/1280;contrast=cv2.GaussianBlur(gray.astype(np.float32),(0,0),max(2,7*scale))-gray;ink=(contrast>6)&(safe>250);words=np.zeros(gray.shape,np.uint8)
 for g in groups:
  p=np.rint(np.asarray(g)*[w-1,h-1]).astype(int);good=np.all((p>=0)&(p<[w,h]),axis=1);p=p[good];words[p[:,1],p[:,0]]=1
 word_radius=max(1,int(np.ceil(2*scale)));words=cv2.dilate(words,np.ones((2*word_radius+1,)*2,np.uint8));result=[]
 for axis,guides in [(1,horizontal),(0,vertical)]:
  for g in guides:
   uv=np.asarray(g,float) if axis==1 else np.asarray(g,float)[:,::-1]
   if uv.ndim!=2 or uv.shape[1]!=2 or len(uv)<2 or not np.isfinite(uv).all() or (uv<0).any() or (uv>1).any():
    raise ValueError('Invalid observed rule trace')
   center=np.zeros(gray.shape,np.uint8);cv2.polylines(center,[np.rint(uv*[w-1,h-1]).astype(np.int32)],False,1,max(1,int(np.ceil(3*scale))));radius=max(1,int(np.ceil(7*scale)));corridor=cv2.dilate(center,np.ones((2*radius+1,)*2,np.uint8));supported=ink&(corridor>0)&(words==0);n,labels,stats,_=cv2.connectedComponentsWithStats(supported.astype(np.uint8));touching=set(np.unique(labels[center>0]));selected=[]
   for index in touching:
    if index==0:continue
    x,y,cw,ch,area=stats[index]
    if area<12*scale or (cw if axis==1 else ch)<12*scale:continue
    yy,xx=np.where(labels[y:y+ch,x:x+cw]==index);selected.append(np.c_[(xx+x)/(w-1),(yy+y)/(h-1)])
   if selected:result.append({'normal_axis':axis,'points':np.vstack(selected),'center':uv})
 return result
