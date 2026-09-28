"""Identify rigid proxy groups that are actually an established straight rule."""
import numpy as np


def orthogonal_rule_components(groups, clouds, native_start, guides, width, height):
    """Return only non-native repeated thin pieces on a known flat crossing row.

    In a normal-coordinate correction these pieces may stretch tangentially.
    Their complete physical footprints and normal positions remain guarded.
    Native word groups are never changed by this proxy classification.
    """
    if min(width,height)<2 or not 0<=native_start<=len(clouds):
        raise ValueError('Invalid printed component owner basis')
    scale=max(width,height)/1280;basis=np.array([width-1,height-1]);measured=[]
    for index,value in enumerate(guides):
        guide=np.asarray(value,dtype=float)
        if (guide.ndim!=2 or guide.shape[1]!=2 or len(guide)<64
                or not np.isfinite(guide).all() or np.ptp(guide[:,0])*basis[0]>.05):continue
        guide=guide[np.argsort(guide[:,1])]
        if np.ptp(guide[:,1])<.15:continue
        measured.append((index,guide))
    roles=[]
    for owner,group in enumerate(groups):
        if len(group)<3 or any(i>=native_start for i in group):continue
        parts=[np.asarray(clouds[i],dtype=float) for i in group]
        if any(q.ndim!=2 or q.shape[1]!=2 or not len(q) or not np.isfinite(q).all() for q in parts):
            raise ValueError('Invalid printed component footprint')
        if any(np.ptp(q[:,0])*basis[0]>4*scale or np.ptp(q[:,1])*basis[1]<4*scale
               or np.ptp(q[:,1])*basis[1]<3*np.ptp(q[:,0])*basis[0] for q in parts):continue
        all_points=np.vstack(parts)
        if np.ptp(all_points[:,1])*basis[1]<max(48*scale,.04*height):continue
        for guide_index,guide in measured:
            if (all_points[:,1].min()<guide[0,1] or all_points[:,1].max()>guide[-1,1]):continue
            expected=np.interp(all_points[:,1]*basis[1],guide[:,1]*basis[1],guide[:,0]*basis[0])
            if abs(all_points[:,0]*basis[0]-expected).max()>3*scale:continue
            roles.append({'proxy_owner':owner,'component_indices':[int(i) for i in group],
                          'guide_index':guide_index,'normal_axis':0,
                          'evidence':'Repeated non-native thin footprints on an established straight orthogonal rule'})
            break
    return roles
