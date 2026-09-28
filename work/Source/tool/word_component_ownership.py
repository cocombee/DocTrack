"""Resolve measured glyphs borrowed by another word's localization halo.

Only fresh observations may use this partition. An earlier paired registry is
immutable. Every observed physical point stays present at the same coordinate.
Ambiguous connected print keeps its existing conservative owners.
"""
import cv2
import numpy as np


def partition_shared_ink(gray,groups,quads):
    h,w=gray.shape;scale=max(h,w)/1280
    groups=[np.asarray(g,dtype=float) for g in groups]
    quads=[np.asarray(q,dtype=float) for q in quads]
    if len(groups)!=len(quads) or any(q.shape!=(4,2) or not np.isfinite(q).all() for q in quads):
        raise ValueError('Word ownership needs paired native observations')
    if any(g.ndim!=2 or g.shape[1]!=2 or not len(g) or not np.isfinite(g).all()
           or (g<0).any() or (g>1).any() for g in groups):
        raise ValueError('Invalid measured word ink')
    contrast=cv2.GaussianBlur(gray.astype(np.float32),(0,0),max(2,7*scale))-gray
    _,labels,stats,_=cv2.connectedComponentsWithStats((contrast>6).astype(np.uint8))
    pixels=[np.rint(g*[w-1,h-1]).astype(int) for g in groups]
    memberships=[labels[p[:,1],p[:,0]] for p in pixels]
    owners={}
    for index,membership in enumerate(memberships):
        ids,counts=np.unique(membership,return_counts=True)
        for comp,count in zip(ids,counts):
            if comp:owners.setdefault(int(comp),[]).append((index,int(count)))
    keep=[np.ones(len(g),bool) for g in groups];extra=[[] for _ in groups];audit=[]
    for comp,observed in owners.items():
        if len(observed)<2:continue
        x,y,cw,ch,area=stats[comp];chosen=None;evidence=None
        ranked=sorted(observed,key=lambda q:-q[1])
        glyph=(6*scale<=ch<=38*scale and 2<=cw<=w*.05 and area>12*scale
               and ranked[0][1]>=.8*area)
        if glyph:
            small=all(n<.15*area for _,n in ranked[1:])
            halo_only=False
            if not small and all(n<.35*area for _,n in ranked[1:]):
                owner=ranked[0][0]
                yy,xx=np.where(labels==comp);center=(float(xx.mean()),float(yy.mean()))
                quad=np.asarray(quads[owner]*[w-1,h-1],np.float32)
                halo_only=cv2.pointPolygonTest(quad,center,False)>0
                for index,_ in ranked[1:]:
                    q=np.asarray(quads[index]*[w-1,h-1],np.float32)
                    borrowed=pixels[index][memberships[index]==comp]
                    halo_only &= all(cv2.pointPolygonTest(q,(float(a),float(b)),False)<0
                                     for a,b in borrowed)
            if small or halo_only:
                chosen=ranked[0][0]
                evidence=('complete measured glyph versus small borrowed edge' if small
                          else 'complete glyph center and exclusively outside competing native envelopes')
        if chosen is None and area<=max(1,scale*scale) and cw<=2 and ch<=2:
            yy,xx=np.where(labels==comp);points=np.c_[xx,yy].astype(float)
            inside=[]
            for index,_ in observed:
                quad=np.asarray(quads[index]*[w-1,h-1],np.float32)
                if all(cv2.pointPolygonTest(quad,(float(a),float(b)),False)>=0 for a,b in points):
                    inside.append(index)
            if len(inside)==1:
                chosen=inside[0];evidence='unique undilated native envelope'
        if chosen is None:continue
        transferred=[]
        for index,_ in observed:
            if index==chosen:continue
            ids=np.flatnonzero(memberships[index]==comp)
            # Never make a word empty or replace its body with a new owner.
            if not keep[index][memberships[index]!=comp].any():
                transferred=[];break
            transferred.append((index,ids))
        if not transferred:continue
        for index,ids in transferred:
            keep[index][ids]=False;extra[chosen].append(groups[index][ids])
        audit.append({'component':comp,'owner':chosen,'evidence':evidence,
                      'transferred':[{ 'word_index':i,'point_indices':ids.tolist(),
                                      'points':groups[i][ids].tolist()} for i,ids in transferred]})
    clean=[np.unique(np.vstack([g[k],*extra[i]]),axis=0)
           for i,(g,k) in enumerate(zip(groups,keep))]
    before=np.unique(np.vstack(groups),axis=0) if groups else np.empty((0,2))
    after=np.unique(np.vstack(clean),axis=0) if clean else np.empty((0,2))
    if not np.array_equal(before,after):raise ValueError('Word partition lost physical ink')
    return clean,{'roles':audit,'all_physical_points_retained':True,
                  'physical_points':len(before)}
