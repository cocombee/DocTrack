"""Preserve already aligned word columns independently of glyph shape.

Native quads use measurement coordinates. A column needs at least three
distinct rows, a common font scale and a consistently vertical leading edge.
This module supplies relations, not new text geometry or correction targets.
"""
import numpy as np


def aligned_columns(quads,pixel_size,tolerance=.5):
    scale=np.asarray(pixel_size,dtype=float)
    records=[]
    for index,quad in enumerate(quads):
        quad=np.asarray(quad,dtype=float)
        if quad.shape!=(4,2) or not np.isfinite(quad).all():continue
        points=quad*scale
        leading=points[np.argsort(points[:,0])[:2]]
        height=np.ptp(points[:,1])
        if height<6 or np.ptp(leading[:,0])>tolerance:continue
        records.append((index,float(leading[:,0].mean()),
                        float(points[:,1].mean()),float(height)))
    parent=list(range(len(records)))
    def root(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]];i=parent[i]
        return i
    for i,(_,x,y,height) in enumerate(records):
        for j in range(i+1,len(records)):
            _,xx,yy,hh=records[j]
            if (abs(x-xx)<=tolerance and max(height,hh)<=min(height,hh)*1.5
                    and min(height,hh)*.7<=abs(y-yy)<=max(height,hh)*3.2):
                parent[root(j)]=root(i)
    groups={}
    for i,record in enumerate(records):groups.setdefault(root(i),[]).append(record)
    return [[q[0] for q in sorted(group,key=lambda q:q[2])]
            for group in groups.values() if len(group)>=3 and
            np.ptp([q[1] for q in group])<=tolerance and
            max(q[3] for q in group)<=min(q[3] for q in group)*1.5]


def independent_owner_links(columns,quads,groups,native_count):
    """Associate only unique current owners; retain ambiguity as uncertainty."""
    output=[]
    for column in columns:
        owners=[]
        for index in column:
            quad=np.asarray(quads[index]);lo,hi=quad.min(0),quad.max(0)
            matches=[]
            for owner,group in enumerate(groups[:native_count]):
                group=np.asarray(group);a,z=group.min(0),group.max(0)
                area=min((hi-lo).prod(),(z-a).prod())
                if area>0 and np.maximum(0,np.minimum(hi,z)-np.maximum(lo,a)).prod()/area>.5:
                    matches.append(owner)
            if len(matches)!=1:break
            owners.append(matches[0])
        else:
            if len(set(owners))==len(owners):output.append(owners)
    return output


def alignment_audit(before,after,pixel_size,*,normal_axis=0,tolerance=.05):
    """Allow one common translation; reject a newly curved alignment."""
    before,after=np.asarray(before),np.asarray(after)
    delta=(after-before)*np.asarray(pixel_size)
    error=float(np.ptp(delta[:,normal_axis]))
    return {'relative_alignment_change_px':error,'accepted':bool(error<tolerance)}
