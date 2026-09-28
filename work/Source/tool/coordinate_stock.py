"""Allocate normal corrections to blank stock without resampling prior maps.

Knots are fixed boundaries of measured ink bands. Equal boundary movements
are rigid translations; only gaps with unequal movements may be stretched.
This module is restricted to ordered normal-coordinate document corrections.
"""
import cv2
import numpy as np


def minimum_cell_jacobian(grid):
    grid=np.asarray(grid,dtype=float)
    rows,cols=grid.shape[:2]
    if grid.ndim!=3 or grid.shape[2]!=2 or min(rows,cols)<2 or not np.isfinite(grid).all():
        raise ValueError('Invalid stock capacity grid')
    dx,dy=np.diff(grid,axis=1),np.diff(grid,axis=0)
    return np.minimum.reduce([(a[:,:,0]*b[:,:,1]-a[:,:,1]*b[:,:,0])*(rows-1)*(cols-1)
                              for a in (dx[:-1],dx[1:]) for b in (dy[:,:-1],dy[:,1:])])


def retained_capacity(parent, rows, columns, query_rows, query_columns,
                      column_knots, weights, normal_center, radius):
    """Bound actual parent cells in the used stock corridor, never an alias."""
    from retained_composition_guard import _quality
    plans=[(col,np.asarray(knots,dtype=float)) for col,knots in enumerate(column_knots)
           if knots is not None]
    if not plans:
        return np.empty((0,0)),0,0
    if any(data.ndim!=2 or data.shape[1]!=2 or len(data)<2 or
           not np.isfinite(data).all() for _,data in plans):
        raise ValueError('Invalid fixed stock boundaries')
    lows=[];highs=[]
    for col,data in plans:
        blend=float(weights[round(normal_center*(query_rows-1)),col])
        sources=data[:,0]+(data[:,1]-data[:,0])*blend
        lows.append(float(sources.min()));highs.append(float(sources.max()))
    c0=max(0,int(np.floor((plans[0][0]-1)/(query_columns-1)*(columns-1))))
    c1=min(columns-1,int(np.floor((plans[-1][0]+1)/(query_columns-1)*(columns-1)))+1)
    r0=max(0,int(np.floor(min(lows)*(rows-1)))-radius-2)
    r1=min(rows-1,int(np.floor(max(highs)*(rows-1)))+radius+3)
    if c0>=c1 or r0>=r1:
        raise ValueError('Stock corridor lies outside the parent grid')
    quality=np.empty((r1-r0,c1-c0))
    for start in range(r0,r1,32):
        stop=min(r1,start+32)
        x,y=np.meshgrid(np.arange(c0,c1)/(columns-1),
                        np.arange(start,stop)/(rows-1))
        low=np.c_[x.ravel(),y.ravel()]
        high=low+[1/(columns-1),1/(rows-1)]
        quality[start-r0:stop-r0]=_quality(parent,low,high).reshape(stop-start,c1-c0)
    return quality,r0,c0


def limited_destination_density(capacity,source_lengths,destination_length,minimum_area=.21):
    """Spend only the capacity needed; keep uniform stock when it is safe.

    Destination length per source length cannot exceed capacity/minimum_area.
    A capped uniform allocation avoids bending unrelated blank stock just
    because neighbouring parent cells happen to have different capacities.
    The final composition still needs its independent continuous-area proof.
    """
    capacity=np.asarray(capacity,dtype=float)
    source_lengths=np.asarray(source_lengths,dtype=float)
    if (capacity<=0).any() or (source_lengths<=0).any():
        raise ValueError('Invalid capacity inside the required stock interval')
    ceiling=np.minimum(capacity/minimum_area,1/minimum_area)
    available=float(ceiling@source_lengths)
    if available<destination_length-1e-12:
        raise ValueError('Insufficient actual capacity for the required stock interval')
    uniform=destination_length/source_lengths.sum()
    if (uniform<=ceiling).all():return np.full_like(capacity,uniform)
    low,high=uniform,float(ceiling.max())
    for _ in range(60):
        middle=(low+high)/2
        if np.minimum(middle,ceiling)@source_lengths<destination_length:low=middle
        else:high=middle
    return np.minimum(high,ceiling)


def _range_minima(values,left,right):
    """Exact minima of inclusive, independently sized source intervals."""
    lengths=right-left+1
    levels=np.floor(np.log2(lengths)).astype(int)
    result=np.empty(len(left));series=np.asarray(values)
    maximum=int(levels.max(initial=0))
    for level in range(maximum+1):
        selected=levels==level;size=1<<level
        result[selected]=np.minimum(series[left[selected]],series[right[selected]-size+1])
        if level<maximum:
            shift=1<<level
            series=np.minimum(series[:-shift],series[shift:])
    return result


def neighbour_span_capacity(capacity,query,column,parent_rows,quality_row):
    """Cover the source stock crossed between neighbouring query columns.

    A diagonal query cell can enter a compressed parent strip even when its
    two column endpoints avoid it. Invert each source-bin endpoint into the
    proposed destination column, then include both neighbours at those same
    destinations. The independent continuous composition guard remains required
    after allocation: reallocating stock can change the proposed footprint.
    """
    qr,qc=query.shape[:2];v=np.arange(qr)/(qr-1)
    low=(np.arange(len(capacity))+quality_row)/(parent_rows-1)
    high=low+1/(parent_rows-1)
    destination_low=np.interp(low,query[:,column,1],v)
    destination_high=np.interp(high,query[:,column,1],v)
    source_low=low.copy();source_high=high.copy()
    for neighbour in (column-1,column+1):
        if 0<=neighbour<qc:
            source_low=np.minimum(source_low,np.interp(destination_low,v,query[:,neighbour,1]))
            source_high=np.maximum(source_high,np.interp(destination_high,v,query[:,neighbour,1]))
    left=np.floor(source_low*(parent_rows-1)).astype(int)-quality_row
    right=np.minimum(parent_rows-2,np.floor(source_high*(parent_rows-1)).astype(int))-quality_row
    covered=(left>=0)&(right<len(capacity))&(right>=left)
    result=np.full(len(capacity),-np.inf)
    result[covered]=_range_minima(capacity,left[covered],right[covered])
    return result


def allocate_stock(old,query,column_knots,weights,normal_center,*,capacity_radius=4,allow_unused_alias=False,parent=None,minimum_area=.21,include_neighbour_span=False):
    """Put more required compression in stock with greater area capacity.

    The supplied weights must be constant across each active normal corridor.
    Ink endpoints and rigid bands remain unchanged. Callers must independently
    validate the final continuous composition and all protected print.
    """
    old=np.asarray(old,dtype=float);query=np.asarray(query,dtype=float)
    qr,qc=query.shape[:2];nr,nc=old.shape[:2]
    if query.ndim!=3 or query.shape[2]!=2 or len(column_knots)!=qc:
        raise ValueError('Invalid normal-coordinate stock grid')
    if np.asarray(weights).shape!=query.shape[:2] or not 0<=normal_center<=1 or capacity_radius<0:
        raise ValueError('Invalid stock corridor weights')
    if not np.isfinite(minimum_area) or not .21<=minimum_area<=1:
        raise ValueError('Stock allocation must retain at least the existing area reserve')
    expected_x=np.arange(qc)/(qc-1)
    if not np.allclose(query[:,:,0],expected_x[None,:],atol=1e-14,rtol=0):
        raise ValueError('Stock allocation requires an unchanged tangent coordinate')
    if include_neighbour_span and (parent is None or not np.isfinite(query).all() or
                                   (np.diff(query[:,:,1],axis=0)<=0).any()):
        raise ValueError('Neighbour capacity requires an ordered query and its actual parent')
    quality_row,quality_column=0,0
    if parent is None:
        quality=minimum_cell_jacobian(old)
        if (quality<=0).any() and not allow_unused_alias:
            raise ValueError('Invalid prior stock capacity')
    else:
        quality,quality_row,quality_column=retained_capacity(parent,nr,nc,qr,qc,
                column_knots,np.asarray(weights),normal_center,capacity_radius)
    result=query.copy();result[:,:,0]=expected_x[None,:]
    v=np.arange(qr)/(qr-1);old_v=np.arange(nr)/(nr-1)
    for col,knots in enumerate(column_knots):
        if knots is None:continue
        data=np.array(knots,dtype=float,copy=True)
        if data.ndim!=2 or data.shape[1]!=2 or len(data)<2 or not np.isfinite(data).all():
            raise ValueError('Invalid fixed stock boundaries')
        data[:,1]=data[:,0]+(data[:,1]-data[:,0])*float(weights[round(normal_center*(qr-1)),col])
        if (np.diff(data,axis=0)<=0).any():
            raise ValueError('Crossed fixed stock boundaries')
        left=max(0,int(np.floor((col-1)/(qc-1)*(nc-1))))
        right=min(nc-2,int(np.floor((col+1)/(qc-1)*(nc-1))))
        cap=cv2.erode(quality[:,left-quality_column:right-quality_column+1].min(1)[:,None],np.ones((2*capacity_radius+1,1),np.uint8),borderType=cv2.BORDER_REPLICATE).ravel()
        if include_neighbour_span:
            cap=neighbour_span_capacity(cap,query,col,nr,quality_row)
        for (d0,s0),(d1,s1) in zip(data,data[1:]):
            inside=(v>=d0)&(v<=d1)
            if not inside.any():continue
            active_weights=np.asarray(weights)[inside,col]
            if not np.allclose(active_weights,weights[round(normal_center*(qr-1)),col],atol=1e-14,rtol=0):
                raise ValueError('Stock corridor weight varies normally')
            if abs((s1-d1)-(s0-d0))<1e-10:
                result[inside,col,1]=v[inside]+s0-d0
                continue
            source_nodes=np.r_[s0,old_v[(old_v>s0)&(old_v<s1)],s1]
            middle=(source_nodes[:-1]+source_nodes[1:])/2
            indices=np.floor(middle*(nr-1)).astype(int)-quality_row
            if (indices<0).any() or (indices>=len(cap)).any():
                raise ValueError('Stock interval exceeds its measured capacity')
            lengths=np.diff(source_nodes)
            areas=(cap[indices]*lengths if parent is None else
                   limited_destination_density(cap[indices],lengths,d1-d0,minimum_area=minimum_area)*lengths)
            if (areas<=0).any():
                raise ValueError('Invalid capacity inside the required stock interval')
            cumulative=np.r_[0,np.cumsum(areas)]
            destination_nodes=d0+cumulative/cumulative[-1]*(d1-d0)
            result[inside,col,1]=np.interp(v[inside],destination_nodes,source_nodes)
    if (np.diff(result[:,:,1],axis=0)<=0).any():
        raise ValueError('Stock allocation lost normal ordering')
    return result
