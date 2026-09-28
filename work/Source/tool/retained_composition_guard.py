"""Conservative continuous area bounds through overlapping retained maps.

Every box is partitioned at actual layer ownership boundaries. Bilinear maps
and their affine determinants are bounded over all touched cell fragments.
An older area floor is used only after recomputing its original certificate.
"""
import copy
import numpy as np
from composite_mesh_sampling import MeshSampler
from composite_mesh_inverse import differential
from normal_composition_guard import cross,validate_normal_composition,validate_source_composition


class _Identity(MeshSampler):
    def __init__(self):
        super().__init__({'rows':2,'columns':2,'grid':[[0,0],[1,0],[0,1],[1,1]]})

    def sample(self,points):
        return np.array(points,dtype=float,copy=True)


def _pieces(indices,low,high,regions):
    remaining=[(indices,low,high)];owned=[]
    for region in regions:
        following=[]
        for ids,a,z in remaining:
            lo=np.maximum(a,region[:2]);hi=np.minimum(z,region[2:])
            hit=np.all(hi>lo,axis=1)
            if (~hit).any():following.append((ids[~hit],a[~hit],z[~hit]))
            if not hit.any():continue
            i,aa,zz,ll,hh=ids[hit],a[hit],z[hit],lo[hit],hi[hit]
            owned.append((i,ll,hh))
            for p,q in [(aa,np.c_[ll[:,0],zz[:,1]]),
                        (np.c_[hh[:,0],aa[:,1]],zz),
                        (np.c_[ll[:,0],aa[:,1]],np.c_[hh[:,0],ll[:,1]]),
                        (np.c_[ll[:,0],hh[:,1]],np.c_[hh[:,0],zz[:,1]])]:
                good=np.all(q>p,axis=1)
                if good.any():following.append((i[good],p[good],q[good]))
        remaining=following
    return owned,remaining


def _cell_bounds(layer,low,high,*,mapping=False):
    nr,nc,ro,co=(*[layer.rows,layer.columns],0,0) if layer.window is None else layer.window
    left=np.clip(np.floor(low[:,0]*(nc-1)).astype(int),co,co+layer.columns-2)
    right=np.clip(np.floor(high[:,0]*(nc-1)).astype(int),co,co+layer.columns-2)
    top=np.clip(np.floor(low[:,1]*(nr-1)).astype(int),ro,ro+layer.rows-2)
    bottom=np.clip(np.floor(high[:,1]*(nr-1)).astype(int),ro,ro+layer.rows-2)
    quality=np.full(len(low),np.inf);mapped_low=np.full_like(low,np.inf);mapped_high=np.full_like(high,-np.inf)
    for dy in range(int((bottom-top).max(initial=0))+1):
        for dx in range(int((right-left).max(initial=0))+1):
            col=np.minimum(left+dx,right);row=np.minimum(top+dy,bottom)
            x0=np.clip(low[:,0]*(nc-1)-col,0,1);x1=np.clip(high[:,0]*(nc-1)-col,0,1)
            y0=np.clip(low[:,1]*(nr-1)-row,0,1);y1=np.clip(high[:,1]*(nr-1)-row,0,1)
            a=layer.grid[row-ro,col-co];b=layer.grid[row-ro,col-co+1]-a
            c=layer.grid[row-ro+1,col-co]-a
            d=layer.grid[row-ro+1,col-co+1]-a-b-c
            j0=cross(b,c);jx=cross(b,d);jy=cross(d,c)
            for x in (x0,x1):
                for y in (y0,y1):
                    quality=np.minimum(quality,(j0+jx*x+jy*y)*(nr-1)*(nc-1))
                    if mapping:
                        value=a+b*x[:,None]+c*y[:,None]+d*(x*y)[:,None]
                        mapped_low=np.minimum(mapped_low,value);mapped_high=np.maximum(mapped_high,value)
    if mapping:return quality-1e-12,mapped_low-4e-15,mapped_high+4e-15
    return quality-1e-12


def _transpose_sampler(layer):
    result=copy.copy(layer)
    result.rows,result.columns=layer.columns,layer.rows
    result.grid=layer.grid.transpose(1,0,2)[:,:,::-1]
    result.window=None if layer.window is None else [layer.window[i] for i in [1,0,3,2]]
    result.regions=[[r[i] for i in [1,0,3,2]] for r in layer.regions]
    result.parent=None if layer.parent is None else _transpose_sampler(layer.parent)
    if hasattr(result,'_verified_interval_floor'):del result._verified_interval_floor
    return result


def _verified_floor(layer):
    cached=getattr(layer,'_verified_interval_floor',None)
    if cached is not None:return cached
    # No JSON audit number can authorize a bound. Re-run the certificate on
    # the actual primitive and its retained parent before caching the result.
    identity=_Identity()
    primitive=copy.copy(layer);primitive.parent=identity
    nr,nc=(layer.rows,layer.columns) if layer.window is None else layer.window[:2]
    query=primitive.dense(nr,nc);floor=0.
    try:
        if np.max(abs(query[:,:,0]-np.arange(nc)[None,:]/(nc-1)))<1e-14:
            proof=validate_normal_composition(layer.parent,query,layer.regions)
        elif np.max(abs(query[:,:,1]-np.arange(nr)[:,None]/(nr-1)))<1e-14:
            proof=validate_normal_composition(_transpose_sampler(layer.parent),
                    query.transpose(1,0,2)[:,:,::-1],[[r[i] for i in [1,0,3,2]] for r in layer.regions])
        else:proof=validate_source_composition(layer.parent,query,layer.regions)
        if proof['accepted']:floor=proof['continuous_lower_bound']
    except ValueError:
        # Overlapping older layers still use the conservative recursive
        # interval bound; unsupported certificates supply no positive floor.
        pass
    layer._verified_interval_floor=floor
    return floor


def _quality(layer,low,high):
    count=len(low);indices=np.arange(count);bounds=np.full(count,np.inf)
    owned,remaining=([(indices,low,high)],[]) if layer.parent is None else _pieces(indices,low,high,layer.regions)
    for ids,a,z in owned:
        if layer.coordinate_transform:
            determinant,lo,hi=_cell_bounds(layer,a,z,mapping=True)
            value=determinant*_quality(layer.parent,lo,hi)
            floor=_verified_floor(layer)
            if floor>0:value=np.maximum(value,floor)
            value[determinant<=0]=-np.inf
            value[np.any(lo < -1e-12,axis=1)|np.any(hi > 1+1e-12,axis=1)]=-np.inf
        else:value=_cell_bounds(layer,a,z)
        np.minimum.at(bounds,ids,value)
    for ids,a,z in remaining:np.minimum.at(bounds,ids,_quality(layer.parent,a,z))
    return bounds


def _refine_bounds(parent,primitive,low,high,minimum,depth):
    determinant,lo,hi=_cell_bounds(primitive,low,high,mapping=True)
    bounds=determinant*_quality(parent,lo,hi)
    risky=bounds<=minimum
    if not risky.any() or depth<=0:return bounds,0
    centers=(low[risky]+high[risky])/2
    witness=np.linalg.det(differential(parent,primitive.sample(centers))@
                          differential(primitive,centers))
    # A failing exact derivative is a rejection witness. It cannot approve
    # a cell or substitute for the conservative interval certificate.
    ids=np.flatnonzero(risky);risky[ids[witness<=minimum-1e-10]]=False
    if not risky.any():return bounds,0
    ids=np.flatnonzero(risky);a,z=low[risky],high[risky];mid=(a+z)/2
    # Subdivide actual query-space rectangles. This tightens an interval
    # proof; sampled derivatives never authorize a risky cell.
    children=[(a,mid),(np.c_[mid[:,0],a[:,1]],np.c_[z[:,0],mid[:,1]]),
              (np.c_[a[:,0],mid[:,1]],np.c_[mid[:,0],z[:,1]]),(mid,z)]
    refined=np.full(len(ids),np.inf);count=0
    for p,q in children:
        value,extra=_refine_bounds(parent,primitive,p,q,minimum,depth-1)
        refined=np.minimum(refined,value);count+=len(value)+extra
    bounds[ids]=refined
    return bounds,count


def validate_retained_composition(parent,query,regions,*,minimum=.2,refinement_depth=6):
    query=np.asarray(query,dtype=float)
    if query.ndim!=3 or query.shape[2]!=2 or min(query.shape[:2])<2 or not np.isfinite(query).all():
        raise ValueError('Invalid retained coordinate query')
    if (query < -1e-12).any() or (query > 1+1e-12).any():
        raise ValueError('Coordinate query leaves the document domain')
    if not isinstance(refinement_depth,int) or not 0<=refinement_depth<=8:
        raise ValueError('Invalid interval refinement depth')
    nr,nc=query.shape[:2];lower=np.inf;checked=0;refined=0
    primitive=MeshSampler({'rows':nr,'columns':nc,'grid':query.reshape(-1,2)})
    for region in regions:
        left=max(0,int(np.floor(region[0]*(nc-1))));right=min(nc-1,int(np.ceil(region[2]*(nc-1))))
        top=max(0,int(np.floor(region[1]*(nr-1))));bottom=min(nr-1,int(np.ceil(region[3]*(nr-1))))
        block=query[top:bottom+1,left:right+1]
        corners=[block[:-1,:-1],block[:-1,1:],block[1:,:-1],block[1:,1:]]
        lo=np.minimum.reduce(corners).reshape(-1,2);hi=np.maximum.reduce(corners).reshape(-1,2)
        dx=np.diff(block,axis=1);dy=np.diff(block,axis=0)
        determinant=np.minimum.reduce([cross(a,b)*(nr-1)*(nc-1)
                    for a in (dx[:-1],dx[1:]) for b in (dy[:,:-1],dy[:,1:])]).ravel()-1e-12
        if (determinant<=0).any():return {'accepted':False,'continuous_lower_bound':float(determinant.min()),'required_minimum':minimum,'checked_cells':checked,'rejected':'folded coordinate primitive'}
        for start in range(0,len(lo),32768):
            stop=min(len(lo),start+32768)
            bounds=determinant[start:stop]*_quality(parent,lo[start:stop]-4e-15,hi[start:stop]+4e-15)
            risky=np.flatnonzero(bounds<=minimum)
            if len(risky) and refinement_depth:
                flat=start+risky;row=top+flat//(right-left);col=left+flat%(right-left)
                a=np.c_[col/(nc-1),row/(nr-1)];z=np.c_[(col+1)/(nc-1),(row+1)/(nr-1)]
                precise,extra=_refine_bounds(parent,primitive,a,z,minimum,refinement_depth)
                bounds[risky]=precise;refined+=len(risky)+extra
            lower=min(lower,float(bounds.min()));checked+=len(bounds)
    return {'accepted':bool(np.isfinite(lower) and lower>minimum),
            'continuous_lower_bound':float(lower),'required_minimum':minimum,
            'checked_cells':checked,'refined_cell_fragments':refined}
