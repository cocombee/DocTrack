"""Continuous area guard for a normal warp through retained source meshes.

Each prior bilinear layer is checked in its actual owned rectangles. Most
cells use conservative bounds; uncertain cells are minimized analytically
across every prior cell seam. No sampled composite approximation is accepted.
"""
import numpy as np


def intersection(a,b):
    lo=np.maximum(a[:2],b[:2]);hi=np.minimum(a[2:],b[2:])
    return None if (hi<=lo).any() else [*lo,*hi]


def subtract(a,b):
    hit=intersection(a,b)
    if hit is None:return [a]
    x0,y0,x1,y1=a;l,t,r,d=hit
    return [q for q in [[x0,y0,l,y1],[r,y0,x1,y1],[l,y0,r,t],[l,d,r,y1]] if q[0]<q[2] and q[1]<q[3]]


def owned_rectangles(sampler,domains):
    if sampler.coordinate_transform:
        # Outside a retained coordinate patch the exact incoming point is
        # forwarded unchanged. Disjoint later corrections can use the older
        # source layers without collapsing or approximating the first patch.
        if all(intersection(domain,region) is None for domain in domains for region in sampler.regions):
            return owned_rectangles(sampler.parent,domains)
        raise ValueError('Nested coordinate layers require their own continuous proof')
    if sampler.parent is None:return [(sampler,q) for q in domains]
    owned=[];remaining=list(domains)
    for region in sampler.regions:
        next_remaining=[]
        for domain in remaining:
            hit=intersection(domain,region)
            if hit is not None:owned.append((sampler,hit))
            next_remaining.extend(subtract(domain,region))
        remaining=next_remaining
    return owned+owned_rectangles(sampler.parent,remaining)


def cross(a,b):return a[...,0]*b[...,1]-a[...,1]*b[...,0]


def layer_quality(layer):
    grid=layer.grid
    nr,nc=(layer.rows,layer.columns) if layer.window is None else layer.window[:2]
    dx,dy=np.diff(grid,axis=1),np.diff(grid,axis=0)
    return np.minimum.reduce([cross(a,b)*(nr-1)*(nc-1) for a in (dx[:-1],dx[1:]) for b in (dy[:,:-1],dy[:,1:])])


def exact_cell(layer,tile,query,row,col):
    """Minimum determinant of the true composition in one new cell."""
    qr,qc=query.shape[:2]
    nr,nc,ro,co=(*[layer.rows,layer.columns],0,0) if layer.window is None else layer.window
    a=query[row,col,1];b=query[row,col+1,1]-a
    c=query[row+1,col,1]-a;d=query[row+1,col+1,1]-a-b-c
    u0=col/(qc-1);du=1/(qc-1)
    qlow=min(a,a+b,a+c,a+b+c+d);qhigh=max(a,a+b,a+c,a+b+c+d)
    xlo=max(u0,tile[0]);xhi=min(u0+du,tile[2]);ylo=max(qlow,tile[1]);yhi=min(qhigh,tile[3])
    if xhi<=xlo or yhi<=ylo:return float('inf')
    left=max(co,int(np.floor(xlo*(nc-1))));right=min(co+layer.columns-2,int(np.floor(xhi*(nc-1))))
    top=max(ro,int(np.floor(ylo*(nr-1))));bottom=min(ro+layer.rows-2,int(np.floor(yhi*(nr-1))))
    minimum=float('inf')
    for sc in range(left,right+1):
        tl=max(0.,(max(sc/(nc-1),tile[0])-u0)/du)
        th=min(1.,(min((sc+1)/(nc-1),tile[2])-u0)/du)
        if th<=tl:continue
        for sr in range(top,bottom+1):
            yl=max(sr/(nr-1),tile[1]);yh=min((sr+1)/(nr-1),tile[3])
            if yh<=yl:continue
            g00,g10,g01,g11=layer.grid[sr-ro,sc-co],layer.grid[sr-ro,sc-co+1],layer.grid[sr-ro+1,sc-co],layer.grid[sr-ro+1,sc-co+1]
            bx=g10-g00;cy=g01-g00;twist=g11-g10-g01+g00
            j0=(cross(bx,cy)+cross(bx,twist)*(u0*(nc-1)-sc)-cross(twist,cy)*sr)*(nr-1)*(nc-1)
            j1=cross(bx,twist)*du*(nc-1)**2*(nr-1)
            j2=cross(twist,cy)*(nr-1)**2*(nc-1)
            split=[tl,th]
            for k,l in [(a,b),(a+c,b+d)]:
                if abs(l)>1e-15:
                    for y in (yl,yh):
                        t=(y-k)/l
                        if tl<t<th:split.append(t)
            split=sorted(set(split))
            for lo,hi in zip(split,split[1:]):
                mid=(lo+hi)/2
                if max(yl,a+b*mid)>min(yh,a+c+(b+d)*mid)+1e-14:continue
                for k,l in [(yl,0) if yl>a+b*mid else (a,b),(yh,0) if yh<a+c+(b+d)*mid else (a+c,b+d)]:
                    aa=j0+j2*k;bb=j1+j2*l
                    quadratic=bb*d;linear=aa*d+bb*c
                    ts=[lo,hi]
                    if quadratic>0:
                        vertex=-linear/(2*quadratic)
                        if lo<vertex<hi:ts.append(vertex)
                    minimum=min(minimum,*[(aa+bb*t)*(c+d*t)*(qr-1) for t in ts])
    return float(minimum)


def validate_normal_composition(parent,query,regions,*,minimum=.2):
    query=np.asarray(query,dtype=float);qr,qc=query.shape[:2]
    if query.ndim!=3 or query.shape[2]!=2 or min(qr,qc)<2 or not np.isfinite(query).all():raise ValueError('Invalid normal warp')
    if not np.allclose(query[:,:,0],(np.arange(qc)/(qc-1))[None,:],atol=1e-14,rtol=0):raise ValueError('Normal warp changed its tangent coordinate')
    if (np.diff(query[:,:,1],axis=0)<=0).any():raise ValueError('Normal warp is not ordered')
    lower=float('inf');uncertain=0;checked=0
    for region in regions:
        left=max(0,int(np.floor(region[0]*(qc-1))));right=min(qc-1,int(np.ceil(region[2]*(qc-1))))
        top=max(0,int(np.floor(region[1]*(qr-1))));bottom=min(qr-1,int(np.ceil(region[3]*(qr-1))))
        block=query[top:bottom+1,left:right+1]
        low=np.minimum.reduce([block[:-1,:-1,1],block[:-1,1:,1],block[1:,:-1,1],block[1:,1:,1]])
        high=np.maximum.reduce([block[:-1,:-1,1],block[:-1,1:,1],block[1:,:-1,1],block[1:,1:,1]])
        normal=np.minimum(np.diff(block[:,:,1],axis=0)[:,:-1],np.diff(block[:,:,1],axis=0)[:,1:])*(qr-1)
        x0=np.broadcast_to(np.arange(left,right)[None,:]/(qc-1),low.shape);x1=x0+1/(qc-1)
        domain=[region[0],float(low.min()),region[2],float(high.max())]
        for layer,tile in owned_rectangles(parent,[domain]):
            active=(x1>tile[0])&(x0<tile[2])&(high>tile[1])&(low<tile[3])
            yy,xx=np.where(active)
            if not len(xx):continue
            nr,nc,ro,co=(*[layer.rows,layer.columns],0,0) if layer.window is None else layer.window
            quality=layer_quality(layer)
            lx=np.clip(np.floor(np.maximum(x0[active],tile[0])*(nc-1)).astype(int)-co,0,layer.columns-2)
            rx=np.clip(np.floor(np.minimum(x1[active],tile[2])*(nc-1)).astype(int)-co,0,layer.columns-2)
            ty=np.clip(np.floor(np.maximum(low[active],tile[1])*(nr-1)).astype(int)-ro,0,layer.rows-2)
            by=np.clip(np.floor(np.minimum(high[active],tile[3])*(nr-1)).astype(int)-ro,0,layer.rows-2)
            bound=np.full(len(xx),np.inf)
            for dy in range(int((by-ty).max())+1):
                for dx in range(int((rx-lx).max())+1):
                    bound=np.minimum(bound,quality[np.minimum(ty+dy,by),np.minimum(lx+dx,rx)])
            bound*=normal[active]
            risky=bound<=minimum;uncertain+=int(risky.sum());checked+=len(xx)
            if (~risky).any():lower=min(lower,float(bound[~risky].min()))
            for y,x in zip(yy[risky],xx[risky]):
                lower=min(lower,exact_cell(layer,tile,query,int(y+top),int(x+left)))
    return {'accepted':bool(np.isfinite(lower) and lower>minimum),'continuous_lower_bound':lower,'required_minimum':minimum,'analytically_minimized_cells':uncertain,'checked_owned_cells':checked}


def validate_source_composition(parent,query,regions,*,minimum=.2):
    """Conservative continuous guard for a general query through source layers.

    Bilinear cell determinants and coordinate extrema attain their limits at
    corners. Multiplying independent positive lower bounds is conservative;
    an uncertain proposal is declined rather than relaxing the area gate.
    """
    from coordinate_stock import minimum_cell_jacobian
    query=np.asarray(query,dtype=float);qr,qc=query.shape[:2]
    query_quality=minimum_cell_jacobian(query)
    if (query_quality<=0).any():
        return {'accepted':False,'continuous_lower_bound':float(query_quality.min()),'required_minimum':minimum}
    lower=float('inf');checked=0
    for region in regions:
        left=max(0,int(np.floor(region[0]*(qc-1))));right=min(qc-1,int(np.ceil(region[2]*(qc-1))))
        top=max(0,int(np.floor(region[1]*(qr-1))));bottom=min(qr-1,int(np.ceil(region[3]*(qr-1))))
        block=query[top:bottom+1,left:right+1]
        corners=[block[:-1,:-1],block[:-1,1:],block[1:,:-1],block[1:,1:]]
        low=np.minimum.reduce(corners);high=np.maximum.reduce(corners)
        domain=[*low.min(axis=(0,1)),*high.max(axis=(0,1))]
        for layer,tile in owned_rectangles(parent,[domain]):
            active=(high[:,:,0]>tile[0])&(low[:,:,0]<tile[2])&(high[:,:,1]>tile[1])&(low[:,:,1]<tile[3])
            if not active.any():continue
            nr,nc,ro,co=(*[layer.rows,layer.columns],0,0) if layer.window is None else layer.window
            quality=layer_quality(layer)
            lx=np.clip(np.floor(np.maximum(low[:,:,0][active],tile[0])*(nc-1)).astype(int)-co,0,layer.columns-2)
            rx=np.clip(np.floor(np.minimum(high[:,:,0][active],tile[2])*(nc-1)).astype(int)-co,0,layer.columns-2)
            ty=np.clip(np.floor(np.maximum(low[:,:,1][active],tile[1])*(nr-1)).astype(int)-ro,0,layer.rows-2)
            by=np.clip(np.floor(np.minimum(high[:,:,1][active],tile[3])*(nr-1)).astype(int)-ro,0,layer.rows-2)
            bound=np.full(len(lx),np.inf)
            for dy in range(int((by-ty).max())+1):
                for dx in range(int((rx-lx).max())+1):
                    bound=np.minimum(bound,quality[np.minimum(ty+dy,by),np.minimum(lx+dx,rx)])
            combined=bound*query_quality[top:bottom,left:right][active]
            lower=min(lower,float(combined.min()));checked+=len(lx)
    return {'accepted':bool(np.isfinite(lower) and lower>minimum),'continuous_lower_bound':lower,'required_minimum':minimum,'checked_owned_cells':checked}
