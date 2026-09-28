"""Repair stock in both source axes while keeping explicit ink constraints."""
import numpy as np
from local_receipt_mesh import sample


def crossing_pins(candidate, desired, source_rule, before, sources, sizes, axes):
    """Pin the new rule and preserve measured normals of crossing rules."""
    rows, cols = candidate.shape[:2]
    positions, targets = [np.asarray(desired)], [np.asarray(source_rule)]
    offsets = np.r_[0, np.cumsum(sizes)]
    for index, axis in enumerate(axes):
        low, high = offsets[index:index + 2]
        trace, source = before[low:high], sources[low:high]
        if axis is None or axis == 1:
            positions.append(trace)
        else:
            destinations = []
            for old_point, source_point in zip(trace, source):
                x = np.clip(old_point[0] * (cols - 1), 0, cols - 1 - 1e-9)
                col, fraction = int(x), x - int(x)
                column = (candidate[:, col, 1] * (1 - fraction) +
                          candidate[:, col + 1, 1] * fraction)
                if np.diff(column).min() <= 0:
                    raise ValueError('Crossing has no ordered tangent inverse')
                destinations.append([old_point[0], np.interp(
                    source_point[1], column, np.linspace(0, 1, rows))])
            positions.append(np.asarray(destinations).reshape(-1, 2))
        targets.append(source)
    return np.vstack(positions), np.vstack(targets)


def project(candidate, boxes, shifts, positions, targets, region,
            iterations=5000, audit_log=None):
    """Bounded local feasibility search; the caller still applies all gates."""
    if iterations <= 0:
        return candidate
    rows, cols = candidate.shape[:2]
    x0, y0, x1, y1 = region
    scale = (rows - 1) * (cols - 1)
    result = candidate.copy()
    free = np.zeros((rows, cols), np.uint8)
    free[max(0, int(np.ceil(y0 * (rows - 1)))):min(rows, int(y1 * (rows - 1)) + 1),
         max(0, int(np.ceil(x0 * (cols - 1)))):min(cols, int(x1 * (cols - 1)) + 1)] = 1
    for box, shift in zip(boxes, shifts):
        low, high = box.min(0) + [0, shift], box.max(0) + [0, shift]
        free[max(0, int(np.floor(low[1] * (rows - 1)))):
             min(rows, int(np.ceil(high[1] * (rows - 1))) + 1),
             max(0, int(np.floor(low[0] * (cols - 1)))):
             min(cols, int(np.ceil(high[0] * (cols - 1))) + 1)] = 0
    positions=np.asarray(positions);targets=np.asarray(targets)
    if (positions.ndim != 2 or positions.shape[1] != 2 or
            targets.shape != positions.shape or not np.isfinite(positions).all() or
            not np.isfinite(targets).all()):
        raise ValueError('Stock constraints must be finite paired coordinates')
    inside=(positions[:,0]>=x0)&(positions[:,0]<=x1)&(positions[:,1]>=y0)&(positions[:,1]<=y1)
    positions=positions[inside];targets=targets[inside]
    if not len(positions) or not free.any():
        return candidate
    dx,dy=np.diff(candidate,axis=1),np.diff(candidate,axis=0)
    area=np.minimum.reduce([(a[:,:,0]*b[:,:,1]-a[:,:,1]*b[:,:,0])*scale for a in (dx[:-1],dx[1:]) for b in (dy[:,:-1],dy[:,1:])])
    yy,xx=np.where(area<=.2)
    # Expand only for bad cells or ink pins whose initial residual needs repair.
    initial_residual=abs(sample(candidate,positions)-targets).max(1)
    active=positions[initial_residual>1e-7]
    if not len(yy) and not len(active):return result
    active_x=active[:,0]*(cols-1);active_y=active[:,1]*(rows-1)
    top=max(0,int(np.floor(min(active_y.min() if len(active) else rows,(yy.min() if len(yy) else rows))))-32)
    bottom=min(rows,int(np.ceil(max(active_y.max() if len(active) else 0,(yy.max() if len(yy) else 0))))+34)
    left=max(0,int(np.floor(min(active_x.min() if len(active) else cols,(xx.min() if len(xx) else cols))))-32)
    right=min(cols,int(np.ceil(max(active_x.max() if len(active) else 0,(xx.max() if len(xx) else 0))))+34)
    selected=(positions[:,0]*(cols-1)>=left)&(positions[:,0]*(cols-1)<right-1)&(positions[:,1]*(rows-1)>=top)&(positions[:,1]*(rows-1)<bottom-1)

    positions=positions[selected];targets=targets[selected]
    if not len(positions):
        return candidate
    coordinates=result[top:bottom,left:right];free=free[top:bottom,left:right].astype(float)
    u=coordinates[:,:,0];v=coordinates[:,:,1]
    def views(a):return [a[:-1,:-1],a[:-1,1:],a[1:,:-1],a[1:,1:]]
    edges=[(0,1,0,2),(0,1,1,3),(2,3,0,2),(2,3,1,3)];ff=views(free)
    xy=positions*[cols-1,rows-1]-[left,top];base=xy.astype(int);a,b=(xy-base).T
    row,col=base[:,1],base[:,0];width=u.shape[1]
    indexes=np.c_[row*width+col,row*width+col+1,(row+1)*width+col,(row+1)*width+col+1]
    weights=np.c_[(1-a)*(1-b),a*(1-b),(1-a)*b,a*b]
    gradients=weights*free.ravel()[indexes];pin_norm=(gradients*gradients).sum(1)
    initial=(coordinates.reshape(-1,2)[indexes]*weights[:,:,None]).sum(1)-targets
    immovable=pin_norm<=1e-30
    report={'pins':len(targets),'immovable_pin_error':float(abs(initial[immovable]).max()) if immovable.any() else 0.}

    previous=coordinates.copy()
    for iteration in range(iterations):
        start=coordinates.copy();worst=1.
        for aa,ab,ba,bb in edges:
            cv=views(coordinates);a=cv[ab]-cv[aa];b=cv[bb]-cv[ba]
            quality=(a[:,:,0]*b[:,:,1]-a[:,:,1]*b[:,:,0])*scale;worst=min(worst,float(quality.min()))
            ga=np.stack([b[:,:,1],-b[:,:,0]],-1)*scale;gb=np.stack([-a[:,:,1],a[:,:,0]],-1)*scale
            g=[np.zeros_like(a) for _ in range(4)];g[aa]-=ga;g[ab]+=ga;g[ba]-=gb;g[bb]+=gb
            g=[p*f[:,:,None] for p,f in zip(g,ff)];norm=sum((p*p).sum(2) for p in g)
            correction=np.maximum(.22-quality,0)*(norm>1e-30)/np.maximum(norm,1e-30)
            change=np.zeros_like(coordinates);count=np.zeros_like(free)
            for dst,den,gradient in zip(views(change),views(count),g):dst+=gradient*correction[:,:,None];den+=(correction!=0)*(np.linalg.norm(gradient,axis=2)!=0)
            coordinates+=1.2*change/np.maximum(count,1)[:,:,None]
        for _ in range(3):
            error=(coordinates.reshape(-1,2)[indexes]*weights[:,:,None]).sum(1)-targets
            correction=-error/np.maximum(pin_norm,1e-30)[:,None]
            count=np.bincount(indexes.ravel(),weights=(gradients!=0).ravel(),minlength=u.size)
            for axis in range(2):
                delta=np.bincount(indexes.ravel(),weights=(gradients*correction[:,axis,None]).ravel(),minlength=u.size)
                coordinates[:,:,axis]+=.9*(delta/np.maximum(count,1)).reshape(u.shape)
        coordinates+=.4*(start-previous)*free[:,:,None];previous=start
        pin_error=float(abs((coordinates.reshape(-1,2)[indexes]*weights[:,:,None]).sum(1)-targets).max())

        if worst>.205 and pin_error<1e-6:break
        if immovable.any() and report['immovable_pin_error']>1e-7:break
    report.update(iterations=iteration, minimum_area=worst, pin_error=pin_error,
                  max_coordinate_change=float(abs(result - candidate).max()))
    if audit_log is not None:
        audit_log.append({'stock_projection': report})
    return result
