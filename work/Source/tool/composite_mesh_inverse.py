"""Local inverse and exact derivatives of retained bilinear mesh layers.

The caller supplies a nearby solution. This never substitutes a sampled
composite grid, whose finite cells can fold even when both actual maps do not.
"""
import numpy as np
import copy
from composite_mesh_sampling import MeshSampler


def differential(sampler, points):
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise ValueError('Mesh queries must be finite paired coordinates')
    active = np.ones(len(points), bool) if sampler.parent is None else np.zeros(len(points), bool)
    if sampler.parent is not None:
        for x0, y0, x1, y1 in sampler.regions:
            active |= ((points[:, 0] >= x0) & (points[:, 0] <= x1) &
                       (points[:, 1] >= y0) & (points[:, 1] <= y1))
    result = np.empty((len(points), 2, 2), dtype=float)
    if (~active).any():
        result[~active] = differential(sampler.parent, points[~active])
    query = points[active]
    if not len(query):
        return result
    if sampler.window is None:
        rows, cols, ro, co = sampler.rows, sampler.columns, 0, 0
    else:
        rows, cols, ro, co = sampler.window
    x = np.clip(query[:, 0]*(cols-1), 0, cols-1-1e-9)-co
    y = np.clip(query[:, 1]*(rows-1), 0, rows-1-1e-9)-ro
    ix = np.clip(np.floor(x).astype(int), 0, sampler.columns-2)
    iy = np.clip(np.floor(y).astype(int), 0, sampler.rows-2)
    fx, fy = x-ix, y-iy
    corners = [sampler.grid[iy, ix], sampler.grid[iy, ix+1],
               sampler.grid[iy+1, ix], sampler.grid[iy+1, ix+1]]
    if sampler.coordinate_transform:
        for index, (dx, dy) in enumerate([(0,0),(1,0),(0,1),(1,1)]):
            corners[index] = corners[index]-np.c_[(ix+co+dx)/(cols-1), (iy+ro+dy)/(rows-1)]
    a,b,c,d = corners
    du = ((b-a)*(1-fy[:,None])+(d-c)*fy[:,None])*(cols-1)
    dv = ((c-a)*(1-fx[:,None])+(d-b)*fx[:,None])*(rows-1)
    jacobian = np.stack([du,dv],axis=2)
    if sampler.coordinate_transform:
        weights = [(1-fx)*(1-fy),fx*(1-fy),(1-fx)*fy,fx*fy]
        mapped = query+sum(g*w[:,None] for g,w in zip(corners,weights))
        jacobian += np.eye(2)[None,:,:]
        jacobian = differential(sampler.parent,mapped) @ jacobian
    result[active] = jacobian
    return result


def inverse_near(sampler, sources, initial, *, tolerance=1e-11, iterations=20):
    """Refine nearby inverse points, rejecting ambiguous or failed solutions."""
    sources = np.asarray(sources,dtype=float)
    position = np.array(initial,dtype=float,copy=True)
    if sources.shape != position.shape or sources.ndim != 2 or sources.shape[1] != 2:
        raise ValueError('Inverse needs paired source and initial points')
    if not np.isfinite(sources).all() or not np.isfinite(position).all():
        raise ValueError('Inverse points must be finite')
    for _ in range(iterations):
        residual = sampler.sample(position)-sources
        active = np.max(abs(residual),axis=1)>tolerance
        if not active.any():
            return position
        jacobian = differential(sampler,position[active])
        if (np.linalg.det(jacobian)<=0).any():
            raise ValueError('Inverse encountered a folded local map')
        step = np.linalg.solve(jacobian,residual[active,:,None])[:,:,0]
        scale = np.maximum(1,np.max(abs(step),axis=1)/.02)
        position[active] -= step/scale[:,None]
        if (position < -1e-9).any() or (position > 1+1e-9).any():
            raise ValueError('Inverse left the document domain')
    if np.max(abs(sampler.sample(position)-sources),initial=0)>tolerance:
        raise ValueError('Inverse did not converge')
    return position


class _ExactIdentity(MeshSampler):
    def __init__(self):
        super().__init__({'rows':2,'columns':2,'grid':[[0,0],[1,0],[0,1],[1,1]]})

    def sample(self,points):
        return np.array(points,dtype=float,copy=True)


def inverse_retained(sampler,sources):
    """Invert each retained coordinate layer before its actual parent.

    A sampled composition can contain aliased folds and multiple apparent
    inverse cells. Such a grid must never be the initial inverse certificate.
    Coordinate primitives remain separate, ordered maps throughout inversion.
    """
    from precise_mesh_inverse import inverse
    if sampler.coordinate_transform:
        before=inverse_retained(sampler.parent,sources)
        primitive=getattr(sampler,'_inverse_primitive',None)
        if primitive is None:
            primitive=copy.copy(sampler);primitive.parent=_ExactIdentity()
            sampler._inverse_primitive=primitive
        seed=getattr(primitive,'_inverse_seed',None)
        if seed is None:
            rows,cols=(sampler.rows,sampler.columns) if sampler.window is None else sampler.window[:2]
            seed=primitive.dense(rows,cols);primitive._inverse_seed=seed
        return inverse_near(primitive,before,inverse(seed,before))
    seed=getattr(sampler,'_inverse_seed',None)
    if seed is None:
        dimensions=[];node=sampler
        while node is not None:
            dimensions.append((node.rows,node.columns) if node.window is None else node.window[:2])
            node=node.parent
        seed=sampler.dense(max(s[0] for s in dimensions),max(s[1] for s in dimensions))
        sampler._inverse_seed=seed
    return inverse_near(sampler,sources,inverse(seed,sources))
