"""Invert the actual bilinear cells with an analytic Jacobian and line search."""
import numpy as np
from local_receipt_mesh import sample


def _newton_inverse(grid,points):
    rows,cols=grid.shape[:2]
    estimate=np.clip(np.asarray(points).copy(),0,1)
    for _ in range(60):
        residual=sample(grid,estimate)-points
        error=np.sum(residual*residual,axis=1)
        if not len(error) or error.max()<1e-24:
            break
        xy=np.clip(estimate*[cols-1,rows-1],0,[cols-1-1e-9,rows-1-1e-9])
        lo=xy.astype(int);fraction=xy-lo
        x,y=lo.T;a,b=fraction.T
        g00,g10,g01,g11=grid[y,x],grid[y,x+1],grid[y+1,x],grid[y+1,x+1]
        dx=((1-b[:,None])*(g10-g00)+b[:,None]*(g11-g01))*(cols-1)
        dy=((1-a[:,None])*(g01-g00)+a[:,None]*(g11-g10))*(rows-1)
        determinant=dx[:,0]*dy[:,1]-dx[:,1]*dy[:,0]
        step=np.c_[(dy[:,1]*residual[:,0]-dy[:,0]*residual[:,1])/determinant,
                   (-dx[:,1]*residual[:,0]+dx[:,0]*residual[:,1])/determinant]
        pending=error>1e-24
        scale=np.ones(len(points))
        for _ in range(16):
            candidate=np.clip(estimate-step*scale[:,None],0,1)
            trial=sample(grid,candidate)-points
            improved=np.sum(trial*trial,axis=1)<=error
            take=pending&improved
            estimate[take]=candidate[take]
            pending &= ~take
            if not pending.any():
                break
            scale[pending]*=.5
    return estimate


def _cross(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def _cell_inverse(grid, point):
    """Find a unique inverse by solving the bilinear source-space cells.

    Local Newton steps can stall beside a steep but valid crease. Only cells
    containing the source point are candidates; every root must map back to
    that point. Shared-edge duplicates are allowed, distinct roots are not.
    """
    rows, cols = grid.shape[:2]
    solutions = []
    for start in range(0, rows - 1, 128):
        stop = min(rows - 1, start + 128)
        a, b = grid[start:stop, :-1], grid[start:stop, 1:]
        c, d = grid[start + 1:stop + 1, :-1], grid[start + 1:stop + 1, 1:]
        low = np.minimum.reduce([a, b, c, d])
        high = np.maximum.reduce([a, b, c, d])
        y, x = np.where(((point >= low - 1e-12) &
                         (point <= high + 1e-12)).all(2))
        if not len(x):
            continue
        origin = a[y, x]
        bx, cy = b[y, x] - origin, c[y, x] - origin
        twist = d[y, x] - b[y, x] - c[y, x] + origin
        offset = point - origin
        qa = -_cross(bx, twist)
        qb = _cross(offset, twist) - _cross(bx, cy)
        qc = _cross(offset, cy)
        discriminant = qb * qb - 4 * qa * qc
        linear = abs(qa) <= 1e-14 * (abs(qb) + abs(qc) + 1e-300)
        stable = -.5 * (qb + np.copysign(
            np.sqrt(np.maximum(discriminant, 0)), qb))
        with np.errstate(divide='ignore', invalid='ignore'):
            roots = [np.where(linear, -qc / qb, stable / qa),
                     np.where(linear, np.nan, qc / stable)]
            for u in roots:
                direction = cy + twist * u[:, None]
                residual = offset - bx * u[:, None]
                v = (residual * direction).sum(1) / (direction * direction).sum(1)
                error = abs(origin + bx * u[:, None] + cy * v[:, None] +
                            twist * (u * v)[:, None] - point).max(1)
                good = ((discriminant >= -1e-28) & np.isfinite(u) &
                        np.isfinite(v) & (u >= -1e-9) & (u <= 1 + 1e-9) &
                        (v >= -1e-9) & (v <= 1 + 1e-9) & (error < 1e-11))
                solutions.extend(np.c_[(x[good] + u[good]) / (cols - 1),
                    (start + y[good] + v[good]) / (rows - 1)])
    if not solutions:
        raise ValueError('Source point has no exact inverse inside the mesh')
    solutions = np.asarray(solutions)
    if np.ptp(solutions, axis=0).max() > 1e-8:
        raise ValueError('Source point has more than one inverse inside the mesh')
    return solutions.mean(0)


def inverse(grid, points):
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise ValueError('Inverse points must be finite coordinate pairs')
    result = _newton_inverse(grid, points)
    misses = np.flatnonzero(abs(sample(grid, result) - points).max(1) > 1e-10)
    for index in misses:
        result[index] = _cell_inverse(grid, points[index])
    return result
