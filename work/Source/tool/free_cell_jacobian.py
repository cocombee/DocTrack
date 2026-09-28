"""Repair a near-threshold mesh cell using only unprinted lattice nodes.

The caller owns the fixed-node mask and must include every material-owner
cell corner and the page boundary. This limited repair cannot rescue a fold
or replace print evidence; it rejects candidates that need a larger move.
"""
import numpy as np


def _determinants(grid):
    rows, columns = grid.shape[:2]
    du, dv = np.diff(grid, axis=1), np.diff(grid, axis=0)
    return np.stack([(a[..., 0]*b[..., 1]-a[..., 1]*b[..., 0])
                     *(rows-1)*(columns-1)
                     for a in (du[:-1], du[1:])
                     for b in (dv[:, :-1], dv[:, 1:])])


def _cell_determinants(grid, y, x):
    rows, columns = grid.shape[:2]
    q = grid[y:y+2, x:x+2]
    top, bottom = q[0, 1]-q[0, 0], q[1, 1]-q[1, 0]
    left, right = q[1, 0]-q[0, 0], q[1, 1]-q[0, 1]
    return (np.array([a[0]*b[1]-a[1]*b[0]
                      for a in (top, bottom) for b in (left, right)])
            *(rows-1)*(columns-1))


def repair_free_cells(grid, fixed_nodes, pixel_size, *, target=.204,
                      maximum_move_px=.05, maximum_steps=80):
    """Return the original field on refusal; never move a protected node."""
    original = np.asarray(grid, dtype=float)
    fixed = np.asarray(fixed_nodes)
    size = np.asarray(pixel_size, dtype=float)
    if (original.ndim != 3 or original.shape[2] != 2 or
            min(original.shape[:2]) < 3 or not np.isfinite(original).all() or
            fixed.shape != original.shape[:2] or fixed.dtype != bool or
            size.shape != (2,) or not np.isfinite(size).all() or
            (size <= 1).any() or not 0 < target < 1 or
            not 0 < maximum_move_px <= .1 or
            not isinstance(maximum_steps, int) or maximum_steps < 1):
        raise ValueError('Invalid free-cell repair input')
    if not (fixed[[0, -1], :].all() and fixed[:, [0, -1]].all()):
        raise ValueError('Free-cell repair requires fixed page boundary')
    basis = size-1
    initial = float(_determinants(original).min())
    if initial >= target:
        return original.copy(), dict(accepted=True, initial_minimum=initial,
                                     final_minimum=initial, steps=0,
                                     maximum_free_node_move_px=0.,
                                     maximum_fixed_node_move_px=0.)
    if initial < .18:
        return original.copy(), dict(accepted=False, initial_minimum=initial,
                                     final_minimum=initial, steps=0,
                                     reason='Beyond bounded near-threshold repair')
    field = original.copy()
    accepted_steps = 0
    for _ in range(maximum_steps):
        evidence = _determinants(field)
        minimum = float(evidence.min())
        if minimum >= target:
            break
        corner, y, x = np.unravel_index(evidence.argmin(), evidence.shape)
        current = _cell_determinants(field, y, x)[corner]
        positions, gradients = [], []
        for yy in (y, y+1):
            for xx in (x, x+1):
                if fixed[yy, xx]:
                    continue
                for axis in (0, 1):
                    epsilon = 1e-4/basis[axis]
                    field[yy, xx, axis] += epsilon
                    derivative = (_cell_determinants(field, y, x)[corner]
                                  - current)/1e-4
                    field[yy, xx, axis] -= epsilon
                    positions.append((yy, xx, axis))
                    gradients.append(derivative)
        direction = np.asarray(gradients)
        norm = float(np.linalg.norm(direction))
        if not positions:
            break
        improved = False
        directions = []
        if norm >= 1e-9:
            directions.append(direction/norm)
        # Several bilinear corners can tie for the minimum. Their individual
        # gradients may disagree; coordinate trials find a shared improvement.
        for index in range(len(positions)):
            for sign in (-1., 1.):
                unit = np.zeros(len(positions))
                unit[index] = sign
                directions.append(unit)
        for direction in directions:
            for step in (.005, .0025, .00125, .000625, .0003125):
                for (yy, xx, axis), value in zip(positions, direction):
                    field[yy, xx, axis] += step*value/basis[axis]
                worst_move = max(abs(field[yy, xx, axis]-original[yy, xx, axis])
                                 *basis[axis] for yy, xx, axis in positions)
                next_minimum = float(_determinants(field).min())
                if worst_move <= maximum_move_px and next_minimum > minimum+1e-8:
                    accepted_steps += 1
                    improved = True
                    break
                for (yy, xx, axis), value in zip(positions, direction):
                    field[yy, xx, axis] -= step*value/basis[axis]
            if improved:
                break
        if not improved:
            break
    final = float(_determinants(field).min())
    fixed_move = float(np.max(np.abs((field-original)[fixed]*basis)))
    free_move = (float(np.max(np.abs((field-original)[~fixed]*basis)))
                 if (~fixed).any() else 0.)
    accepted = bool(final >= target and fixed_move == 0 and
                    free_move <= maximum_move_px and
                    np.isfinite(field).all() and field.min() >= 0 and
                    field.max() <= 1)
    report = dict(accepted=accepted, initial_minimum=initial,
                  final_minimum=final, steps=accepted_steps,
                  maximum_free_node_move_px=free_move,
                  maximum_fixed_node_move_px=fixed_move)
    if not accepted:
        report['reason'] = 'Bounded blank-node moves cannot certify geometry'
    return (field if accepted else original.copy()), report
