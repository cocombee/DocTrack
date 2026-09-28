"""A straight projection reference independent of the photographed crop.

The first pinpoints and measured perimeter remain source/crop geometry. Four
straight side-angle references intersect into a second virtual frame, following
the ID method without assigning card dimensions or changing the paper contour.
This proposes camera geometry only; residual dewarping and ink QA follow it.
"""
import itertools
import numpy as np

from content_plane_evidence import transform_points


def _points(value, minimum):
    points = np.asarray(value, dtype=float)
    if (points.ndim != 2 or points.shape[1:] != (2,) or len(points) < minimum
            or not np.isfinite(points).all() or (points < 0).any() or (points > 1).any()):
        raise ValueError('Projection frame requires finite normalized source observations')
    return points


def _signed_area(points):
    following = np.roll(points, -1, axis=0)
    return float(np.sum(points[:, 0]*following[:, 1]-points[:, 1]*following[:, 0])/2)


def _side_angle(points, size, expected_length):
    pixels = points * (size - 1)
    tolerance = 2.2 * max(size) / 640
    indexes = np.unique(np.linspace(0, len(points)-1, 21).astype(int))
    best = None
    for a, b in itertools.combinations(indexes, 2):
        direction = pixels[b] - pixels[a]
        length = np.linalg.norm(direction)
        if length < expected_length * .4:
            continue
        normal = np.array([-direction[1], direction[0]]) / length
        error = abs((pixels-pixels[a]) @ normal)
        keep = error <= tolerance
        extent = float(np.ptp(pixels[keep] @ (direction/length))) if keep.any() else 0.
        score = (int(keep.sum()), extent, -float(error[keep].sum()))
        if best is None or score > best[0]:
            best = (score, keep)
    if best is None:
        raise ValueError('A side lacks a supported projected angle')
    keep = best[1]
    for _ in range(3):
        if keep.mean() < .6:
            raise ValueError('A side has insufficient straight angle support')
        center = pixels[keep].mean(0)
        _, _, directions = np.linalg.svd(pixels[keep]-center, full_matrices=False)
        normal = directions[-1]
        error = abs((pixels-center) @ normal)
        next_keep = error <= tolerance
        if np.array_equal(next_keep, keep):
            break
        keep = next_keep
    extent = float(np.ptp(pixels[keep] @ directions[0]))
    if keep.mean() < .6 or extent < expected_length * .6:
        raise ValueError('A projected side is supported by too short a source span')
    line = np.r_[normal, -normal @ center]
    return line, dict(supported_samples=np.flatnonzero(keep).tolist(),
        observed_samples=len(points), supported_fraction=float(keep.mean()),
        supported_extent_px=extent, residual_tolerance_px=float(tolerance),
        angle_radians=float(np.arctan2(directions[0,1], directions[0,0])),
        virtual_reference_only=True)


def second_projection_frame(first_pinpoints, side_observations, size, *, material_points=()):
    """Propose a second frame; never replace the first pins or measured mask.

    Side observations are ordered top/right/bottom/left measured edge samples.
    Excluded curled samples remain crop/source observations. Unverified frames
    raise ValueError; there is no fallback that pretends to supply four angles.
    """
    pins = _points(first_pinpoints, 4)
    size = np.asarray(size, dtype=float)
    if (pins.shape != (4,2) or size.shape != (2,) or not np.isfinite(size).all()
            or (size < 16).any() or _signed_area(pins) <= 0 or len(side_observations) != 4):
        raise ValueError('A projection reference needs four ordered sides and crop pins')
    sides = [_points(points, 16) for points in side_observations]
    lines, evidence = [], []
    for edge, points in enumerate(sides):
        expected = np.linalg.norm((pins[(edge+1)%4]-pins[edge])*(size-1))
        line, report = _side_angle(points, size, expected)
        lines.append(line); evidence.append(report)
    virtual = []
    for edge in range(4):
        intersection = np.cross(lines[(edge-1)%4], lines[edge])
        if abs(intersection[2]) < 1e-10:
            raise ValueError('Projected side angles have no stable corner intersection')
        virtual.append(intersection[:2]/intersection[2]/(size-1))
    virtual = np.asarray(virtual)
    edges = np.roll(virtual,-1,axis=0)-virtual
    turns = edges[:,0]*np.roll(edges,-1,axis=0)[:,1]-edges[:,1]*np.roll(edges,-1,axis=0)[:,0]
    if not (turns > 0).all() or _signed_area(virtual) <= 0:
        raise ValueError('The second projection frame is not a convex camera plane')
    if np.max(np.linalg.norm((virtual-pins)*(size-1),axis=1)) > .15*max(size):
        raise ValueError('The projected reference is unsupported near the crop')
    square = np.array([[0,0],[1,0],[1,1],[0,1]],float)
    rows, targets = [], []
    for (x,y),(u,v) in zip(virtual,square):
        rows.extend([[x,y,1,0,0,0,-u*x,-u*y], [0,0,0,x,y,1,-v*x,-v*y]])
        targets.extend([u,v])
    matrix = np.r_[np.linalg.solve(rows,targets),1.].reshape(3,3)
    observed = [pins,*sides]
    if len(material_points):
        observed.append(_points(material_points,1))
    mapped = transform_points(matrix,np.vstack(observed))
    if np.linalg.det(matrix) <= 0:
        raise ValueError('The projection reference mirrors original material')
    return dict(first_pinpoints=pins.tolist(), first_pinpoint_role='Source location and crop',
        second_reference_corners=virtual.tolist(), second_reference_role='Straight angle-projected camera frame',
        side_angle_evidence=evidence, source_to_reference_matrix=matrix.tolist(),
        observed_source_bounds_in_reference=[*mapped.min(0),*mapped.max(0)],
        source_boundary_changed=False, source_material_changed=False,
        virtual_lines_are_printed_ink=False, initial_projection_qualified=False,
        residual_dewarp_applied=False)
