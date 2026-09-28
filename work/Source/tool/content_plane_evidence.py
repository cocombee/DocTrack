"""Estimate a printed plane without using the paper silhouette as warp evidence.

This optional estimator only proposes a projective map. Local bends remain
separate residual evidence. Callers must retain the original photo and perform
ink/topology/render audits before promoting any map.
"""
import itertools
import numpy as np


def transform_points(matrix, points):
    points = np.asarray(points, dtype=float)
    q = np.c_[points, np.ones(len(points))] @ np.asarray(matrix).T
    if not np.isfinite(q).all() or np.any(q[:, 2] <= 1e-8):
        raise ValueError('Unbounded printed plane')
    return q[:, :2] / q[:, 2, None]


def _measure(trace, size, index):
    kind = trace.get('kind')
    if kind not in ('rule', 'text_baseline', 'shape_stroke'):
        return None, 'not printed evidence'
    points = np.asarray(trace.get('points', []), dtype=float)
    support = trace.get('glyph_baseline_support', {})
    measured_text = (kind == 'text_baseline' and isinstance(support, dict)
                     and support.get('basis') == 'Native row with measured raster glyph bottoms'
                     and isinstance(support.get('native_word_count'), int)
                     and support['native_word_count'] >= 3
                     and support.get('glyph_count') == len(points))
    # A rule contains dense stroke samples; a text baseline contains one
    # independently measured anchor per glyph, never interpolated samples.
    minimum = 6 if measured_text else 16
    if (points.ndim != 2 or points.shape[1:] != (2,) or len(points) < minimum
            or not np.isfinite(points).all()):
        return None, 'insufficient measured trace'
    pixels = points * (size - 1)
    center = pixels.mean(axis=0)
    _, singular, vectors = np.linalg.svd(pixels - center, full_matrices=False)
    direction = vectors[0]
    extent = float(np.ptp((pixels - center) @ direction))
    residual = np.abs((pixels - center) @ vectors[1])
    if extent < max(size) * .09:
        return None, 'short trace'
    if residual.max() > max(1., max(size) / 1280):
        return None, 'curved trace'
    # A short bent fragment cannot disappear in a much longer owner. Check
    # every original section against the common physical-stroke fit.
    for section in trace.get('source_sections', []):
        offset = np.asarray(section) * (size - 1) - center
        if np.max(np.abs(offset @ vectors[1])) > max(1., max(size) / 1280):
            return None, 'curved physical stroke'
    # Use isotropic image coordinates to compare direction errors in pixels.
    scale = max(size)
    p = pixels / scale
    c = p.mean(axis=0)
    line = np.r_[vectors[1], -vectors[1] @ c]
    return dict(index=index, line=line, center=c, points=p, extent=extent,
                source_indices=trace.get('source_indices', [index]),
                p90_px=float(np.percentile(residual, 90)),
                maximum_px=float(residual.max())), None


def _direction_error(vanishing, trace):
    vector = vanishing[:2] - vanishing[2] * trace['center']
    length = np.linalg.norm(vector)
    if length < 1e-10:
        return np.inf
    return abs(trace['line'][:2] @ vector) / length * trace['extent']


def _independent(traces, size):
    """Collinear fragments count as one physical line, not multiple votes."""
    chosen = []
    for trace in sorted(traces, key=lambda t: -t['extent']):
        duplicate = False
        for previous in chosen:
            sine = abs(np.linalg.det([trace['line'][:2], previous['line'][:2]]))
            distance = max(abs(trace['line'] @ np.r_[previous['center'], 1]),
                           abs(previous['line'] @ np.r_[trace['center'], 1]))
            if sine < np.sin(np.deg2rad(.2)) and distance * max(size) < 2:
                duplicate = True
                break
        if not duplicate:
            chosen.append(trace)
    return chosen


def _family(traces, size):
    traces = _independent(traces, size)
    if len(traces) < 3:
        return None, 'fewer than three independent printed lines'
    centers = np.asarray([t['center'] for t in traces])
    normals = np.asarray([t['line'][:2] for t in traces])
    normal = normals[0]
    normal *= np.sign(normal @ normals.mean(axis=0)) or 1
    if np.ptp(centers @ normal) < .15:
        return None, 'printed lines cover too little of the page'
    # Two independent lines hypothesize a finite or infinite vanishing point.
    # Additional measured lines must agree; a single corner cannot define it.
    best = None
    weights = np.asarray([t['extent'] for t in traces])
    for a, b in itertools.combinations(traces, 2):
        vp = np.cross(a['line'], b['line'])
        if np.linalg.norm(vp) < 1e-10:
            continue
        vp /= np.linalg.norm(vp)
        errors = np.asarray([_direction_error(vp, t) for t in traces])
        keep = errors <= 2
        score = (int(keep.sum()), float(weights[keep].sum()),
                 -float(np.sum(np.minimum(errors, 10) * weights)))
        if best is None or score > best[0]:
            best = (score, keep, vp)
    if best is None:
        return None, 'no independent direction estimate'
    keep, vp = best[1:]
    for _ in range(3):
        if keep.sum() < 3:
            break
        lines = np.asarray([t['line'] for t in traces])[keep]
        _, _, vectors = np.linalg.svd(lines * np.sqrt(weights[keep, None]),
                                      full_matrices=True)
        proposed = vectors[-1]
        errors = np.asarray([_direction_error(proposed, t) for t in traces])
        next_keep = errors <= 2
        if next_keep.sum() < keep.sum():
            break
        vp, keep = proposed, next_keep
    errors = np.asarray([_direction_error(vp, t) for t in traces])
    if (keep.sum() < 3 or keep.mean() < .7
            or weights[keep].sum() / weights.sum() < .7):
        return None, 'conflicting printed line directions'
    if np.ptp(centers[keep] @ normal) < .15:
        return None, 'consensus covers too little of the page'
    return dict(vanishing=vp, traces=traces, keep=keep, errors=errors), None


def estimate_printed_plane(horizontal, vertical, size, *, gray=None, safe=None,
                           word_mask=None):
    """Return a conservative proposal plus explicit reasons for abstention.

    Input points are normalized photo/measurement coordinates. The supplied
    size is [width, height]. Silhouette points are intentionally not accepted.
    Text baselines or verified shape strokes may supplement printed rules;
    two independent direction families are required for a projective estimate.
    """
    size = np.asarray(size, dtype=float)
    if size.shape != (2,) or not np.isfinite(size).all() or np.any(size < 16):
        raise ValueError('Invalid measurement size')
    report = dict(accepted=False, matrix=None, used=[], rejected=[])
    if any(value is not None for value in (gray, safe, word_mask)):
        if any(value is None for value in (gray, safe, word_mask)):
            raise ValueError('Physical stroke evidence needs all three rasters')
        if np.asarray(gray).shape != tuple(size[::-1].astype(int)):
            raise ValueError('Stroke raster must match the measurement size')
        from content_stroke_ownership import connected_rule_traces
        horizontal, hl = connected_rule_traces(horizontal, gray, safe, word_mask, 0)
        vertical, vl = connected_rule_traces(vertical, gray, safe, word_mask, 1)
        report['ownership'] = [dict(axis=axis, groups=[t['source_indices'] for t in raw],
                                   links=links)
                               for axis, raw, links in ((0, horizontal, hl), (1, vertical, vl))]
    families = []
    for axis, raw in enumerate((horizontal, vertical)):
        traces = []
        for index, trace in enumerate(raw):
            fitted, reason = _measure(trace, size, index)
            if fitted is None:
                report['rejected'].append(dict(axis=axis, index=index,
                    source_indices=trace.get('source_indices', [index]), reason=reason))
            else:
                traces.append(fitted)
        family, reason = _family(traces, size)
        if family is None:
            report['reason'] = f'direction family {axis}: {reason}'
            return report
        families.append(family)
        for t, keep, error in zip(family['traces'], family['keep'], family['errors']):
            record = dict(axis=axis, index=t['index'], direction_error_px=float(error),
                          source_indices=t['source_indices'],
                          straight_fit_p90_px=t['p90_px'],
                          straight_fit_maximum_px=t['maximum_px'])
            if keep:
                report['used'].append(record)
            else:
                report['rejected'].append(dict(**record, reason='direction outlier'))
    vh, vv = [f['vanishing'] for f in families]
    horizon = np.cross(vh, vv)
    if abs(horizon[2]) < 1e-8:
        report['reason'] = 'horizon crosses the document'
        return report
    horizon /= horizon[2]
    projective = np.eye(3)
    projective[2] = horizon
    camera_scale = (size - 1) / max(size)
    corners = np.array([[0, 0], [camera_scale[0], 0], camera_scale,
                        [0, camera_scale[1]]])
    denominator = np.c_[corners, np.ones(4)] @ horizon
    if np.min(denominator) < .5 or np.max(denominator) > 1.5:
        report['reason'] = 'unstable perspective over the paper area'
        return report
    directions = np.column_stack(((projective @ vh)[:2], (projective @ vv)[:2]))
    directions /= np.linalg.norm(directions, axis=0)
    directions[:, 0] *= np.sign(directions[0, 0]) or 1
    directions[:, 1] *= np.sign(directions[1, 1]) or 1
    if np.linalg.det(directions) < .5:
        report['reason'] = 'direction families do not define a stable printed plane'
        return report
    affine = np.eye(3)
    affine[:2, :2] = np.linalg.inv(directions)
    matrix = affine @ projective
    center = camera_scale / 2
    # Preserve the local measurement scale and center instead of letting an
    # irregular outline choose new corner targets or interior deformation.
    q = matrix @ np.r_[center, 1]
    jacobian = (matrix[:2, :2] * q[2] - q[:2, None] * matrix[2, :2]) / q[2]**2
    scale = np.linalg.norm(jacobian, axis=1)
    matrix[:2] /= scale[:, None]
    mapped_center = transform_points(matrix, center[None])[0]
    shift = np.eye(3)
    shift[:2, 2] = center - mapped_center
    matrix = shift @ matrix
    to_camera = np.diag([*camera_scale, 1])
    normalized = np.linalg.inv(to_camera) @ matrix @ to_camera
    report.update(accepted=True, matrix=normalized.tolist(), reason='independent printed plane',
                  vanishing_points=[vh.tolist(), vv.tolist()])
    return report


def guard_printed_plane(proposal, owners, size, *, unknown_owners=()):
    """Reject a plane proposal that changes any immutable complete ink owner.

    Plane consensus alone cannot authorize deformation. This check retains the
    established .1px shape/pair limits and stricter .03px unknown-print limit.
    Caller-owned arrays and the original proposal remain unchanged.
    """
    report = dict(proposal)
    if not report.get('accepted'):
        return report
    size = np.asarray(size, dtype=float)
    if size.shape != (2,) or not np.isfinite(size).all() or np.any(size < 16):
        raise ValueError('Invalid material measurement size')
    records = []
    for kind, clouds, limit in (('word', owners, .1), ('unknown print', unknown_owners, .03)):
        for index, cloud in enumerate(clouds):
            cloud = np.asarray(cloud, dtype=float)
            if (cloud.ndim != 2 or cloud.shape[1:] != (2,) or len(cloud) < 3
                    or not np.isfinite(cloud).all() or (cloud < 0).any() or (cloud > 1).any()):
                raise ValueError('Invalid immutable material owner')
            before = cloud * size
            after = transform_points(report['matrix'], cloud) * size
            x, y = before - before.mean(0), after - after.mean(0)
            u, _, vt = np.linalg.svd(x.T @ y)
            rotation = u @ vt
            shape = float(np.linalg.norm(x @ rotation - y, axis=1).max())
            pair = 0.
            # Compare every pair, using bounded scratch memory for large ink.
            for start in range(0, len(cloud), 128):
                old_distance = np.linalg.norm(x[start:start + 128, None] - x[None], axis=2)
                new_distance = np.linalg.norm(y[start:start + 128, None] - y[None], axis=2)
                pair = max(pair, float(np.max(abs(old_distance - new_distance))))
            records.append(dict(kind=kind, index=index, shape_error_px=shape,
                                pair_error_px=pair, limit_px=limit,
                                accepted=bool(np.linalg.det(rotation) > 0 and shape <= limit and pair <= limit)))
    valid = bool(records) and all(r['accepted'] for r in records)
    report['material_audit'] = dict(accepted=valid, owners=records)
    if not valid:
        report.update(accepted=False, matrix=None, reason='original material geometry declined printed plane')
    return report
