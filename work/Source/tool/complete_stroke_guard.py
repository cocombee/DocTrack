"""Preserve observed printed strokes across local coordinate corrections.

The registry contains material samples, never a chord inferred across missing
ink. Chords below measure bending between observed samples; they supply no
geometry to the solver. Both the original and last accepted frames are kept.
"""
import numpy as np

from composite_mesh_inverse import inverse_retained
from composite_mesh_sampling import MeshSampler
from content_stroke_ownership import connected_rule_traces


def _points(value):
    p = np.asarray(value, dtype=float)
    if (p.ndim != 2 or p.shape[1:] != (2,) or len(p) < 3
            or not np.isfinite(p).all()):
        raise ValueError('Complete strokes need finite observed material samples')
    return p


def capture_strokes(base, horizontal, vertical, gray, safe, word_mask):
    """Capture every measured section and only raster-proven continuations."""
    sampler = MeshSampler(base)
    owners = []
    for axis, sections in enumerate((horizontal, vertical)):
        traces = [dict(kind='rule', points=p[:, ::-1] if axis else p)
                  for p in sections]
        grouped, _ = connected_rule_traces(traces, gray, safe, word_mask, axis)
        for owner in grouped:
            p = np.asarray(owner['points'], dtype=float)
            tangent = p[:, axis]
            # A measurement cloud cannot become an ordered line by sorting.
            # Sorting is permitted only for the union of independently ordered
            # traces whose connection was established from actual raster ink.
            if len(owner['source_indices']) > 1:
                p = p[np.lexsort((p[:, 1-axis], tangent))]
                tangent = p[:, axis]
            elif np.all(np.diff(tangent) < 0):
                p = p[::-1]
                tangent = p[:, axis]
            if (len(p) < 3 or not np.isfinite(p).all()
                    or (np.diff(tangent) < 0).any() or np.ptp(tangent) <= 0):
                raise ValueError('A complete stroke must retain an ordered observed trace')
            owners.append(dict(axis=axis, source_indices=owner['source_indices'],
                               source_points=sampler.sample(p).tolist(),
                               reference_points=p.tolist()))
    if not owners:
        raise ValueError('Complete printed-stroke protection has no observed owners')
    return owners


def _line_error(points):
    centered = points - points.mean(0)
    _, _, directions = np.linalg.svd(centered, full_matrices=False)
    return float(abs(centered @ directions[-1]).max())


def bend_change(before, after):
    """Measure added bending at the whole-stroke and local observed scales.

    Point identities and order stay fixed. Every dyadic sample span and the
    complete span are checked; a lower global fitting error cannot conceal a
    new small kink. Distances are invariant under rigid motion.
    """
    before, after = _points(before), _points(after)
    if before.shape != after.shape:
        raise ValueError('Complete stroke material identities changed')
    if any((np.linalg.norm(np.diff(p,axis=0),axis=1) <= 1e-9).any()
           for p in (before,after)):
        raise ValueError('A printed-stroke span collapsed')
    worst = 0.
    spans = sorted({len(before)-1, *[2**k for k in range(1, len(before).bit_length())
                                   if 2**k < len(before)]})
    for span in spans:
        deviations = []
        for p in (before, after):
            chord = p[span:] - p[:-span]
            length = np.linalg.norm(chord, axis=1)
            if (length <= 1e-9).any():
                raise ValueError('A printed-stroke span collapsed')
            offset = p[span//2:len(p)-span+span//2] - p[:-span]
            deviations.append(abs(chord[:, 0]*offset[:, 1]
                                  - chord[:, 1]*offset[:, 0]) / length)
        worst = max(worst, float((deviations[1]-deviations[0]).max()))
    first, last = _line_error(before), _line_error(after)
    return dict(before_line_error_px=first, after_line_error_px=last,
                added_line_error_px=last-first, added_local_bend_px=worst,
                accepted=bool(max(last-first, worst) <= .1))


def audit_strokes(base, result, owners, pixel_size):
    """Audit exact maps against immutable origins and the last accepted frame."""
    if not owners:
        raise ValueError('Complete printed-stroke protection has no observed owners')
    scale = np.asarray(pixel_size, dtype=float) - 1
    if scale.shape != (2,) or not np.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError('Stroke audit needs a valid paired pixel size')
    sources = [_points(o['source_points']) for o in owners]
    references = [_points(o['reference_points']) for o in owners]
    if any(s.shape != r.shape for s, r in zip(sources, references)):
        raise ValueError('Complete stroke material identities changed')
    offsets = np.cumsum([len(p) for p in sources])[:-1]
    material = np.vstack(sources)
    before = np.split(inverse_retained(MeshSampler(base), material)*scale, offsets)
    after = np.split(inverse_retained(MeshSampler(result), material)*scale, offsets)
    checks = []
    for index, (owner, reference, old, new) in enumerate(zip(owners, references, before, after)):
        origin = bend_change(reference*scale, new)
        current = bend_change(old, new)
        checks.append(dict(index=index, axis=owner['axis'], samples=len(new),
                           original=origin, previous=current,
                           accepted=origin['accepted'] and current['accepted']))
    return checks


def current_stroke_guides(base, owners):
    """Return complete observed proposal guides, keeping every audit owner.

    At duplicate tangent coordinates a proposal uses one actual observed
    member. No audit sample is deleted, reset, averaged or extrapolated.
    Strokes that cease to be monotonic remain owners but cannot guide this
    one-axis solver.
    """
    if not owners:
        raise ValueError('Complete printed-stroke protection has no observed owners')
    sources=[_points(owner['source_points']) for owner in owners]
    offsets=np.cumsum([len(p) for p in sources])[:-1]
    current=np.split(inverse_retained(MeshSampler(base),np.vstack(sources)),offsets)
    guides=[[],[]]
    for owner,points in zip(owners,current):
        axis=owner['axis']
        if axis not in (0,1):raise ValueError('Invalid printed-stroke axis')
        trace=points if axis==0 else points[:,::-1]
        delta=np.diff(trace[:,0])
        if (delta < -1e-10).any():continue
        trace=trace[np.r_[True,delta>1e-10]]
        if len(trace)>=16:guides[axis].append(trace)
    return guides
