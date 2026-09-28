"""Group rule fragments only through continuous measured raster ink.

Endpoint proximity proposes a search corridor, never a printed bridge. Every
column of a join must contain the same connected thin ink component, outside
word owners and unsafe stock. Returned points include only observed samples.
"""
import itertools

import cv2
import numpy as np


def connected_rule_traces(traces, gray, safe, word_mask, axis):
    gray = np.asarray(gray)
    safe = np.asarray(safe)
    word_mask = np.asarray(word_mask)
    if (gray.ndim != 2 or min(gray.shape) < 16 or safe.shape != gray.shape
            or word_mask.shape != gray.shape or axis not in (0, 1)
            or not np.isfinite(gray).all() or not np.isfinite(safe).all()
            or not np.isfinite(word_mask).all()):
        raise ValueError('Stroke ownership needs paired measurement rasters')
    h, w = gray.shape
    basis = np.array([w - 1, h - 1])
    scale = max(h, w) / 1280
    contrast = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), max(2, 6 * scale)) - gray
    ink = (contrast > 3) & (safe > 250) & (word_mask == 0)
    # Work in increasing tangent coordinates for either direction family.
    if axis:
        ink, contrast = ink.T, contrast.T
    height, width = ink.shape
    radius = max(2, int(np.ceil(3 * scale)))
    pixels = {}
    for index, trace in enumerate(traces):
        points = np.asarray(trace.get('points', []), dtype=float)
        if (trace.get('kind') != 'rule' or points.ndim != 2
                or points.shape[1:] != (2,) or len(points) < 2
                or not np.isfinite(points).all() or (points < 0).any()
                or (points > 1).any()):
            continue
        p = points * basis
        if axis:
            p = p[:, ::-1]
        # Ordered traces, unlike ink clouds, must be monotonic on their axis.
        delta = np.diff(p[:, 0])
        if np.all(delta < 0):
            p = p[::-1]
        elif not np.all(delta > 0):
            continue
        pixels[index] = p
    parent = list(range(len(traces)))

    def root(index):
        while parent[index] != index:
            index = parent[index]
        return index

    links = []
    bridges = []
    for a, b in itertools.combinations(pixels, 2):
        pa, pb = pixels[a], pixels[b]
        if pa[0, 0] > pb[0, 0]:
            pa, pb = pb, pa
        # Only local continuations are eligible. A long missing stroke or a
        # network connection through an orthogonal rule supplies no ownership.
        gap = pb[0, 0] - pa[-1, 0]
        if gap > max(8, 64 * scale):
            continue
        if gap >= 0:
            endpoints = np.array([pa[-1], pb[0]])
            if abs(np.diff(endpoints[:, 1])[0]) > radius + .15 * gap:
                continue
            lo = max(0, int(np.floor(pa[-1, 0])) - 2)
            hi = min(width - 1, int(np.ceil(pb[0, 0])) + 2)
            xs = np.arange(lo, hi + 1)
            guide = np.interp(xs, endpoints[:, 0], endpoints[:, 1])
        else:
            lo = max(0, int(np.ceil(pb[0, 0])))
            hi = min(width - 1, int(np.floor(min(pa[-1, 0], pb[-1, 0]))))
            if hi - lo < 3:
                continue
            xs = np.arange(lo, hi + 1)
            ya, yb = np.interp(xs, pa[:, 0], pa[:, 1]), np.interp(xs, pb[:, 0], pb[:, 1])
            if np.max(abs(ya - yb)) > radius:
                continue
            guide = (ya + yb) / 2
        top = max(0, int(np.floor(guide.min())) - radius)
        bottom = min(height, int(np.ceil(guide.max())) + radius + 1)
        ys = np.arange(top, bottom)[:, None]
        corridor = abs(ys - guide[None]) <= radius
        body = ink[top:bottom, lo:hi + 1] & corridor
        count, labels, stats, _ = cv2.connectedComponentsWithStats(body.astype(np.uint8))
        qualifying = []
        for component in range(1, count):
            measured = labels == component
            counts = measured.sum(0)
            if not np.all(counts > 0) or np.max(counts) > max(3, 6 * scale):
                continue
            profiles = [np.flatnonzero(measured[:, col]) for col in range(len(xs))]
            if any(len(p) != np.ptp(p) + 1 for p in profiles):
                continue
            if np.mean(contrast[top:bottom, lo:hi + 1][measured] > 6) < .7:
                continue
            centers = np.array([p.mean() + top for p in profiles])
            if np.max(abs(centers - guide)) > max(1.5, 1.5 * scale):
                continue
            qualifying.append(np.c_[xs, centers])
        if len(qualifying) != 1:
            continue
        observed = qualifying[0]
        if axis:
            observed = observed[:, ::-1]
        observed = observed / basis
        parent[root(b)] = root(a)
        bridges.append((a, observed))
        links.append(dict(indices=[a, b], observed_samples=len(observed),
                          evidence='One continuous thin raster component'))
    groups = {}
    for index in range(len(traces)):
        groups.setdefault(root(index), []).append(index)
    result = []
    for owner, indices in groups.items():
        if len(indices) == 1:
            result.append(dict(traces[indices[0]], source_indices=indices))
            continue
        originals = [np.asarray(traces[i]['points'], dtype=float) for i in indices]
        observed = [p for i, p in bridges if root(i) == owner]
        result.append(dict(kind='rule', points=np.unique(np.vstack(originals + observed), axis=0),
                           source_indices=indices, source_sections=originals))
    return result, links
