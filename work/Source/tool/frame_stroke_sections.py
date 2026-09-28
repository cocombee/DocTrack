"""Observe complete thin frame strokes before assigning unrecognized ink.

Only actual connected pixels between two independently measured cross-rules
can qualify. A matching existing same-axis section is also mandatory. The
returned centers use observed cross-sections, never extrapolated endpoints.
"""
import cv2
import numpy as np


def frame_sections(gray, safe, guides, perpendicular, word_mask):
    h, w = gray.shape
    if safe.shape != gray.shape or word_mask.shape != gray.shape or min(h, w) < 2:
        raise ValueError('Frame strokes need paired raster and ownership evidence')
    scale = max(h, w) / 1280
    basis = np.array([w - 1, h - 1])
    contrast = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), max(2, 6 * scale)) - gray
    ink = ((contrast > 3) & (safe > 250) & (word_mask == 0)).astype(np.uint8)
    crossings = np.zeros(gray.shape, np.uint8)
    for guide in perpendicular:
        cv2.polylines(crossings, [np.rint(np.asarray(guide) * basis).astype(np.int32)],
                      False, 1, max(3, round(5 * scale)))
    ink[crossings > 0] = 0
    _, labels, stats, _ = cv2.connectedComponentsWithStats(ink)
    result = []
    for component, (x, y, cw, ch, area) in enumerate(stats[1:], start=1):
        if cw < max(64 * scale, w * .08) or cw < 6 * ch:
            continue
        body = labels[y:y + ch, x:x + cw] == component
        counts = body.sum(0)
        active = np.flatnonzero(counts)
        if len(active) / cw < .95 or np.quantile(counts[active], .95) > max(2, 6 * scale):
            continue
        profiles = [np.flatnonzero(body[:, col]) for col in active]
        if np.mean([len(q) == np.ptp(q) + 1 for q in profiles]) < .95:
            continue
        # Weak edges may continue a real stroke, but cannot constitute it.
        if np.count_nonzero((contrast[y:y + ch, x:x + cw] > 6) & body) < .7 * area:
            continue
        points = np.c_[active + x, [float(q.mean()) + y for q in profiles]]
        matches = []
        for index, value in enumerate(guides):
            guide = np.asarray(value)
            overlap = (points[:, 0] >= guide[0, 0] * basis[0]) & (points[:, 0] <= guide[-1, 0] * basis[0])
            if overlap.sum() < 48 * scale:
                continue
            error = abs(points[overlap, 1] - np.interp(points[overlap, 0],
                         guide[:, 0] * basis[0], guide[:, 1] * basis[1]))
            if np.quantile(error, .95) < 1.5 * scale:
                matches.append(index)
        if not matches:
            continue
        ends = []
        for point in (points[0], points[-1]):
            near = []
            for index, value in enumerate(perpendicular):
                guide = np.asarray(value) * basis
                vector = np.diff(guide, axis=0)
                length = np.sum(vector * vector, axis=1)
                valid = length > 0
                if not valid.any():
                    continue
                start, vector, length = guide[:-1][valid], vector[valid], length[valid]
                t = np.clip(np.sum((point - start) * vector, axis=1) / length, 0, 1)
                distance = np.linalg.norm(start + t[:, None] * vector - point, axis=1).min()
                if distance <= max(4, 6 * scale):
                    near.append(index)
            ends.append(near)
        if not any(a != b for a in ends[0] for b in ends[1]):
            continue
        result.append({'points': points / basis, 'matching_sections': matches,
                       'crossing_indices': ends, 'component': int(component),
                       'observed_pixels': int(area),
                       'basis': 'Thin connected ink between two measured frame crossings'})
    return result
