"""Retain compact measured print that native OCR did not observe.

This supplies conservative ink owners, never text recognition or new rulers.
Native/retained owners stay immutable; partially owned or rule-crossed bodies
are not reassigned. Callers must retain and audit these bodies like other ink.
"""
import cv2
import numpy as np


def unrecognized_groups(gray, groups, safe, horizontal, vertical):
    h, w = gray.shape
    if min(h, w) < 2 or np.asarray(safe).shape != gray.shape:
        raise ValueError('Unrecognized print needs paired safe raster evidence')
    scale = max(h, w) / 1280
    contrast = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), max(2, 7 * scale)) - gray
    supported = contrast > 6
    _, labels, stats, _ = cv2.connectedComponentsWithStats(supported.astype(np.uint8))
    known = np.zeros(gray.shape, np.uint8)
    for group in groups:
        group = np.asarray(group, dtype=float)
        if (group.ndim != 2 or group.shape[1] != 2 or not len(group)
                or not np.isfinite(group).all() or (group < 0).any() or (group > 1).any()):
            raise ValueError('Invalid existing print owner')
        xy = np.rint(group * [w - 1, h - 1]).astype(int)
        known[xy[:, 1], xy[:, 0]] = 1
    occupied = np.bincount(labels[known > 0], minlength=len(stats))
    structural = np.zeros(gray.shape, np.uint8)
    for guide in [*horizontal, *[np.asarray(q)[:, ::-1] for q in vertical]]:
        guide = np.asarray(guide, dtype=float)
        if (guide.ndim != 2 or guide.shape[1] != 2 or len(guide) < 2
                or not np.isfinite(guide).all()):
            raise ValueError('Invalid measured rule trace')
        cv2.polylines(structural, [np.rint(guide * [w - 1, h - 1]).astype(np.int32)],
                      False, 1, max(3, round(3 * scale)))
    components = []
    for comp, (x, y, cw, ch, area) in enumerate(stats[1:], start=1):
        if (occupied[comp] or not max(3, 3 * scale) <= ch <= 38 * scale
                or not 2 <= cw <= min(2.4 * ch, w * .04)
                or area < max(6, 6 * scale * scale)):
            continue
        region = labels[y:y + ch, x:x + cw] == comp
        if ((np.asarray(safe)[y:y + ch, x:x + cw][region] <= 250).any()
                or structural[y:y + ch, x:x + cw][region].any()
                or np.count_nonzero(contrast[y:y + ch, x:x + cw][region] > 10) < 3):
            continue
        yy, xx = np.where(region)
        points = np.c_[xx + x, yy + y] / [w - 1, h - 1]
        components.append({'component': comp, 'bounds': [int(x), int(y), int(cw), int(ch)],
                           'points': points})
    # Adjacent character bodies can form a missed number/word. A substantial
    # gap, a different row or different glyph height keeps independent owners.
    parent = list(range(len(components)))

    def root(i):
        while parent[i] != i:
            i = parent[i]
        return i

    order = sorted(range(len(components)), key=lambda i: components[i]['bounds'][0])
    for pos, i in enumerate(order):
        x, y, cw, ch = components[i]['bounds']
        for j in order[pos + 1:]:
            xx, yy, ww, hh = components[j]['bounds']
            gap = xx - (x + cw)
            if gap > .75 * max(ch, hh):
                continue
            if (gap >= -1 and max(ch, hh) <= 1.4 * min(ch, hh)
                    and abs(y + ch / 2 - yy - hh / 2) <= .25 * max(ch, hh)):
                parent[root(j)] = root(i)
    owners = {}
    for i, component in enumerate(components):
        owners.setdefault(root(i), []).append(component)
    result = [np.unique(np.vstack([q['points'] for q in owner]), axis=0)
              for owner in owners.values()]
    return result, {'measured_owners': [
        {'components': [q['component'] for q in owner],
         'component_bounds_px': [q['bounds'] for q in owner], 'points': len(group)}
        for owner, group in zip(owners.values(), result)],
        'existing_owners_changed': False, 'recognition_inferred': False}
