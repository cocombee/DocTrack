"""Whole-word tilt evidence when blur connects the individual glyphs.

The measured ink cloud remains the complete geometric owner. These independent
contrast profiles only decide whether a rigid correction has raster support.
"""
import cv2
import numpy as np


def envelope_consensus(gray, group, *, pixel_size=None):
    gray = np.asarray(gray)
    group = np.asarray(group, dtype=float)
    if (gray.ndim != 2 or min(gray.shape) < 2 or not np.isfinite(gray).all()
            or group.ndim != 2 or group.shape[1] != 2 or not len(group)
            or not np.isfinite(group).all() or (group < 0).any()
            or (group > 1).any()):
        return None
    h, w = gray.shape
    basis = np.asarray([w, h] if pixel_size is None else pixel_size, dtype=float)
    if basis.shape != (2,) or not np.isfinite(basis).all() or (basis <= 0).any():
        return None
    pixels = (np.floor(group * basis) if pixel_size is None
              else np.rint(group * basis)).astype(int)
    if ((pixels < 0).any() or (pixels >= [w, h]).any()):
        return None
    low, high = pixels.min(0), pixels.max(0)
    if high[0] <= low[0] or high[1] <= low[1]:
        return None
    mask = np.zeros((high[1] - low[1] + 1, high[0] - low[0] + 1), np.uint8)
    local_points = pixels - low
    mask[local_points[:, 1], local_points[:, 0]] = 1
    scale = max(h, w) / 1280
    contrast = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), max(2, 7 * scale)) - gray
    local = contrast[low[1]:high[1] + 1, low[0]:high[0] + 1]
    step = max(4, int(round(mask.shape[0] * .5)))
    profiles = []
    for threshold in (4, 6, 8, 10, 12, 14, 16):
        parts = []
        for x in range(0, mask.shape[1], step):
            weights = np.where((mask[:, x:x + step] > 0)
                               & (local[:, x:x + step] > threshold),
                               local[:, x:x + step] - threshold, 0).sum(1)
            total = float(weights.sum())
            if total < 15 * scale * scale:
                continue
            cumulative = np.cumsum(weights) / total
            top = np.interp(.1, cumulative, np.arange(len(weights))) + low[1]
            bottom = np.interp(.9, cumulative, np.arange(len(weights))) + low[1]
            parts.append([x + low[0] + min(step, mask.shape[1] - x) / 2, top, bottom])
        if len(parts) < 6:
            continue
        anchors = np.asarray(parts)
        coverage = np.ptp(anchors[:, 0]) / (high[0] - low[0])
        top_slope = np.polyfit(anchors[:, 0], anchors[:, 1], 1)[0]
        bottom_slope = np.polyfit(anchors[:, 0], anchors[:, 2], 1)[0]
        center = anchors[:, 1:3].mean(1)
        slope, intercept = np.polyfit(anchors[:, 0], center, 1)
        errors = center - slope * anchors[:, 0] - intercept
        scatter = float(np.std(errors))
        explained = float(1 - np.mean(errors ** 2) / max(np.var(center), 1e-12))
        excursion = float(abs(slope) * np.ptp(anchors[:, 0]))
        edge_difference = float(abs(np.arctan(top_slope) - np.arctan(bottom_slope)))
        if (coverage <= .75 or edge_difference >= .012 or scatter >= .7 * scale
                or explained <= .7 or excursion <= 1.5 * scale):
            continue
        profiles.append({'threshold': threshold, 'angle': float(np.arctan(slope)),
                         'slope': float(slope), 'intercept': float(intercept),
                         'scatter_px': scatter, 'explained': explained,
                         'excursion_px': excursion, 'coverage': float(coverage),
                         'top_bottom_angle_difference': edge_difference,
                         'points': np.c_[anchors[:, 0], center].tolist()})
    # Three levels must agree. A single sharp fragment or font edge cannot
    # authorize a whole-word rotation, and no full owner is discarded.
    for first in profiles:
        agreement = [q for q in profiles if abs(q['angle'] - first['angle']) < .003]
        if len(agreement) < 3:
            continue
        slope = float(np.median([q['slope'] for q in agreement]))
        return {**first, 'angle': float(np.arctan(slope)), 'slope': slope,
                'independent_profiles': agreement,
                'basis': 'Full-width paired cap and baseline contrast envelopes'}
    return None
