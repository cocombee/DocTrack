"""Measure a bend against an adjacent supported, already straight section."""
import numpy as np


def before_stable_tail(guide, normal_pixels, tolerance_px=1.5):
    """Return only the bent prefix and its fixed join, or decline the guide.

    The majority tail must already fit one normal coordinate within measurement
    tolerance. No target comes from the mean of the bend or a distant crossing.
    The join is a measured point; the accepted tail remains outside the patch.
    """
    points = np.asarray(guide, dtype=float)
    if (points.ndim != 2 or points.shape[1] != 2 or len(points) < 32 or
            not np.isfinite(points).all() or np.any(np.diff(points[:, 0]) <= 0) or
            not np.isfinite(normal_pixels) or normal_pixels <= 0):
        return None
    baseline = float(np.median(points[int(len(points) * .55):, 1]))
    stable = abs(points[:, 1] - baseline) * normal_pixels <= tolerance_px
    suffix = np.logical_and.accumulate(stable[::-1])[::-1]
    choices = np.flatnonzero(suffix)
    if not len(choices):
        return None
    join = int(choices[0])
    if join < 4 or join >= len(points) * .55:
        return None
    target = float(points[join, 1])
    prefix = points[:join + 1]
    if abs(prefix[:, 1] - target).max() * normal_pixels <= 2:
        return None
    return prefix.copy(), target


def anchored_remainder(guide, anchors, tangent_pixels, normal_pixels):
    """Correct a visible continuation without moving its established anchor."""
    points = np.asarray(guide, dtype=float)
    if (points.ndim != 2 or points.shape[1] != 2 or len(points) < 16 or
            not np.isfinite(points).all() or np.any(np.diff(points[:, 0]) <= 0) or
            not np.isfinite([tangent_pixels, normal_pixels]).all() or
            min(tangent_pixels, normal_pixels) <= 0):
        return None
    choices = []
    for trace in anchors:
        trace = np.asarray(trace)
        if (trace.ndim != 2 or trace.shape[1] != 2 or not len(trace) or
                not np.isfinite(trace).all() or
                np.ptp(trace[:, 1]) * normal_pixels > .1):
            continue
        low, high = trace[:, 0].min(), trace[:, 0].max()
        overlap = (points[:, 0] >= low) & (points[:, 0] <= high)
        if overlap.sum() < 12 or np.ptp(points[overlap, 0]) < .02:
            continue
        target = float(np.median(trace[:, 1]))
        error = np.percentile(abs(points[overlap, 1] - target), 95) * normal_pixels
        if error >= 2:
            continue
        remaining = points[(points[:, 0] < low - 1 / tangent_pixels) |
                           (points[:, 0] > high + 1 / tangent_pixels)]
        if (len(remaining) < 5 or
                abs(remaining[:, 1] - target).max() * normal_pixels <= .75):
            continue
        choices.append((error, -int(overlap.sum()), remaining, target))
    if not choices:
        return None
    _, _, remaining, target = min(choices, key=lambda choice: choice[:2])
    return remaining.copy(), target
