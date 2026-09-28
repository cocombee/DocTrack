"""Recognize separate printed sections flanking one measured word row.

The gap is word evidence, never invented rule ink. Only an already straight,
registered section near the word baseline can supply the common target.
"""
import numpy as np
from text_column_alignment import independent_owner_links


def supported_registered_sections(guides, registered, word_mask, safe, width, height):
    """Use a complete earlier trace only when current ink supports both ends.

    At least 95% of the trace must agree with current measured sections. Small
    weak-stock gaps cannot contain words, and never create new source points.
    """
    scale = np.array([width - 1, height - 1], dtype=float)
    if word_mask.shape != (height, width) or safe.shape != word_mask.shape:
        raise ValueError('Registered sections need the same real raster evidence')
    result = []
    for trace in registered:
        trace = np.asarray(trace, dtype=float)
        if (len(trace) < 64 or np.any(np.diff(trace[:, 0]) <= 0) or
                np.ptp(trace[:, 1]) * scale[1] > 2.5 * max(width, height) / 1280):
            continue
        coverage = np.zeros(len(trace), dtype=bool);matches=[]
        for guide in guides:
            if len(guide) < 32:
                continue
            overlap = (trace[:, 0] >= guide[0, 0]) & (trace[:, 0] <= guide[-1, 0])
            error = abs(trace[:, 1] - np.interp(trace[:, 0], guide[:, 0], guide[:, 1])) * scale[1]
            hit = overlap & (error < 1.5 * max(width, height) / 1280)
            if hit.sum() >= 24:
                coverage |= hit;matches.append(guide)
        if (coverage.mean() < .95 or not any(a[-1, 0] < b[0, 0] or b[-1, 0] < a[0, 0]
                for i, a in enumerate(matches) for b in matches[i + 1:])):
            continue
        pixels=np.rint(trace*scale).astype(int)
        if ((pixels<0).any() or (pixels>=np.array([width,height])).any() or
                (safe[pixels[:,1],pixels[:,0]]<=250).any()):
            continue
        missing=pixels[~coverage]
        if any(word_mask[max(0,y-2):min(height,y+3),max(0,x-2):min(width,x+3)].any()
               for x,y in missing):
            continue
        blocks=np.split(trace[~coverage,0],np.flatnonzero(np.diff(trace[~coverage,0])*scale[0]>2)+1)
        if any(len(b) and (b[-1]-b[0])*scale[0]>12*max(width,height)/1280 for b in blocks):
            continue
        result.append(trace)
    return result


def observed_registered_sections(guides, registered, word_mask, safe, width, height):
    """Keep an old trace and append only its short, actually observed tails.

    A current section can continue a few pixels beyond its old registration.
    Those pixels must belong to the same measured section, with at least 95%
    overlapping the old stroke. Never extrapolate a registration into a tail.
    Conflicting observations, word ink and unsafe stock leave that end alone.
    """
    scale = np.array([width - 1, height - 1], dtype=float)
    pixel_scale = max(width, height) / 1280
    recovered = supported_registered_sections(
        guides, registered, word_mask, safe, width, height)
    result = []
    for trace in recovered:
        ends = [[], []]
        for guide in guides:
            guide = np.asarray(guide, dtype=float)
            if (guide.ndim != 2 or guide.shape[1] != 2 or len(guide) < 32 or
                    not np.isfinite(guide).all() or (np.diff(guide[:, 0]) <= 0).any()):
                continue
            covered = ((guide[:, 0] >= trace[0, 0] - 1e-10) &
                       (guide[:, 0] <= trace[-1, 0] + 1e-10))
            if covered.mean() < .95 or covered.sum() < 24:
                continue
            error = abs(guide[covered, 1] - np.interp(
                guide[covered, 0], trace[:, 0], trace[:, 1])) * scale[1]
            if error.max() > 1.5 * pixel_scale:
                continue
            for side, tail in enumerate((guide[guide[:, 0] < trace[0, 0] - 1e-10],
                                         guide[guide[:, 0] > trace[-1, 0] + 1e-10])):
                if not len(tail):
                    continue
                edge = trace[0 if side == 0 else -1, 0]
                if abs(tail[:, 0] - edge).max() * scale[0] > 12 * pixel_scale:
                    continue
                pixels = np.rint(tail * scale).astype(int)
                if ((pixels < 0).any() or (pixels >= [width, height]).any() or
                        (safe[pixels[:, 1], pixels[:, 0]] <= 250).any() or
                        any(word_mask[max(0, y-2):min(height, y+3),
                                      max(0, x-2):min(width, x+3)].any()
                            for x, y in pixels)):
                    continue
                ends[side].append(tail)
        additions = []
        for pieces in ends:
            if not pieces:
                additions.append(np.empty((0, 2)));continue
            # Multiple measurements must describe the same material points.
            # Refuse an ambiguous end rather than choosing one observation.
            points = np.vstack(pieces)
            points = points[np.argsort(points[:, 0], kind='stable')]
            unique = []; ambiguous = False
            for point in points:
                if unique and abs(point[0] - unique[-1][0]) < 1e-10:
                    if abs(point[1] - unique[-1][1]) > 1e-10:
                        ambiguous = True;break
                else:
                    unique.append(point)
            additions.append(np.empty((0, 2)) if ambiguous else np.asarray(unique))
        result.append(np.vstack([additions[0], trace, additions[1]]))
    return result


def word_flanked_rows(guides, quads, word_groups, native_count, anchors,
                      width, height):
    scale = np.array([width - 1, height - 1], dtype=float)
    if (scale <= 0).any():
        raise ValueError('Rule rows need a valid measurement basis')
    if (not isinstance(native_count, (int, np.integer)) or isinstance(native_count, bool)
            or not 0 <= native_count <= len(word_groups)):
        raise ValueError('Rule rows need an explicit current native owner boundary')
    pixel_scale = max(width, height) / 1280
    sections = []
    for guide in guides:
        guide = np.asarray(guide, dtype=float)
        if (guide.ndim != 2 or guide.shape[1] != 2 or len(guide) < 32 or
                not np.isfinite(guide).all() or np.any(np.diff(guide[:, 0]) <= 0)):
            continue
        if (np.ptp(guide[:, 0]) * scale[0] < 48 * pixel_scale or
                np.ptp(guide[:, 1]) * scale[1] > 2.5 * pixel_scale):
            continue
        # Prefer complete observed chains over their contained fragments.
        if any(q[0, 0] <= guide[0, 0] + .5/scale[0] and q[-1, 0] >= guide[-1, 0] - .5/scale[0] and
               abs(np.median(q[:, 1]) - np.median(guide[:, 1])) * scale[1] < 2
               for q in sections):
            continue
        sections.append(guide)
    sections.sort(key=lambda g: g[0, 0])
    quads = [np.asarray(q, dtype=float) for q in quads]
    if any(q.shape != (4, 2) or not np.isfinite(q).all() for q in quads):
        raise ValueError('Word rows need complete finite native quadrilaterals')
    candidates = []
    for left in sections:
        for right in sections:
            gap_low, gap_high = left[-1, 0], right[0, 0]
            if (gap_high - gap_low) * scale[0] < 8 * pixel_scale:
                continue
            levels = np.array([np.median(left[:, 1]), np.median(right[:, 1])])
            difference = np.ptp(levels) * scale[1]
            if difference <= 2.5 * pixel_scale:
                continue
            indices = []
            for index, quad in enumerate(quads):
                low, high = quad.min(0), quad.max(0)
                # Both flanks must end outside the actual native word box.
                if not gap_low <= low[0] < high[0] <= gap_high:
                    continue
                if np.all(levels >= low[1] - 2 / scale[1]) and np.all(
                        levels <= high[1] + 2 / scale[1]):
                    indices.append(index)
            if len(indices) < 2:
                continue
            indices.sort(key=lambda i: quads[i][:, 0].min())
            lows = np.array([quads[i].min(0) for i in indices])
            highs = np.array([quads[i].max(0) for i in indices])
            fonts = (highs[:, 1] - lows[:, 1]) * scale[1]
            baseline = np.array([quads[i][np.argsort(quads[i][:, 1])[-2:], 1].mean()
                                 for i in indices])
            if (fonts.min() < 6 * pixel_scale or fonts.max() > 1.3 * fonts.min() or
                    np.ptp(baseline) * scale[1] > 2 * pixel_scale or
                    difference > .8 * fonts.min()):
                continue
            intervals = np.r_[
                (lows[0, 0] - gap_low) * scale[0],
                (lows[1:, 0] - highs[:-1, 0]) * scale[0],
                (gap_high - highs[-1, 0]) * scale[0]]
            if (intervals < 0).any() or intervals.max() > fonts.min():
                continue
            coverage = np.sum(highs[:, 0] - lows[:, 0]) / (gap_high - gap_low)
            if coverage < .8:
                continue
            links = independent_owner_links([indices], quads, word_groups, native_count)
            if not links or any(len(word_groups[i]) < 12 for i in links[0]):
                continue
            # The upper edge of a letter box cannot be an assumed baseline.
            # Decline if both flanks could independently supply the target.
            near_baseline = abs(levels - np.median(baseline)) * scale[1] <= .45 * fonts.min()
            if near_baseline.sum() != 1:
                continue
            anchor_side = int(np.flatnonzero(near_baseline)[0])
            anchor_section = (left, right)[anchor_side]
            matches = []
            for index, anchor in enumerate(anchors):
                anchor = np.asarray(anchor, dtype=float)
                if (anchor.ndim != 2 or anchor.shape[1] != 2 or len(anchor) < 32 or
                        not np.isfinite(anchor).all() or
                        np.ptp(anchor[:, 1]) * scale[1] > .1):
                    continue
                overlap = min(anchor[:, 0].max(), anchor_section[-1, 0]) - max(
                    anchor[:, 0].min(), anchor_section[0, 0])
                if overlap < .8 * np.ptp(anchor_section[:, 0]):
                    continue
                target = float(np.median(anchor[:, 1]))
                if abs(target - levels[anchor_side]) * scale[1] <= 1.5 * pixel_scale:
                    matches.append((index, target))
            if not matches or np.ptp([m[1] for m in matches]) * scale[1] >= .1:
                continue
            candidates.append({
                'guides': [left, right], 'target': matches[0][1],
                'anchor_indices': [m[0] for m in matches],
                'anchor_side': anchor_side, 'word_indices': indices,
                'word_owners': links[0], 'word_coverage': float(coverage),
                'section_offset_px': float(difference),
                'basis': 'Two observed sections and complete native words; no rule points in word gap'})
    return candidates
