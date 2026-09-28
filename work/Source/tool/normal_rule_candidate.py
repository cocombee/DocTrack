"""Optional vertical-rule step, preserving prior ink and horizontal normals.

The normal map is ordered piecewise, rather than a small smooth displacement
which can fold between a bent rule and a nearby stationary word. Measurements
and word masks retain the document's actual reading direction.
"""
import cv2
import numpy as np

import section_row_candidate as rows
from printed_rule_sections import measure_sections, word_ink_mask

p = rows.p


def ink_boxes(gray, safe, horizontal, vertical, words):
    """Protect non-rule ink, including isolated numerals and small glyph parts."""
    h, w = gray.shape
    contrast = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 7) - gray
    ink = (contrast > 10).astype(np.uint8)
    ink[safe < 250] = 0
    rule_mask = np.zeros_like(ink)
    for guide in horizontal + vertical:
        cv2.polylines(rule_mask, [np.rint(guide * [w - 1, h - 1]).astype(np.int32)], False, 1, 5)
    ink[(rule_mask > 0) & (words == 0)] = 0
    _, _, stats, _ = cv2.connectedComponentsWithStats(ink)
    # Tiny horizontal remnants at a measured junction have the same normal
    # as their supported rule. They may slide tangentially with that rule;
    # treating them as rigid glyph boxes would lock the whole intersection.
    for x, y, cw, ch, area in stats[1:]:
        if ch > max(3, round(h * .002)) or cw < 8 or area < cw / 2:
            continue
        center = (x + cw / 2) / (w - 1)
        for guide in horizontal:
            gap = max(guide[:, 0].min() - (x + cw - 1) / (w - 1),
                      x / (w - 1) - guide[:, 0].max(), 0)
            expected = np.interp(center, guide[:, 0], guide[:, 1]) * (h - 1)
            if gap * w > max(8, w * .1) or abs(y + (ch - 1) / 2 - expected) >= 2:
                continue
            if gap * w >= 8:
                # Long-rule summaries trim their ends. Extend only across
                # actual continuous ink, not across an empty gap or a word.
                edge = guide[0] if center < guide[0, 0] else guide[-1]
                near = x + cw - 1 if center < guide[0, 0] else x
                xs = np.arange(round(min(near, edge[0] * (w - 1))),
                               round(max(near, edge[0] * (w - 1))) + 1)
                ys = np.rint(np.interp(xs / (w - 1), guide[:, 0], guide[:, 1]) * (h - 1)).astype(int)
                strengths = np.maximum.reduce([contrast[np.clip(ys + offset, 0, h - 1), xs]
                                                for offset in (-1, 0, 1)])
                if np.mean(strengths > 3) < .9 or words[ys, xs].any():
                    continue
            view = ink[y:y + ch, x:x + cw]
            view[words[y:y + ch, x:x + cw] == 0] = 0
            break
    _, _, stats, _ = cv2.connectedComponentsWithStats(ink)
    boxes = []
    for x, y, cw, ch, area in stats[1:]:
        if area < 5:
            continue
        # A connected frame's empty interior is not a rigid glyph rectangle.
        # Keep its actual residual ink below, rather than pinning the whole
        # page-width bounding box where it contains no ink at all.
        if ch > h * .05 or cw > w * .2:
            continue
        boxes.append([x / (w - 1), y / (h - 1),
                      (x + cw - 1) / (w - 1), (y + ch - 1) / (h - 1)])
        ink[y:y + ch, x:x + cw] = 0
    return np.asarray(boxes, float).reshape(-1, 4), ink


def same_trace(first, second, width):
    lo = max(first[0, 1], second[0, 1])
    hi = min(first[-1, 1], second[-1, 1])
    if hi <= lo:
        return False
    y = np.linspace(lo, hi, 65)
    a = np.interp(y, first[:, 1], first[:, 0])
    b = np.interp(y, second[:, 1], second[:, 0])
    return np.percentile(abs(a - b), 95) * width < 1.5


def ordered_knots(intervals):
    """Join compatible fixed bands, reject contradictory ordering."""
    merged = []
    for d0, d1, s0, s1 in sorted(intervals):
        if merged and d0 <= merged[-1][1] + 1e-12:
            previous = merged[-1]
            if abs((d0 - s0) - (previous[0] - previous[2])) > 1e-12:
                return None
            merged[-1] = (previous[0], max(previous[1], d1),
                          previous[2], max(previous[3], s1))
        else:
            merged.append((d0, d1, s0, s1))
    knots = np.asarray(sorted(set((d, s) for d0, d1, s0, s1 in merged
                                 for d, s in ((d0, s0), (d1, s1)))))
    dest, source = knots.T
    if np.any(np.diff(dest) <= 0) or np.any(np.diff(source) <= 0):
        return None
    return dest, source


def refine(path, base, *, audit_log=None, selected_guide=None):
    gray, safe, source, old = rows.current_measurement(path, base)
    h, w = gray.shape
    words = word_ink_mask(gray)
    horizontal = measure_sections(gray, safe, word_mask=words)
    crossing = np.zeros_like(gray)
    for guide in horizontal:
        cv2.polylines(crossing, [np.rint(guide * [w - 1, h - 1]).astype(np.int32)], False, 1, 9)
    vertical = [g[:, ::-1] for g in measure_sections(gray.T, safe.T,
                                                   word_mask=words.T,
                                                   crossing_mask=crossing.T)]
    # A printed grid supplies cross-axis evidence; bars or solitary borders do
    # not qualify merely because there are many dark parallel strokes.
    if (sum(np.ptp(g[:, 0]) > .3 for g in horizontal) < 3 or
            sum(np.ptp(g[:, 1]) > .2 for g in vertical) < 3):
        return base
    # The long-rule detector also covers short intersections where section
    # tracking stops. It supplies an ink-classification mask only: the actual
    # measured primary guide still drives the correction, and word ink wins
    # wherever a line classification would cross a glyph.
    mask_h = horizontal + p.reference.extract_rules(gray)
    mask_v = vertical + [g[:, ::-1] for g in p.reference.extract_rules(gray.T)]
    if selected_guide is not None:
        mask_v.append(selected_guide)
    boxes, residual_ink = ink_boxes(gray, safe, mask_h, mask_v, words)
    if not len(boxes):
        return base
    corners = np.asarray([[[x0, y0], [x1, y0], [x1, y1], [x0, y1],
                           [(x0 + x1) / 2, y0], [(x0 + x1) / 2, y1]]
                          for x0, y0, x1, y1 in boxes])
    body = p.sample(old, corners.reshape(-1, 2))
    py, px = np.where(residual_ink)
    ink_points = np.c_[px / (w - 1), py / (h - 1)]
    residual_body = p.sample(old, ink_points)
    source_h = p.sample(old, np.vstack(horizontal))
    frozen = base.get('established_rulers', [])
    frozen_current = [(r['normal_axis'], p.inverse(old, np.asarray(r['points'])))
                      for r in frozen]
    shapes = rows.accepted.accepted.circular_landmarks(source) + p.reference.barcode_rulers(source)
    candidates = []
    for guide in vertical if selected_guide is None else [selected_guide]:
        qualifies, evidence = rows.accepted.accepted.residual_evidence(guide[:, ::-1], w)
        if qualifies and np.ptp(guide[:, 1]) > .1:
            candidates.append((evidence['excursion_px'], guide, evidence))
    candidates.sort(key=lambda item: -item[0])
    for _, guide, evidence in candidates:
        source_rule = p.sample(old, guide)
        target = float(guide[:, 0].mean())
        low_y, high_y = guide[0, 1], guide[-1, 1]
        y0, y1 = max(0, low_y - .02), min(1, high_y + .02)
        x0 = max(0, min(target, guide[:, 0].min()) - .035)
        x1 = min(1, max(target, guide[:, 0].max()) + .035)
        # End fades fit in blank stock, never through nearby stationary ink.
        for endpoint, before in ((guide[0], True), (guide[-1], False)):
            near = abs(ink_points[:, 0] - endpoint[0]) < 2 / w
            near_boxes = (boxes[:, 0] - .002 < endpoint[0]) & (boxes[:, 2] + .002 > endpoint[0])
            if before:
                ys = ink_points[near & (ink_points[:, 1] < low_y), 1]
                by = boxes[near_boxes & (boxes[:, 3] < low_y), 3]
                if len(ys) + len(by):
                    y0 = max(y0, float(np.r_[ys, by].max()) + 1 / h)
            else:
                ys = ink_points[near & (ink_points[:, 1] > high_y), 1]
                by = boxes[near_boxes & (boxes[:, 1] > high_y), 1]
                if len(ys) + len(by):
                    y1 = min(y1, float(np.r_[ys, by].min()) - 1 / h)
        if not y0 < low_y < high_y < y1:
            if audit_log is not None:
                audit_log.append({'position': guide.mean(0).tolist(), 'rejected': 'no blank endpoint fade'})
            continue
        other_v = [g for g in vertical if not same_trace(guide, g, w)]
        fixed_v = [g for axis, g in frozen_current if axis == 0] + other_v
        for normal_resolution in (max(old.shape[1], 1025), max(old.shape[1], 2049)):
            nr, nc = max(old.shape[0], 1025), normal_resolution
            u, v = np.meshgrid(np.linspace(0, 1, nc), np.linspace(0, 1, nr))
            field = u.copy()
            ribbon = 1.25 / w
            possible = True
            for row, y in enumerate(v[:, 0]):
                if not y0 < y < y1:
                    continue
                observed = float(np.interp(y, guide[:, 1], guide[:, 0]))
                amount = float(p.window(np.array([y]), y0, low_y, high_y, y1)[0])
                destination = observed + (target - observed) * amount
                intervals = [(x0, x0, x0, x0), (x1, x1, x1, x1),
                             (destination - ribbon, destination + ribbon,
                              observed - ribbon, observed + ribbon)]
                active = boxes[(boxes[:, 1] - .002 <= y) & (boxes[:, 3] + .002 >= y)]
                for b0, _, b1, _ in active:
                    left, right = max(x0, b0 - .001), min(x1, b1 + .001)
                    if left <= right:
                        intervals.append((left, right, left, right))
                sy = round(y * (h - 1))
                hits = np.flatnonzero(residual_ink[max(0, sy - 1):min(h, sy + 2)].any(axis=0))
                for run in np.split(hits, np.where(np.diff(hits) > 1)[0] + 1):
                    if not len(run):
                        continue
                    left = max(x0, run[0] / (w - 1) - .0006)
                    right = min(x1, run[-1] / (w - 1) + .0006)
                    if left <= right:
                        intervals.append((left, right, left, right))
                for fixed in fixed_v:
                    if fixed[:, 1].min() - .001 <= y <= fixed[:, 1].max() + .001:
                        q = fixed[np.argsort(fixed[:, 1])]
                        normal = float(np.interp(y, q[:, 1], q[:, 0]))
                        left, right = max(x0, normal - ribbon), min(x1, normal + ribbon)
                        if left <= right:
                            intervals.append((left, right, left, right))
                knots = ordered_knots(intervals)
                if knots is None:
                    possible = False
                    if audit_log is not None:
                        audit_log.append({'rejected': 'overlapping fixed band',
                                          'row_px': y * h,
                                          'guide_band_px': [observed * w, destination * w],
                                          'intervals_px': (np.asarray(intervals) * w).tolist()})
                    break
                inside = (u[row] >= x0) & (u[row] <= x1)
                field[row, inside] = np.interp(u[row, inside], *knots)
            if not possible:
                if audit_log is not None:
                    audit_log.append({'position': guide.mean(0).tolist(), 'rejected': 'fixed ink ordering'})
                break
            candidate = p.sample(old, np.c_[field.ravel(), v.ravel()]).reshape(nr, nc, 2)
            if not p.valid(candidate):
                if audit_log is not None:
                    audit_log.append({'position': guide.mean(0).tolist(), 'rejected': 'mesh topology', 'columns': nc})
                continue
            after = p.inverse(candidate, source_rule)
            error = float(np.max(abs(after[:, 0] - target)) * w)
            new_body = p.inverse(candidate, body).reshape(-1, 6, 2)
            movement = float(np.max(abs(new_body - corners) * [w, h]))
            if len(ink_points):
                movement = max(movement, float(np.max(abs(p.inverse(candidate, residual_body) - ink_points) * [w, h])))
            normal = float(np.max(abs(p.inverse(candidate, source_h)[:, 1] - np.vstack(horizontal)[:, 1])) * h)
            guards = [p.sample(old, np.vstack(other_v))] if other_v else []
            guards += shapes
            other_shift = max((float(np.max(abs(p.inverse(candidate, g) - p.inverse(old, g)) * [w, h])) for g in guards), default=0)
            established = max((float(np.max(abs(p.inverse(candidate, np.asarray(r['points']))[:, r['normal_axis']] - current[:, r['normal_axis']])) * [w, h][r['normal_axis']]) for r, (_, current) in zip(frozen, frozen_current)), default=0)
            take = error < .6 * max(h, w) / 1280 and max(movement, normal, other_shift, established) < .1
            audit = {**evidence, 'accepted': bool(take), 'position': guide.mean(0).tolist(),
                     'full_rule_error_px': error, 'fixed_ink_shift_px': movement,
                     'horizontal_normal_shift_px': normal, 'other_structure_shift_px': other_shift,
                     'established_normal_shift_px': established, 'columns': nc}
            if audit_log is not None:
                audit_log.append(audit)
            if not take:
                continue
            return {**base, 'preserved_mesh': base,
                    'refinement_regions': [[np.floor(x0 * (nc - 1)) / (nc - 1),
                                            np.floor(y0 * (nr - 1)) / (nr - 1),
                                            np.ceil(x1 * (nc - 1)) / (nc - 1),
                                            np.ceil(y1 * (nr - 1)) / (nr - 1)]],
                    'rows': nr, 'columns': nc, 'grid': candidate.reshape(-1, 2).tolist(),
                    'method': 'Experimental ordered vertical-rule correction',
                    'normal_rule_audit': audit,
                    'established_rulers': [*frozen, {'normal_axis': 0, 'points': source_rule.tolist()}]}
    return base
