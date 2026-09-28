"""Measure visible printed-rule sections; stop at actual neighbouring ink."""
import cv2
import numpy as np

def word_ink_mask(gray):
    h, w = gray.shape
    contrast = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), max(2, 7 * max(h, w) / 1280)) - gray
    mask = np.zeros(gray.shape, np.uint8)
    for threshold in (10, 16, 24):
        _, _, stats, _ = cv2.connectedComponentsWithStats((contrast > threshold).astype(np.uint8))
        boxes = []
        for x, y, cw, ch, area in stats[1:]:
            if not (max(5, h * 0.004) < ch < h * 0.035 and 2 < cw < w * 0.08 and (cw < ch * 3.5) and (area > 12)):
                continue
            boxes.append((int(x), int(y), int(cw), int(ch)))
        boxes.sort()
        parent = list(range(len(boxes)))

        def root(i):
            while parent[i] != i:
                i = parent[i]
            return i
        for i, (x, y, cw, ch) in enumerate(boxes):
            for j in range(i + 1, len(boxes)):
                xx, yy, ww, hh = boxes[j]
                if xx > x + cw + ch * 2:
                    break
                if xx < x + cw - 2 or max(ch, hh) > min(ch, hh) * 1.8:
                    continue
                if abs(y + ch / 2 - (yy + hh / 2)) > max(ch, hh) * 0.45:
                    continue
                parent[root(j)] = root(i)
        groups = {}
        for i, b in enumerate(boxes):
            groups.setdefault(root(i), []).append(b)
        for group in groups.values():
            if len(group) < 4:
                continue
            for x, y, cw, ch in group:
                mask[y:y + ch, x:x + cw] = 1
    return cv2.dilate(mask, np.ones((3, 3), np.uint8))

def measure_sections(gray, alpha=None, *, broad_profile=False, word_mask=None,
                     crossing_mask=None, word_safe_ridge=False):
    """Return measured ink fragments, stopping at words and unsupported stock."""
    h, w = gray.shape
    scale = max(h, w) / 1280
    words = word_ink_mask(gray) if word_mask is None else np.asarray(word_mask)
    if words.shape != gray.shape:
        raise ValueError('Word mask must use the measurement coordinates')
    crossings = np.zeros(gray.shape, bool) if crossing_mask is None else np.asarray(crossing_mask, bool)
    if crossings.shape != gray.shape:
        raise ValueError('Crossing mask must use the measurement coordinates')
    valid = np.ones(gray.shape, bool) if alpha is None else alpha > 250
    contrast = np.maximum(0, cv2.GaussianBlur(gray.astype(np.float32), (0, 0), max(2, (12 if broad_profile else 6) * scale)) - gray)
    if broad_profile:
        contrast = cv2.GaussianBlur(contrast, (3, 1), 0)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 41, 3)
    seeds = cv2.HoughLinesP(binary, 1, np.pi / 720, threshold=max(16, int(w * 0.015)), minLineLength=max(25, w * 0.04), maxLineGap=3)
    if seeds is None:
        return []
    found = []
    for ax, ay, bx, by in sorted(seeds.reshape(-1, 4), key=lambda s: -abs(s[2] - s[0])):
        if bx < ax:
            ax, ay, bx, by = (bx, by, ax, ay)
        if bx - ax < max(25, w * 0.04) or abs(by - ay) > abs(bx - ax) * 0.45:
            continue
        slope = (by - ay) / (bx - ax)
        cx = (ax + bx) // 2
        cy = ay + slope * (cx - ax)
        if any((g[0, 0] <= cx <= g[-1, 0] and abs(np.interp(cx, g[:, 0], g[:, 1]) - cy) < 4 * scale for g in found)):
            continue
        widths = []
        for sx in np.linspace(ax, bx, 31).astype(int):
            sy = int(round(ay + slope * (sx - ax)))
            lo = max(0, sy - 8)
            hi = min(h, sy + 9)
            profile = contrast[lo:hi, sx]
            hits = np.flatnonzero(profile > max(3, float(profile.max()) * 0.5))
            if len(hits):
                groups = np.split(hits, np.where(np.diff(hits) > 1)[0] + 1)
                group = min(groups, key=lambda g: abs(float(g.mean() + lo) - sy))
                widths.append(len(group))
        if not widths:
            continue
        stock_offset = max(3, round(np.median(widths) / 2) + 2)
        halves = []
        for direction in (-1, 1):
            points = []
            y = cy
            velocity = slope
            missing = 0
            for x in range(cx, 2 if direction < 0 else w - 2, direction):
                pred = y + direction * velocity
                radius = 4 if broad_profile else 3
                lo = max(3, int(round(pred)) - radius)
                hi = min(h - 3, int(round(pred)) + radius + 1)
                if lo >= hi:
                    break
                ys = np.arange(lo, hi)
                band = max(2, round(4 * scale))
                if not word_safe_ridge and words[max(0, round(pred) - band):min(h, round(pred) + band + 1), x].any():
                    break
                score = np.minimum(contrast[ys, x], 8) - 0.7 * np.abs(ys - pred)
                if word_safe_ridge:
                    blocked=words[ys,x]>0
                    if blocked.all():break
                    score[blocked]=-np.inf
                yy = int(ys[np.argmax(score)])
                if word_safe_ridge and words[yy-1:yy+2,x].any():break
                if not valid[max(0, yy - 3):min(h, yy + 4), x].all():
                    break
                if contrast[yy, x] < 3:
                    missing += 1
                    if missing > max(3, round((6 if broad_profile else 4) * scale)):
                        break
                    y = pred
                    continue
                if abs(yy - pred) > (3.5 if broad_profile else 2.5):
                    break
                check = np.array([yy - stock_offset - 1, yy - stock_offset, yy + stock_offset, yy + stock_offset + 1])
                check = np.clip(check, 0, h - 1)
                side_ink=contrast[check,x]>max(8,float(contrast[yy,x])*.6)
                if word_safe_ridge:
                    # Known adjacent word ink cannot masquerade as a thick
                    # stroke. Unknown side ink still requires a measured cross.
                    side_ink &= words[check,x]==0
                if (not crossings[yy, x] and contrast[yy, x] > 8 and side_ink.any()):
                    break
                measured = float(yy)
                values = contrast[max(0, yy - 2):min(h, yy + 3), x]
                hits = np.flatnonzero(values > max(3, float(values.max()) * 0.55)) + max(0, yy - 2)
                if len(hits):
                    if word_safe_ridge and words[hits,x].any():break
                    measured = float(hits.mean())
                velocity = 0.8 * velocity + 0.2 * ((measured - y) * direction)
                velocity = float(np.clip(velocity, -0.45, 0.45))
                y = measured
                points.append([x, y])
                missing = 0
            halves.append(np.array(points, float).reshape(-1, 2))
        if not len(halves[0]) or not len(halves[1]):
            continue
        g = np.vstack([halves[0][::-1], halves[1][1:]])
        if len(g) < w * 0.06 or np.ptp(g[:, 0]) < w * 0.045:
            continue
        x = g[:, 0]
        curve = np.array([np.median(g[np.abs(x - xx) < max(3, round(4 * scale)), 1]) for xx in x])
        error = np.abs(curve - g[:, 1])
        strength = []
        for xx, yy in np.rint(g).astype(int):
            strength.append(contrast[yy, xx])
        if np.percentile(error, 90) > 1.5 or np.mean(np.array(strength) > 3) < 0.85:
            continue
        g = np.column_stack([x, curve])
        found.append(g)
    chosen = []
    for g in sorted(found, key=lambda g: -np.ptp(g[:, 0])):
        duplicate = False
        for q in chosen:
            lo = max(g[0, 0], q[0, 0])
            hi = min(g[-1, 0], q[-1, 0])
            if hi - lo < min(np.ptp(g[:, 0]), np.ptp(q[:, 0])) * 0.6:
                continue
            xs = np.linspace(lo, hi, 20)
            if np.max(abs(np.interp(xs, g[:, 0], g[:, 1]) - np.interp(xs, q[:, 0], q[:, 1]))) < 3 * scale:
                duplicate = True
                break
        if not duplicate:
            chosen.append(g)
    return [g / [w - 1, h - 1] for g in sorted(chosen, key=lambda g: g[:, 1].mean())]
