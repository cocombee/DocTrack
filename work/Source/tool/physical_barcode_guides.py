"""Corroborated physical barcode bars near a repeated vertical-rule cluster.

The rule detector locates the cluster; connected printed components recover
bars that a Hough line misses. Every guide and owner consists of observed ink.
"""
import cv2
import numpy as np


def barcode_components(gray, safe, vertical_rules):
    h, w = gray.shape
    basis = np.array([w - 1., h - 1.])
    seeds = []
    for rule in vertical_rules:
        q = np.asarray(rule, float)
        if q.ndim != 2 or q.shape[1] != 2 or len(q) < 24:
            continue
        p = q * basis
        y0, y1 = float(p[:, 1].min()), float(p[:, 1].max())
        if y1 - y0 >= 35 and np.ptp(p[:, 0]) <= max(10, (y1-y0)*.15):
            seeds.append((float(np.median(p[:, 0])), y0, y1))
    if len(seeds) < 8:
        return []
    seeds = np.asarray(seeds)
    length = np.median(seeds[:, 2] - seeds[:, 1])
    center = np.median((seeds[:, 1] + seeds[:, 2])/2)
    chosen = seeds[(abs((seeds[:, 1] + seeds[:, 2])/2-center) < length*.16) &
                   (abs((seeds[:, 2]-seeds[:, 1])-length) < length*.25)]
    if len(chosen) < 8:
        return []
    xs = np.sort(chosen[:, 0])
    gaps = np.diff(xs)
    if (np.median(gaps) >= length*.4 or
            xs[-1]-xs[0] < length*1.8):
        return []
    pad_x = max(10., np.median(gaps)*2.5)
    pad_y = max(5., length*.1)
    x0 = max(0, int(np.floor(xs[0]-pad_x)))
    x1 = min(w, int(np.ceil(xs[-1]+pad_x+1)))
    y0 = max(0, int(np.floor(chosen[:, 1].min()-pad_y)))
    y1 = min(h, int(np.ceil(chosen[:, 2].max()+pad_y+1)))
    if x1-x0 < length*2 or y1-y0 < length*.7:
        return []
    contrast = cv2.GaussianBlur(gray.astype(np.float32), (0, 0),
                                max(2, 7*max(w,h)/1280))-gray

    def components(threshold):
        mask = ((contrast[y0:y1, x0:x1] > threshold) &
                (safe[y0:y1, x0:x1] > 250)).astype(np.uint8)
        _, labels, stats, centers = cv2.connectedComponentsWithStats(mask)
        found = []
        for index, (x, y, cw, ch, area) in enumerate(stats[1:], 1):
            if not (length*.68 <= ch <= length*1.27 and
                    2 <= cw <= min(20, length*.23) and
                    area >= max(30, ch*cw*.30)):
                continue
            px = float(centers[index, 0]+x0)
            py = float(centers[index, 1]+y0)
            found.append((px, py, x, y, cw, ch, index, labels))
        return found

    lower = components(10)
    upper = components(16)
    if len(lower) < 12 or len(upper) < 12:
        return []
    matched = []
    used = set()
    for item in lower:
        px, py, x, y, cw, ch, index, labels = item
        choices = [(abs(px-other[0]), j) for j, other in enumerate(upper)
                   if j not in used and abs(px-other[0]) <= max(2.5,cw*.4)
                   and abs(py-other[1]) <= length*.12]
        if not choices:
            continue
        _, j = min(choices)
        used.add(j)
        roi = labels[y:y+ch, x:x+cw] == index
        yy, xx = np.where(roi)
        yy, xx = yy+y+y0, xx+x+x0
        ink = np.c_[xx, yy].astype(float)/basis
        unique_y = np.unique(yy)
        guide = np.array([[np.mean(xx[yy==row]), row] for row in unique_y],float)/basis
        matched.append(dict(points=guide, ink=ink,
                            bounds_px=[int(x+x0),int(y+y0),int(cw),int(ch)],
                            center_px=[px,py]))
    if len(matched) < 12:
        return []
    matched.sort(key=lambda item:item['center_px'][0])
    widths = np.array([q['bounds_px'][2] for q in matched])
    positions = np.array([q['center_px'][0] for q in matched])
    if np.median(np.diff(positions)) >= length*.4 or np.ptp(widths) < 2:
        return []
    return matched
