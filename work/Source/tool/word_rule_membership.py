"""Separate supported divider fragments from disjoint word ink cores."""
import cv2
import numpy as np


def separate_rule_ink(groups, rule, width, height):
    """Keep all word components except small, separated ink on this rule.

    OCR envelopes can include nearby divider dashes. A trace alone cannot
    remove a letter: the component must also be short compared with the word
    and separated from its tall glyph components by a visible stock gap.
    Ambiguous contact keeps the original ink protection.
    """
    rule = np.asarray(rule)
    rule = rule[np.argsort(rule[:, 0])]
    output, removed = [], []
    scale = max(width, height) / 1280
    for group in groups:
        group = np.asarray(group)
        if not len(group):
            output.append(np.empty((0,2)))
            removed.append(np.empty((0,2)))
            continue
        if (group.ndim != 2 or group.shape[1] != 2 or
                not np.isfinite(group).all() or (group < 0).any() or (group >= 1).any()):
            raise ValueError('Word ink points must be finite image coordinates')
        pixels = np.floor(group * [width, height]).astype(int)
        lo, hi = pixels.min(0), pixels.max(0)
        mask = np.zeros((hi[1]-lo[1]+1, hi[0]-lo[0]+1), np.uint8)
        mask[pixels[:, 1]-lo[1], pixels[:, 0]-lo[0]] = 1
        _, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        glyph_height = max(stats[1:, cv2.CC_STAT_HEIGHT], default=0)
        tall = np.flatnonzero(stats[:, cv2.CC_STAT_HEIGHT] >= .5*glyph_height)
        tall = tall[tall != 0]
        keep = np.ones(len(group), bool)
        membership = labels[pixels[:, 1]-lo[1], pixels[:, 0]-lo[0]]
        for index, (x,y,cw,ch,area) in enumerate(stats[1:], 1):
            if ch >= .3*glyph_height or not len(tall):
                continue
            subset = membership == index
            points = group[subset]
            if (points[:,0].min() < rule[0,0] or points[:,0].max() > rule[-1,0] or
                    np.max(abs(points[:,1] - np.interp(points[:,0], rule[:,0],rule[:,1]))) * height > 3*scale):
                continue
            # Require empty stock in the normal direction between this short
            # component and every tall letter in the same word envelope.
            gaps = []
            for other in tall:
                _, gy, _, gh, _ = stats[other]
                gaps.append(max(gy-(y+ch), y-(gy+gh), 0))
            if min(gaps) < 2*scale:
                continue
            keep[subset] = False
        output.append(group[keep])
        removed.append(group[~keep])
    return output, removed
