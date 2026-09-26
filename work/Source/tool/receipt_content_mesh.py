"""Evidence-gated receipt refinement; original solver remains the fallback.

Uses the existing MIT Text Line Dewarp checkout. No new models or assets.
"""
from pathlib import Path
import json
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent / 'mesh_frame'))
import dewarp_mesh as base
import cv2
import numpy as np
import dewarping.textline as textline
from mesh_ruler import text_rulers, ruler_mesh
from printed_line_mesh import extract_rules
from barcode_ruler import barcode_rulers


def supplemental_text(gray, existing):
    # Extract dark text independently. Combining both polarities can merge
    # letters and their bright gaps into a non-text block, hiding whole rows.
    groups = textline._link_ccs_into_lines(
        textline._extract_ccs(textline.binarize(gray)))
    features = SimpleNamespace(lines=[
        SimpleNamespace(centers=np.array([c.center for c in group]))
        for group in groups if textline._is_high_confidence_line(group)])
    added = []
    for guide in text_rulers(gray, features):
        if np.ptp(guide[:, 0]) < .2:
            continue
        duplicate = False
        for previous in existing + added:
            lo = max(guide[0, 0], previous[0, 0])
            hi = min(guide[-1, 0], previous[-1, 0])
            if hi - lo < min(np.ptp(guide[:, 0]), np.ptp(previous[:, 0])) * .5:
                continue
            x = np.linspace(lo, hi, 33)
            if np.median(np.abs(np.interp(x, guide[:, 0], guide[:, 1]) -
                                np.interp(x, previous[:, 0], previous[:, 1]))) < .006:
                duplicate = True
                break
        if not duplicate:
            added.append(guide)
    return added


def measured_envelopes(gray, bars):
    if len(bars) < 12:
        return []
    x = np.array([p[:, 0].mean() for p in bars])
    lo, hi = np.quantile(x, [.05, .95])
    keep = (x >= lo) & (x <= hi)
    if keep.sum() < 8 or hi - lo < .15:
        return []
    guides = []
    for index in (0, -1):
        y = np.array([p[index, 1] for p in bars])
        design = np.column_stack([x*x, x, np.ones(len(x))])
        weights = keep.astype(float)
        for _ in range(5):
            coefficients = np.linalg.lstsq(
                design * weights[:, None], y * weights, rcond=None)[0]
            residual = y - design @ coefficients
            weights = keep * np.minimum(1, 1.5 / (np.abs(residual) * gray.shape[0] + 1e-6))
        if np.mean(np.abs(residual[keep]) * gray.shape[0] < 2.5) < .85:
            return []
        xx = np.linspace(x[keep].min(), x[keep].max(), 33)
        guides.append(np.column_stack([xx, np.polyval(coefficients, xx)]))
    return guides


def solve(path, working=None):
    original = base.solve(path, working)
    try:
        return refine(path, original)
    except (OSError, ValueError, ArithmeticError, AttributeError, cv2.error) as error:
        # Optional refinement must never replace a working correction with
        # the adapter's identity fallback when its feature pass fails.
        original['content_refinement'] = {'accepted': False, 'error': type(error).__name__}
        return original


def refine(path, original):
    previous = np.array(original['grid']).reshape(base.N, base.N, 2)
    if original['method'] != 'Shared mesh rulers' or any(
            original.get(key, 0) for key in
            ('frame_rulers', 'vertical_rules', 'vertical_text_rulers', 'circular_landmarks')):
        return original
    rgba = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    h, w = rgba.shape[:2]
    mask = rgba[:, :, 3] if rgba.shape[2] == 4 else np.full((h, w), 255, np.uint8)
    alpha = mask[:, :, None] / 255
    gray = cv2.cvtColor((rgba[:, :, :3]*alpha + 255*(1-alpha)).astype(np.uint8), cv2.COLOR_BGR2GRAY)
    features = base.extract_text_features(gray, mask=mask)
    texts = text_rulers(gray, features)
    rules = extract_rules(gray)
    bars = barcode_rulers(gray)
    envelopes = measured_envelopes(gray, bars)
    added = supplemental_text(gray, texts + rules)
    # A repeated barcode and independent printed rules anchor this refinement.
    # Other page layouts keep the original solver without any additional warp.
    if len(bars) < 12 or len(rules) < 3 or len(added) < 2 or len(envelopes) != 2:
        return original
    horizontal = [(p, 8.) for p in rules + envelopes] + [(p, 6.) for p in texts + added]
    grid = ruler_mesh(horizontal, [(p, 8.) for p in bars], prior=previous, prior_weight=.045)
    def errors(g, guides, axis):
        return np.array([np.std(base.inverse_points(g, p)[:, axis]) for p in guides])
    gates = base.valid(grid, edge_extension=True)
    summary = {}
    for label, guides, axis, size in (
            ('printed', rules, 1, h), ('text', texts, 1, h),
            ('barcode', bars, 0, w), ('envelopes', envelopes, 1, h),
            ('supplemental', added, 1, h)):
        before, after = errors(previous, guides, axis), errors(grid, guides, axis)
        summary[label] = {'before_max_px': float(before.max(initial=0)*size),
                          'after_max_px': float(after.max(initial=0)*size)}
        gates = gates and bool(np.all(after <= before + .5/size))
    gains = errors(previous, added + envelopes, 1).mean()
    remaining = errors(grid, added + envelopes, 1).mean()
    accepted = bool(gates and gains > .5/h and remaining < gains*.5)
    original['content_refinement'] = {'accepted': accepted, 'guides': len(added), 'errors': summary,
                                    'before_mean_px': float(gains*h), 'after_mean_px': float(remaining*h)}
    if accepted:
        original['grid'] = grid.reshape(-1, 2).tolist()
        original['method'] = 'Printed receipt content refinement'
    return original


if __name__ == '__main__':
    result = solve(sys.argv[1], sys.argv[3] if len(sys.argv) > 3 else None)
    Path(sys.argv[2]).write_text(json.dumps(result, allow_nan=False))
