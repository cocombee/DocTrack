"""Transfer only barcode and final-row displacements onto the accepted mesh."""
from pathlib import Path
import json
import sys

import cv2
import numpy as np
import receipt_content_mesh as reference


def sample(field, points):
    rows, columns = field.shape[:2]
    xy = np.clip(points * [columns - 1, rows - 1], 0,
                 [columns - 1 - 1e-9, rows - 1 - 1e-9])
    lo = xy.astype(int)
    fraction = xy - lo
    x, y = lo[:, 0], lo[:, 1]
    a, b = fraction[:, 0, None], fraction[:, 1, None]
    return ((1-a)*(1-b)*field[y, x] + a*(1-b)*field[y, x+1] +
            (1-a)*b*field[y+1, x] + a*b*field[y+1, x+1])


def inverse(grid, points):
    rows, columns = grid.shape[:2]
    estimate = points.copy()
    gx = np.gradient(grid, 1/(columns-1), axis=1)
    gy = np.gradient(grid, 1/(rows-1), axis=0)
    for _ in range(10):
        delta = sample(grid, estimate) - points
        a, b = sample(gx, estimate), sample(gy, estimate)
        determinant = a[:, 0]*b[:, 1] - a[:, 1]*b[:, 0]
        estimate -= np.column_stack([
            (b[:, 1]*delta[:, 0] - b[:, 0]*delta[:, 1])/determinant,
            (-a[:, 1]*delta[:, 0] + a[:, 0]*delta[:, 1])/determinant])
        estimate = np.clip(estimate, 0, 1)
    return estimate


def valid(grid):
    rows, columns = grid.shape[:2]
    u, v = np.meshgrid(np.linspace(0, 1, columns), np.linspace(0, 1, rows))
    identity = np.stack([u, v], -1)
    dx, dy = np.diff(grid, axis=1), np.diff(grid, axis=0)
    areas = [(a[:, :, 0]*b[:, :, 1] - a[:, :, 1]*b[:, :, 0]) *
             (rows-1)*(columns-1)
             for a in (dx[:-1], dx[1:]) for b in (dy[:, :-1], dy[:, 1:])]
    return bool(np.isfinite(grid).all() and
                np.max(np.abs(grid-identity)) < .14 and
                min(area.min() for area in areas) > .2 and
                grid.min() >= -.025 and grid.max() <= 1.025)


def window(t, start, inner_start, inner_end, end):
    left = np.clip((t-start)/max(inner_start-start, 1e-9), 0, 1)
    right = np.clip((end-t)/max(end-inner_end, 1e-9), 0, 1)
    return (left*left*(3-2*left)) * (right*right*(3-2*right))


def solve(path, working=None):
    original = reference.base.solve(path, working)
    try:
        return refine(path, original)
    except (OSError, ValueError, ArithmeticError, AttributeError,
            IndexError, KeyError, TypeError, cv2.error) as error:
        original['local_combination'] = {
            'accepted': False, 'error': type(error).__name__}
        return original


def refine(path, original):
    best = reference.refine(path, dict(original))
    if best['method'] != 'Printed receipt content refinement':
        return original
    rows, columns = original['rows'], original['columns']
    old = np.array(original['grid']).reshape(rows, columns, 2)
    improved = np.array(best['grid']).reshape(best['rows'], best['columns'], 2)
    rgba = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    h, w = rgba.shape[:2]
    mask = rgba[:, :, 3] if rgba.shape[2] == 4 else np.full((h, w), 255, np.uint8)
    alpha = mask[:, :, None]/255
    gray = cv2.cvtColor((rgba[:, :, :3]*alpha + 255*(1-alpha)).astype(np.uint8), cv2.COLOR_BGR2GRAY)
    bars = reference.barcode_rulers(gray)
    envelopes = reference.measured_envelopes(gray, bars)
    features = reference.base.extract_text_features(gray, mask=mask)
    texts = reference.text_rulers(gray, features)
    rules = reference.extract_rules(gray)
    footer = max(texts, key=lambda p: np.mean(p[:, 1]))
    if np.mean(footer[:, 1]) < .8 or np.ptp(footer[:, 0]) < .4:
        return original
    # Refine only where the observed barcode and final text row reside. Four
    # subdivisions of each original cell preserve its bilinear map exactly.
    dense_rows, dense_columns = (rows-1)*4+1, (columns-1)*4+1
    u, v = np.meshgrid(np.linspace(0, 1, dense_columns),
                       np.linspace(0, 1, dense_rows))
    points = np.column_stack([u.ravel(), v.ravel()])
    baseline = sample(old, points).reshape(dense_rows, dense_columns, 2)
    better = sample(improved, points).reshape(dense_rows, dense_columns, 2)
    observed = inverse(old, np.vstack(bars))
    x0, y0 = observed.min(axis=0)
    x1, y1 = observed.max(axis=0)
    height = y1-y0
    barcode_weight = (window(u, x0-.025, x0-.006, x1+.006, x1+.025) *
                      window(v, y0-height*.25, y0-height*.1,
                             y1+height*.1, y1+height*.25))
    mapped_footer = inverse(old, footer)
    fx0, fy0 = mapped_footer.min(axis=0)
    fx1, fy1 = mapped_footer.max(axis=0)
    footer_weight = (window(u, fx0-.04, fx0-.015, fx1+.015, fx1+.04) *
                     window(v, fy0-.025, fy0-.012, fy1+.012,
                            min(1.015, fy1+.04)))
    delta = better-baseline
    combined = baseline + delta*barcode_weight[:, :, None]
    # Solve the last observed baseline directly at a constant output height.
    # Keep its horizontal map and the rest of the accepted mesh intact; a
    # whole-page refit's competing guides need not dictate the footer shape.
    target_y = np.mean(mapped_footer[:, 1])
    at_target = sample(old, np.column_stack([u[0], np.full(dense_columns, target_y)]))
    coefficients = np.polyfit(footer[:, 0], footer[:, 1], 2)
    footer_delta = np.polyval(coefficients, at_target[:, 0]) - at_target[:, 1]
    combined[:, :, 1] += footer_delta[None, :] * footer_weight

    def errors(g, guides, axis):
        return np.array([np.std(inverse(g, p)[:, axis]) for p in guides])
    protected = rules + [p for p in texts if p is not footer]
    before = errors(old, protected, 1)
    after = errors(combined, protected, 1)
    protected_points = np.vstack(protected)
    protected_shift = np.max(np.abs(
        (inverse(combined, protected_points) - inverse(old, protected_points)) * [w, h]))
    bar_errors = errors(combined, bars, 0)*w
    envelope_errors = errors(combined, envelopes, 1)*h
    footer_before = errors(old, [footer], 1)[0]*h
    footer_after = errors(combined, [footer], 1)[0]*h
    accepted = bool(valid(combined) and protected_shift <= .1 and
                    np.all(after <= before + .1/h) and
                    np.all(bar_errors <= errors(improved, bars, 0)*w + .15) and
                    np.all(envelope_errors <= errors(improved, envelopes, 1)*h + .15) and
                    footer_after < footer_before*.75)
    original['local_combination'] = {
        'accepted': accepted,
        'protected_max_increase_px': float(np.max((after-before)*h)),
        'protected_max_displacement_px': float(protected_shift),
        'barcode_max_px': float(bar_errors.max()),
        'barcode_envelope_max_px': float(envelope_errors.max()),
        'footer_before_px': float(footer_before),
        'footer_after_px': float(footer_after),
        'barcode_region': [float(x0-.025), float(y0-height*.25),
                           float(x1+.025), float(y1+height*.25)],
        'footer_region': [float(fx0-.04), float(fy0-.025),
                          float(fx1+.04), float(min(1.015, fy1+.04))],
    }
    if accepted:
        original.update(rows=dense_rows, columns=dense_columns,
                        grid=combined.reshape(-1, 2).tolist(),
                        method='Local barcode and final-row correction')
    return original


if __name__ == '__main__':
    result = solve(sys.argv[1], sys.argv[3] if len(sys.argv) > 3 else None)
    Path(sys.argv[2]).write_text(json.dumps(result, allow_nan=False))
