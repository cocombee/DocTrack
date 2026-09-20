"""Offline release gate. Reference labels are never inputs to the detector.

Compare complete boundaries at uniform arc-length spacing in both directions.
An exit code of zero means no measured regression; it is NOT visual approval.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import cv2


def xy(p):
    return np.asarray([p['x'], p['y']], dtype=float)


def reference_segments(vertices):
    result = []
    for i, a in enumerate(vertices):
        b = vertices[(i + 1) % len(vertices)]
        s = {'start': a, 'end': b}
        if a.get('smooth') or b.get('smooth'):
            s['control1'] = dict(zip(('x', 'y'), a['hOut'])) if a.get('smooth') else a
            s['control2'] = dict(zip(('x', 'y'), b['hIn'])) if b.get('smooth') else b
        result.append(s)
    return result


def sample_boundary(segments, width, height, spacing=1.0):
    """Equal weighting per unit perimeter, including the middle of long lines."""
    scale = np.asarray([width, height])
    pieces = []
    for s in segments:
        controls = np.asarray([xy(s[k]) for k in ('start', 'end')]) * scale
        a, d = controls
        if 'control1' in s:
            b, c = xy(s['control1']) * scale, xy(s['control2']) * scale
            bound = np.linalg.norm(b-a) + np.linalg.norm(c-b) + np.linalg.norm(d-c)
            t = np.linspace(0, 1, max(2, int(np.ceil(bound / .25)) + 1))[:, None]
            points = (1-t)**3*a + 3*(1-t)**2*t*b + 3*(1-t)*t*t*c + t**3*d
        else:
            points = np.asarray([a, d])
        pieces.extend(points[:-1])
    points = np.asarray(pieces + [pieces[0]])
    distances = np.r_[0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    keep = np.r_[True, np.diff(distances) > 1e-10]
    distances, points = distances[keep], points[keep]
    positions = np.linspace(0, distances[-1], max(4, int(np.ceil(distances[-1]/spacing))), endpoint=False)
    return np.column_stack([np.interp(positions, distances, points[:, axis]) for axis in (0, 1)])


def errors(candidate, reference):
    def nearest(source, target):
        matches = cv2.BFMatcher(cv2.NORM_L2).match(
            np.asarray(source, dtype=np.float32), np.asarray(target, dtype=np.float32))
        return np.asarray([match.distance for match in matches])
    forward = nearest(candidate, reference)
    reverse = nearest(reference, candidate)
    return {
        'p95_px': float(max(np.quantile(forward, .95), np.quantile(reverse, .95))),
        'max_px': float(max(forward.max(), reverse.max())),
        'mean_px': float((forward.mean() + reverse.mean()) / 2),
    }


def topology_issues(segments, width, height):
    issues = []
    for i, s in enumerate(segments):
        next_s = segments[(i + 1) % len(segments)]
        if np.linalg.norm((xy(s['end']) - xy(next_s['start'])) * [width, height]) > 1e-4:
            issues.append('disconnected_segments')
        if s.get('isImageClip') and ('control1' in s or 'control2' in s):
            issues.append('curved_crop_closure')
    points = sample_boundary(segments, width, height, spacing=3)
    if not np.isfinite(points).all():
        issues.append('nonfinite_boundary')
        return sorted(set(issues))
    if (points < -1e-4).any() or (points > np.asarray([width, height]) + 1e-4).any():
        issues.append('outside_raster')
    # Proper crossings (ignore shared endpoints and near-collinear contacts).
    starts, ends = points, np.roll(points, -1, axis=0)
    def cross(a, b):
        return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]
    for i in range(len(points)-2):
        js = np.arange(i+2, len(points))
        if i == 0:
            js = js[js != len(points)-1]
        a, b, c, d = starts[i], ends[i], starts[js], ends[js]
        if ((cross(b-a, c-a)*cross(b-a, d-a) < -1e-8) &
                (cross(d-c, a-c)*cross(d-c, b-c) < -1e-8)).any():
            issues.append('self_intersection')
            break
    return sorted(set(issues))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--tolerance-px', type=float, default=1.0,
                        help='Raster/reference sampling allowance, not an accuracy target.')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    manifest = json.loads((root/'Data/manifest.json').read_text())
    rows, blocked = [], []
    for number, entry in enumerate(manifest, 1):
        path = root/'Data'/entry['label']
        label = json.loads(path.read_text())
        candidate = json.loads((args.candidate/(entry['name']+'.json')).read_text())
        baseline = json.loads((args.baseline/(entry['name']+'.json')).read_text())
        width, height = candidate['width'], candidate['height']
        assert (width, height) == (baseline['width'], baseline['height'])
        ref = sample_boundary(reference_segments(label['fullPolygon']), width, height)
        scores = []
        for geometry in (baseline, candidate):
            scores.append(errors(sample_boundary(geometry['segments'], width, height), ref))
        issues = topology_issues(candidate['segments'], width, height)
        regressed = [key for key in ('p95_px', 'max_px', 'mean_px')
                     if scores[1][key] > scores[0][key] + args.tolerance_px]
        if regressed or issues:
            blocked.append(number)
        rows.append({'index': number, 'name': entry['name'], 'baseline': scores[0],
                     'candidate': scores[1], 'regressed_metrics': regressed,
                     'geometry_issues': issues,
                     'reference_sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    report = {'release_gate': 'BLOCKED' if blocked else 'NO_MEASURED_REGRESSIONS',
              'visual_approval': False, 'images_checked': len(rows),
              'blocked_images': blocked, 'rows': rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != 'rows'}))
    return 1 if blocked else 0


if __name__ == '__main__':
    raise SystemExit(main())
