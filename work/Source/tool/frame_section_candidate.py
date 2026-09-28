"""Optional complete faint-curve refinement after the accepted row step.

This research path is not the application default. A wider normal profile may
extend a previously measured curve, but it must agree with that measurement and
show a coherent bend. The complete extended trace is validated, not a subset.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

import section_row_candidate as rows
from printed_rule_sections import measure_sections
from local_mesh_storage import publish


def supported_extensions(narrow, broad, width, height):
    candidates = []
    for guide in broad:
        if np.ptp(guide[:, 0]) < .2:
            continue
        agreement = False
        for measured in narrow:
            lo = max(guide[0, 0], measured[0, 0])
            hi = min(guide[-1, 0], measured[-1, 0])
            if hi - lo < .12:
                continue
            x = np.linspace(lo, hi, 65)
            error = abs(np.interp(x, guide[:, 0], guide[:, 1]) -
                        np.interp(x, measured[:, 0], measured[:, 1])) * height
            if np.percentile(error, 95) < 2 * max(width, height) / 1280:
                agreement = True
                break
        if not agreement:
            continue
        x, y = guide.T
        straight = np.polyval(np.polyfit(x, y, 1), x)
        bend = float(np.std(y - straight) * height)
        smooth = np.polyval(np.polyfit(x, y, 4), x)
        explained = 1 - np.mean((smooth - y) ** 2) / max(np.mean((y - straight) ** 2), 1e-16)
        if bend > 2 * max(width, height) / 1280 and explained > .65:
            candidates.append((bend, guide))
    return [guide for _, guide in sorted(candidates, key=lambda item: -item[0])]


def refine(path, base):
    gray, mask, _, _ = rows.current_measurement(path, base)
    height, width = gray.shape
    narrow = measure_sections(gray, mask)
    broad = measure_sections(gray, mask, broad_profile=True)
    for guide in supported_extensions(narrow, broad, width, height):
        attempts = []
        result = rows.refine(path, base, guides=[guide], include_wide_ink=True,
                             minimum_resolution=(2049, 1025), audit_log=attempts)
        if result is not base:
            result['method'] = 'Experimental complete faint-curve correction'
            result['frame_section_audit'] = attempts
            return result
    return base


def solve(path, working=None):
    base = rows.solve(path, working)
    try:
        return refine(path, base)
    except (OSError, ValueError, ArithmeticError, AttributeError, IndexError,
            KeyError, TypeError, cv2.error):
        return base


def solve_staged(path, output, working=None):
    base = rows.accepted.solve(path, working)
    publish(base, output)
    for step in (rows.refine, refine):
        try:
            result = step(path, base)
            if result is not base:
                publish(result, output)
            base = result
        except (OSError, ValueError, ArithmeticError, AttributeError, IndexError,
                KeyError, TypeError, cv2.error):
            return base
    return base


if __name__ == '__main__':
    solve_staged(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
