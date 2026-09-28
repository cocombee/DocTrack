"""Measure sparse text baselines without treating OCR boxes as printed lines.

Native quadrilaterals associate words only. Baseline geometry comes from glyph
ink in the paired raster. No interpolated points or invented gap ink are used;
the caller's original material arrays and registrations remain untouched.
"""
import numpy as np

from native_word_evidence import ink_observations, measurement_words
from rigid_word_row import find_rows, glyph_baseline


def capture_text_baselines(gray, safe, words):
    gray, safe = np.asarray(gray), np.asarray(safe)
    if (gray.ndim != 2 or min(gray.shape) < 16 or safe.shape != gray.shape
            or not np.isfinite(gray).all() or not np.isfinite(safe).all()):
        raise ValueError('Text baselines require paired finite measurement rasters')
    height, width = gray.shape
    basis = np.array([width - 1, height - 1])
    observations = ink_observations(gray, words, safe, localization_halo=3)
    rows = find_rows(measurement_words(words, width, height), basis)
    traces = []
    for row in rows:
        selected = []
        for quad in row:
            matches = [i for i, owner in enumerate(observations)
                       if np.allclose(owner['quad'], quad, rtol=0, atol=1e-12)]
            if len(matches) != 1:
                break
            selected.extend(matches)
        if len(selected) != len(row) or len(set(selected)) != len(row):
            continue
        # Only this proposal measurement is combined. Every original owner
        # still exists separately in the material registry and all audits.
        body = np.unique(np.vstack([observations[i]['ink'] for i in selected]), axis=0)
        evidence = glyph_baseline(gray, body, pixel_size=basis)
        if evidence is None:
            continue
        points = np.asarray(evidence['points']) / basis
        if (not np.isfinite(points).all() or len(points) < 6
                or (points < 0).any() or (points > 1).any()):
            continue
        traces.append(dict(kind='text_baseline', points=points.tolist(),
            glyph_baseline_support=dict(
                basis='Native row with measured raster glyph bottoms',
                native_word_count=len(selected), glyph_count=len(points),
                native_observation_indices=selected,
                scatter_px=evidence['scatter_px']),
            measured_angle_radians=evidence['angle']))
    return traces
