"""Guard one physical text row within an already protected page field.

The caller supplies measured rows, observed material owners and a bounded
window. This helper does not choose a document, a row, or sample coordinates.
It supports a complete row first and a corroborated base-character fallback.
"""
import cv2
import numpy as np

from composite_mesh_inverse import inverse_retained
from composite_mesh_sampling import MeshSampler
from material_pixel_footprints import paired_points
from physical_character_material import relative_guard_owners
from shared_material_mesh import shared_material_field


def _smooth(value):
    value = np.clip(value, 0, 1)
    return value * value * (3 - 2 * value)


def _character_models(gray, guide, sampler, source_basis, output_basis,
                      thresholds=(16, 20)):
    """Require two agreeing curves from actual principal character bodies."""
    contrast = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 7) - gray
    center_y = float(np.median(guide['points'][:, 1]) * source_basis[1])
    left, right = guide['domain_px']
    models = []
    for threshold in thresholds:
        _, labels, stats, _ = cv2.connectedComponentsWithStats(
            (contrast > threshold).astype(np.uint8))
        anchors = []
        for label, (px, py, width, height, area) in enumerate(stats[1:], 1):
            if (px < left - 52 or px > right + 24 or
                    py + height - 1 < center_y - 6.5 or
                    py > center_y + 13.5 or width < 3 or height < 7 or
                    height > 15 or width > 25 or area < 18):
                continue
            body = labels[py:py+height, px:px+width] == label
            bottoms = [np.flatnonzero(body[:, column])[-1]
                       for column in range(width) if body[:, column].any()]
            anchors.append((px + width/2, py + np.quantile(bottoms, .85)))
        if len(anchors) < 13:
            raise ValueError('Insufficient principal physical characters')
        points = inverse_retained(sampler,
            np.asarray(sorted(anchors)) / source_basis) * output_basis
        px, py = points.T
        origin, span = float(px.mean()), float(np.ptp(px))
        z = (px - origin) / span
        coefficients = np.polyfit(z, py, 2)
        for _ in range(5):
            keep = abs(py - np.polyval(coefficients, z)) < 2.3
            coefficients = np.polyfit(z[keep], py[keep], 2)
        if keep.sum() < 12:
            raise ValueError('Principal character curve lacks inliers')
        models.append((coefficients, origin, span, points))
    sample = np.linspace(max(q[3][:, 0].min() for q in models),
                         min(q[3][:, 0].max() for q in models), 65)
    fitted = [np.polyval(q[0], (sample-q[1])/q[2]) for q in models]
    if float(np.max(abs(fitted[0]-fitted[1]))) > 2:
        raise ValueError('Physical character profiles disagree')
    return models


def refine_row(parent, gray, guides, ledger, *, upper_row, lower_row,
               mode, scale, output_size, material_source_size,
               source_selection_bounds, x_window, y_window,
               size=1025, clip_top=None, clip_bottom=None):
    """Return a material-guarded coordinate layer and its local certificate.

    ``mode`` is ``upper`` (paired line spacing), ``lower`` (whole baseline),
    or ``lower_character``. All windows and owner bounds are caller evidence;
    none are inferred from a document name or OCR spelling.
    """
    if mode not in ('upper', 'lower', 'lower_character') or not 0 < scale <= 1:
        raise ValueError('Invalid typography correction request')
    if size < 33 or size % 2 != 1:
        raise ValueError('A regular odd lattice is required')
    source_basis = np.asarray(material_source_size, float) - 1
    output_basis = np.asarray(output_size, float) - 1
    sampler = MeshSampler(parent)
    upper = inverse_retained(sampler, guides[upper_row]['points']) * output_basis
    lower_source = np.vstack([np.asarray(p) for p in
        guides[lower_row]['accepted_glyph_anchors_px']]) / source_basis
    lower = inverse_retained(sampler, lower_source) * output_basis
    models = None
    if mode == 'upper':
        target = float(np.median(lower[:, 1]) - 1.6 *
            guides[upper_row]['median_glyph_height_px'] *
            output_basis[1] / source_basis[1])
    elif mode == 'lower':
        upper = inverse_retained(sampler, guides[lower_row]['points']) * output_basis
        target = float(np.median(lower[:, 1]))
    else:
        models = _character_models(gray, guides[lower_row], sampler,
                                   source_basis, output_basis)
        target = float(np.median(np.vstack([q[3] for q in models])[:, 1]))

    u, v = np.meshgrid(np.linspace(0, 1, size), np.linspace(0, 1, size))
    x, y = u * output_basis[0], v * output_basis[1]
    if models is None:
        curve = np.interp(x, upper[:, 0], upper[:, 1])
    else:
        curve = sum(np.polyval(q[0], (x-q[1])/q[2]) for q in models) / len(models)
        curve = np.clip(curve, target-6, target+6)
    shift = np.clip(target-curve, -8, 8) * scale
    x0, x1, x2, x3 = x_window
    y0, y1, y2, y3 = y_window
    weight = (_smooth((x-x0)/(x1-x0)) * _smooth((x3-x)/(x3-x2)) *
              _smooth((y-y0)/(y1-y0)) * _smooth((y3-y)/(y3-y2)))
    seed = dict(rows=size, columns=size,
                grid=np.stack([u, v-shift*weight/output_basis[1]], axis=-1)
                    .reshape(-1, 2))

    left, right, top, bottom = source_selection_bounds
    selected = []
    for owner in ledger['retained_pixel_footprints']:
        source, _ = paired_points(owner)
        center = np.median(source[::5] * source_basis, axis=0)
        if left < center[0] < right and top < center[1] < bottom:
            selected.append(owner)
    if not selected:
        raise ValueError('No observed material in the selected text row')
    relative = relative_guard_owners(parent, selected)
    field, report = shared_material_field(seed,
        dict(retained_pixel_footprints=relative, complete_printed_strokes=[]),
        output_size, shape=(size, size), iterations=64,
        repair_near_folds=True)
    if clip_top is not None or clip_bottom is not None:
        identity = np.stack([u, v], axis=-1)
        clip = np.ones((size, size), float)
        if clip_top is not None:
            clip *= _smooth((y-clip_top)/10)
        if clip_bottom is not None:
            clip *= _smooth((clip_bottom-y)/10)
        field = identity + (field-identity)*clip[:, :, None]
    du, dv = np.diff(field, axis=1), np.diff(field, axis=0)
    minimum = float(min(((a[..., 0]*b[..., 1]-a[..., 1]*b[..., 0])
                         * (size-1)**2).min()
                        for a in (du[:-1], du[1:])
                        for b in (dv[:, :-1], dv[:, 1:])))
    report['postclip_minimum_continuous_bilinear_jacobian'] = minimum
    report['accepted'] = bool(report['accepted'] and minimum > .2)
    if not report['accepted']:
        raise ValueError('Local material or continuous-geometry guard failed')
    result = dict(size=size, scale=scale, mode=mode, target=target,
                  selected_material=len(selected), solver=report)
    if models is not None:
        result['physical_character_profiles'] = [dict(
            components=len(q[3]), x_range=[float(q[3][:, 0].min()),
                                              float(q[3][:, 0].max())])
            for q in models]
    candidate = dict(rows=size, columns=size, grid=field.reshape(-1, 2),
                     preserved_mesh=parent, coordinate_transform=True,
                     refinement_regions=[[0, 0, 1, 1]])
    mapped = MeshSampler(candidate)
    for index in (upper_row, lower_row):
        anchors = np.vstack([np.asarray(p) for p in
            guides[index]['accepted_glyph_anchors_px']]) / source_basis
        actual = inverse_retained(mapped, anchors) * output_basis
        result[f'row_{index}'] = dict(
            median=float(np.median(actual[:, 1])),
            p90=float(np.percentile(abs(actual[:, 1]-np.median(actual[:, 1])), 90)))
    return field, result
