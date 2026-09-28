"""Observed dash-body and complete-row evidence for curved printed rules.

This opt-in diagnostic separates two physical requirements: each real dash
keeps its own printed shape, while the row of dashes becomes straighter. It
does not replace the immutable complete-stroke ledger or paint blank gaps.
"""

import cv2
import numpy as np

from composite_mesh_inverse import inverse_retained
from composite_mesh_sampling import MeshSampler
from material_pixel_footprints import audit, capture, paired_points
from physical_character_material import IDENTITY


def register_observed_dashes(contrast, strokes, pixel_size, *, threshold=10,
                             first_owner=3000):
    """Register actual connected dash pixels touched by complete rule traces."""
    image = np.asarray(contrast)
    size = np.asarray(pixel_size, dtype=float)
    if image.ndim != 2 or list(image.shape) != [int(size[1]), int(size[0])]:
        raise ValueError('Dash evidence and stroke pixel size must match')
    if not strokes:
        raise ValueError('Complete printed strokes are required')
    _, labels, stats, _ = cv2.connectedComponentsWithStats(
        (image > threshold).astype(np.uint8))
    basis = size-1
    groups = []
    rows = []
    used = set()
    for stroke_index, stroke in enumerate(strokes):
        if stroke['axis'] != 0:
            continue
        source = np.asarray(stroke['source_points'], dtype=float)
        if source.ndim != 2 or source.shape[1] != 2 or len(source) < 3:
            raise ValueError('A horizontal stroke needs physical trace samples')
        pixels = np.rint(source*basis).astype(int)
        if np.any(pixels < 0) or np.any(pixels >= size.astype(int)):
            raise ValueError('A printed trace leaves the source raster')
        hit = labels[pixels[:, 1], pixels[:, 0]]
        values, counts = np.unique(hit, return_counts=True)
        chosen = []
        for label, count in zip(values, counts):
            if label == 0 or count < 2:
                continue
            _, _, width, height, area = stats[label]
            if width < 3 or height > 15 or area < 5:
                continue
            if int(label) in used:
                raise ValueError('One physical dash belongs to multiple complete rows')
            used.add(int(label))
            chosen.append(int(label))
        owner_indices = []
        for label in sorted(chosen):
            y, x = np.where(labels == label)
            groups.append(np.c_[x, y]/basis)
            owner_indices.append(first_owner+len(groups)-1)
        rows.append(dict(stroke_index=stroke_index, owner_indices=owner_indices,
                         original_trace_samples=len(source),
                         directly_matched_trace_samples=int(np.isin(hit, chosen).sum()),
                         observed_dash_bodies=len(chosen)))
    if not groups:
        raise ValueError('No original dash material is supported by these traces')
    owners = capture(IDENTITY, groups, size, first_owner=first_owner,
                     kind='unknown', packed=True)
    report = dict(rows=rows, original_dash_pixels=sum(len(group) for group in groups),
                  material_owner_count=len(owners), observed_gaps_remain_empty=True,
                  original_complete_strokes_unchanged=True)
    return owners, report


def audit_dash_rows(result_mesh, dash_owners, registration):
    """Report rigid printed bodies and horizontal alignment separately."""
    checked = audit(result_mesh, dash_owners)
    sampler = MeshSampler(result_mesh)
    by_index = {owner['owner_index']: (owner, check)
                for owner, check in zip(dash_owners, checked)}
    rows = []
    for row in registration['rows']:
        centers_before = []
        centers_after = []
        body_checks = []
        for index in row['owner_indices']:
            owner, check = by_index[index]
            source, _ = paired_points(owner)
            basis = np.asarray(owner['pixel_size'], dtype=float)-1
            body = source[::5]
            output = inverse_retained(sampler, body)
            centers_before.append(np.median(body*basis, axis=0))
            centers_after.append(np.median(output*basis, axis=0))
            body_checks.append(check)
        before = np.asarray(centers_before)
        after = np.asarray(centers_after)
        if len(before) < 3:
            raise ValueError('A complete row needs at least three observed dashes')
        def scatter(points):
            y = points[:, 1]
            return dict(p90_px=float(np.percentile(abs(y-np.median(y)), 90)),
                        maximum_px=float(abs(y-np.median(y)).max()),
                        span_px=float(np.ptp(y)),
                        horizontal_coverage_px=float(np.ptp(points[:, 0])))
        rows.append(dict(stroke_index=row['stroke_index'],
                         observed_dash_bodies=len(body_checks),
                         original_shape_pass=sum(q['accepted'] for q in body_checks),
                         original_shape_worst_rigid_error_px=max(q['rigid_shape_error_px']
                                                                  for q in body_checks),
                         before=scatter(before), after=scatter(after)))
    return rows
