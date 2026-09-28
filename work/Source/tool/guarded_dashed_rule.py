"""Straighten an observed dashed ruling without flattening its ink bodies.

Adjacent dashes can touch at pixel corners. Treating their complete run as one
rigid owner preserves a visible paper bend. This correction instead follows
the measured dash centers while requiring every ink column to keep its width
and thickness, all neighbouring owners to remain rigid, and the mesh to stay
continuous and nonfolding.
"""
import numpy as np

from composite_mesh_inverse import inverse_retained
from composite_mesh_sampling import MeshSampler
from material_pixel_footprints import audit, paired_points


def _smooth(value):
    value = np.clip(value, 0, 1)
    return value * value * (3 - 2 * value)


def _line_profile(line):
    ordered = np.asarray(line, float)[np.argsort(np.asarray(line)[:, 0])]
    xx, inverse = np.unique(ordered[:, 0], return_inverse=True)
    yy = np.asarray([np.median(ordered[inverse == i, 1])
                     for i in range(len(xx))])
    return xx, yy


def refine_dashed_rule(parent, line, ledger, *, output_size,
                       material_source_size, rows=1025, columns=1025):
    """Return one accepted coordinate layer and a physical-ink certificate.

    ``line`` is a measured horizontal rule in source-normalized coordinates.
    The ledger was registered on that same unwarped source raster. Neither a
    document identifier nor a fixed page position selects the ruling.
    """
    source_basis = np.asarray(material_source_size, float) - 1
    output_basis = np.asarray(output_size, float) - 1
    rule = np.asarray(line, float)
    if (rule.ndim != 2 or rule.shape[1] != 2 or len(rule) < 25 or
            not np.isfinite(rule).all() or np.ptp(rule[:, 0]) < .35):
        raise ValueError('A broad observed horizontal rule is required')
    if rows < 33 or columns < 33 or rows % 2 != 1 or columns % 2 != 1:
        raise ValueError('A regular odd lattice is required')
    source_x = rule[:, 0] * source_basis[0]
    line_x, line_y = _line_profile(rule * source_basis)
    selected, remaining, source_footprints, centers = [], [], [], []
    for owner in ledger['retained_pixel_footprints']:
        source, _ = paired_points(owner)
        pixels = source[::5] * source_basis
        center = np.mean(pixels, axis=0)
        span = np.ptp(pixels, axis=0)
        on_rule = (owner['kind'] == 'unknown' and
                   source_x.min() - 5 < center[0] < source_x.max() + 5 and
                   abs(center[1] - np.interp(center[0], line_x,
                                                line_y)) < 9 and
                   span[0] > 1.15 * max(1, span[1]))
        if on_rule:
            selected.append(owner)
            source_footprints.append(source)
            centers.append(np.mean(source[::5], axis=0))
        else:
            remaining.append(owner)
    if len(selected) < 12:
        raise ValueError('The dashed ruling lacks independent ink bodies')

    parent_sampler = MeshSampler(parent)
    original_centers = np.asarray(centers)
    mapped_centers = inverse_retained(parent_sampler, original_centers) * output_basis
    order = np.argsort(mapped_centers[:, 0])
    x, y = mapped_centers[order].T
    if np.ptp(x) < output_basis[0] * .5 or np.max(np.diff(x)) > 5 * np.median(np.diff(x)):
        raise ValueError('The dash centers do not cover one continuous ruling')
    before_range = float(np.ptp(y))
    if before_range < .5:
        raise ValueError('The observed ruling is already straight')
    footprint_outputs = inverse_retained(parent_sampler,
                                         np.vstack(source_footprints)) * output_basis
    low = footprint_outputs.min(axis=0)
    high = footprint_outputs.max(axis=0)
    target = float(np.median(y))
    if max(abs(target-low[1]), abs(high[1]-target)) > 20:
        raise ValueError('The ruling spans an implausible print thickness')

    u, v = np.meshgrid(np.linspace(0, 1, columns), np.linspace(0, 1, rows))
    px, py = u * output_basis[0], v * output_basis[1]
    shift = np.interp(px, x, target-y)
    weight = (_smooth((px-(low[0]-18))/16) *
              _smooth(((high[0]+18)-px)/16) *
              _smooth((py-(low[1]-18))/16) *
              _smooth(((high[1]+18)-py)/16))
    field = np.stack([u, v-shift*weight/output_basis[1]], axis=-1)
    candidate = dict(rows=rows, columns=columns, grid=field.reshape(-1, 2),
                     preserved_mesh=parent, coordinate_transform=True,
                     refinement_regions=[[0, 0, 1, 1]])
    sampler = MeshSampler(candidate)
    after_centers = inverse_retained(sampler, original_centers) * output_basis
    after_range = float(np.ptp(after_centers[:, 1]))
    if after_range > .25 or after_range > before_range * .25:
        raise ValueError('The proposed ruling is still visibly bent')

    normal_change = []
    lateral_change = []
    maximum_shift = 0.
    for source in source_footprints:
        before = inverse_retained(parent_sampler, source) * output_basis
        after = inverse_retained(sampler, source) * output_basis
        lateral_change.append(float(np.max(abs(after[:, 0]-before[:, 0]))))
        maximum_shift = max(maximum_shift,
                            float(np.max(abs(after[:, 1]-before[:, 1]))))
        columns_px = np.rint(source[:, 0] * source_basis[0] * 2).astype(int)
        for column in np.unique(columns_px):
            chosen = columns_px == column
            if chosen.sum() >= 2:
                normal_change.append(float(abs(np.ptp(after[chosen, 1]) -
                                               np.ptp(before[chosen, 1]))))
    thickness_error = max(normal_change, default=float('inf'))
    lateral_error = max(lateral_change)
    if thickness_error > .05 or lateral_error > .02 or maximum_shift > 4:
        raise ValueError('A dash lost thickness or moved sideways')
    other_checks = audit(candidate, remaining)
    if not all(item['accepted'] for item in other_checks):
        raise ValueError('Straightening the ruling changed neighbouring print')

    du, dv = np.diff(field, axis=1), np.diff(field, axis=0)
    minimum = float(min(((a[..., 0]*b[..., 1]-a[..., 1]*b[..., 0]) *
                         (rows-1)*(columns-1)).min()
                        for a in (du[:-1], du[1:])
                        for b in (dv[:, :-1], dv[:, 1:])))
    if (minimum <= .2 or not np.isfinite(field).all() or
            field.min() < 0 or field.max() > 1):
        raise ValueError('The ruling correction folds or leaves the raster')
    return field, dict(
        accepted=True, rule_ink_bodies=len(selected),
        rule_owner_indices=[owner['owner_index'] for owner in selected],
        other_material_owners=len(other_checks),
        other_material_passes=sum(item['accepted'] for item in other_checks),
        centerline_range_before_px=before_range,
        centerline_range_after_px=after_range,
        maximum_column_thickness_change_px=thickness_error,
        maximum_lateral_ink_change_px=lateral_error,
        maximum_rule_ink_shift_px=maximum_shift,
        minimum_local_continuous_bilinear_jacobian=minimum,
        material_policy='rule columns preserve thickness; other owners remain rigid',
    )
