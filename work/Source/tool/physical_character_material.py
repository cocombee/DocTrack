"""Physical character owners for a failed complete-text-row correction.

This is an opt-in diagnostic helper. The immutable original owner ledger still
proves coverage of every pixel and corner; OCR owners are never rewritten.
Character owners can instead guard shape when an OCR owner spans two real rows.
The caller must establish that a complete-row warp failed before using them.
"""

import cv2
import numpy as np

from composite_mesh_inverse import inverse_retained
from composite_mesh_sampling import MeshSampler
from material_pixel_footprints import capture, packed_owner, paired_points


IDENTITY = dict(rows=2, columns=2, grid=[[0, 0], [1, 0], [0, 1], [1, 1]])


def partition_initial_material_by_components(
    owners, component_mask, *, first_owner=1000, max_weak_distance_px=4
):
    """Partition selected original pixel centers without losing weak ink.

    ``owners`` must be registered on the initial projected raster. The mask
    contains observed print, never OCR rectangles or inferred line segments.
    A weak registered pixel joins a nearby component already containing
    registered print; otherwise it remains an individual material owner.
    Every resulting owner includes the original center and four half-pixel
    corners. The returned owners do not replace the original ledger.
    """
    mask = np.asarray(component_mask)
    if mask.ndim != 2 or mask.size == 0 or not owners:
        raise ValueError('Physical character partition needs a mask and original owners')
    if not isinstance(first_owner, int) or first_owner < 0:
        raise ValueError('Invalid first character owner index')
    if not isinstance(max_weak_distance_px, int) or max_weak_distance_px < 0:
        raise ValueError('Invalid weak-pixel assignment radius')
    height, width = mask.shape
    size = np.array([width, height], dtype=float)
    basis = size - 1
    all_centers = []
    for owner in owners:
        if np.asarray(owner['pixel_size'], dtype=float).tolist() != size.tolist():
            raise ValueError('Original owners and mask have different pixel sizes')
        source, reference = paired_points(owner)
        if len(source) % 5 or np.max(np.abs(source-reference)) > 1e-12:
            raise ValueError('Character partition requires initial, unmodified registration')
        centers = source[::5] * basis
        rounded = np.rint(centers)
        if np.max(np.abs(centers - rounded)) > 1e-8:
            raise ValueError('Original registered centers must be physical pixels')
        all_centers.append(rounded.astype(np.int32))
    registered = np.vstack(all_centers)
    if np.any(registered < 0) or np.any(registered >= size.astype(int)):
        raise ValueError('Original registered material leaves the image')
    unique = np.unique(registered, axis=0)
    _, labels, _, _ = cv2.connectedComponentsWithStats((mask != 0).astype(np.uint8))
    assigned = labels[unique[:, 1], unique[:, 0]].astype(np.int64)
    eligible = set(int(q) for q in np.unique(assigned) if q > 0)
    weak = np.flatnonzero(assigned == 0)
    nearest_count = 0
    isolated_count = 0
    singleton_id = int(labels.max()) + 1
    for index in weak:
        x, y = unique[index]
        found = None
        for radius in range(1, max_weak_distance_px + 1):
            left, top = max(0, x-radius), max(0, y-radius)
            patch = labels[top:min(height, y+radius+1), left:min(width, x+radius+1)]
            yy, xx = np.where(np.isin(patch, list(eligible)))
            if not len(xx):
                continue
            actual_x, actual_y = xx+left, yy+top
            squared = (actual_x-x)**2 + (actual_y-y)**2
            best = np.lexsort((patch[yy, xx], squared))[0]
            found = int(patch[yy[best], xx[best]])
            break
        if found is None:
            assigned[index] = singleton_id
            singleton_id += 1
            isolated_count += 1
        else:
            assigned[index] = found
            nearest_count += 1
    groups = [unique[assigned == label] for label in np.unique(assigned)]
    if sum(map(len, groups)) != len(unique):
        raise AssertionError('A registered physical pixel was not assigned')
    character_owners = capture(
        IDENTITY, [group / basis for group in groups], size,
        first_owner=first_owner, kind='word', packed=True,
    )
    report = dict(
        original_owners=len(owners),
        registered_centers=len(registered),
        unique_original_pixel_centers=len(unique),
        duplicate_centers_removed=len(registered)-len(unique),
        direct_component_centers=len(unique)-len(weak),
        weak_centers_assigned_to_observed_components=nearest_count,
        isolated_original_centers=isolated_count,
        physical_component_owners=len(character_owners),
        complete_original_center_coverage=True,
        original_ledger_unchanged=True,
    )
    return character_owners, report


def relative_guard_owners(parent_mesh, original_character_owners):
    """Express the same original footprint samples in the prior output frame.

    Relative owners check only *new* shape damage; they cannot certify that
    the prior page mesh was already shape-perfect. Keep the original owners
    and original source ledger for end-to-end qualification.
    """
    parent = MeshSampler(parent_mesh)
    result = []
    for owner in original_character_owners:
        source, _ = paired_points(owner)
        prior_output = inverse_retained(parent, source)
        result.append(packed_owner(dict(
            owner_index=owner['owner_index'], kind=owner['kind'],
            pixel_size=owner['pixel_size'],
            source_points=prior_output.tolist(),
            reference_points=prior_output.tolist(),
            guard_basis='relative_prior_output',
        )))
    return result
