"""Immutable pixel-body coverage for optional exact content reconstruction.

Pixel centers still supply recognition and ownership. This separate ledger
protects their physical half-pixel corners, paired before the next content
correction. It never paints extra ink or replaces an earlier owner reference.
"""
import numpy as np
import base64
import hashlib
import zlib

from composite_mesh_inverse import inverse_retained
from composite_mesh_sampling import MeshSampler


def paired_points(owner):
    """Recover every source/reference float bit from either ledger format."""
    if 'points_f64_zlib' not in owner:
        return np.asarray(owner['source_points'], float), np.asarray(owner['reference_points'], float)
    raw=zlib.decompress(base64.b64decode(owner['points_f64_zlib'],validate=True))
    count=owner['point_count']
    if (not isinstance(count,int) or isinstance(count,bool) or count<=0
            or len(raw)!=count*4*8 or hashlib.sha256(raw).hexdigest()!=owner['points_sha256']):
        raise ValueError('Pixel footprint payload changed')
    points=np.frombuffer(raw,dtype='<f8').reshape(count,4)
    return points[:,:2],points[:,2:]


def packed_owner(owner):
    source,reference=paired_points(owner)
    if source.shape!=reference.shape:
        raise ValueError('Pixel footprint pairs changed')
    raw=np.c_[source,reference].astype('<f8').tobytes()
    return {**{k:v for k,v in owner.items() if k not in ('source_points','reference_points')},
            'point_count':len(source),'points_sha256':hashlib.sha256(raw).hexdigest(),
            'points_f64_zlib':base64.b64encode(zlib.compress(raw)).decode('ascii')}


def transpose_owner(owner):
    source,reference=paired_points(owner)
    swapped={**{k:v for k,v in owner.items() if k not in
                ('points_f64_zlib','point_count','points_sha256')},
             'pixel_size':owner['pixel_size'][::-1],
             'source_points':source[:,::-1].tolist(),'reference_points':reference[:,::-1].tolist()}
    return packed_owner(swapped) if 'points_f64_zlib' in owner else swapped


def capture(parent, groups, pixel_size, *, first_owner=0, kind='word',packed=True):
    size = np.asarray(pixel_size, dtype=float)
    if (size.shape != (2,) or not np.isfinite(size).all() or (size <= 1).any()
            or kind not in ('word', 'unknown')):
        raise ValueError('Pixel footprints need a valid measurement frame and kind')
    halo = np.array([[0, 0], [-.5, -.5], [.5, -.5], [.5, .5], [-.5, .5]]) / (size - 1)
    sampler = parent if isinstance(parent, MeshSampler) else MeshSampler(parent)
    result = []
    for index, group in enumerate(groups, first_owner):
        group = np.asarray(group, dtype=float)
        if (group.ndim != 2 or group.shape[1:] != (2,) or not len(group)
                or not np.isfinite(group).all()):
            raise ValueError('Pixel footprints need complete finite ink bodies')
        reference = (group[:, None, :] + halo).reshape(-1, 2)
        if ((reference < 0) | (reference > 1)).any():
            raise ValueError('A pixel footprint leaves the measured frame')
        owner=dict(owner_index=index, kind=kind, pixel_size=size.tolist(),
                   source_points=sampler.sample(reference).tolist(),reference_points=reference.tolist())
        result.append(packed_owner(owner) if packed else owner)
    return result


def shape_and_pairs(before, after, limit):
    a, z = np.asarray(before, float), np.asarray(after, float)
    aa, zz = a-a.mean(0), z-z.mean(0)
    u, _, v = np.linalg.svd(aa.T @ zz)
    rotation = u @ v
    fit = float(np.linalg.norm(aa @ rotation-zz, axis=1).max())
    if fit >= limit:
        # The rigid-fit violation already proves refusal; an exhaustive pair
        # search cannot turn it into a pass. Keep the all-pair upper bound.
        pairs=2*fit
        method='Rigid-fit violation; all-pair triangle upper bound'
    elif 2*fit < limit:
        pairs = 2*fit
        method = 'All-pair triangle inequality certificate'
    else:
        pairs = 0.
        method = 'Every pair evaluated in bounded blocks'
        for start in range(0, len(a), 128):
            first = np.linalg.norm(a[start:start+128, None]-a[None], axis=2)
            last = np.linalg.norm(z[start:start+128, None]-z[None], axis=2)
            pairs = max(pairs, float(abs(last-first).max()))
            if pairs>=limit:
                method='Observed pair violation certificate'
                break
    return dict(rigid_shape_error_px=fit, pair_distance_change_or_bound_px=pairs,
                pair_proof=method,
                accepted=bool(np.linalg.det(rotation) > 0 and max(fit, pairs) < limit))


def audit(result, owners):
    if not owners:
        raise ValueError('Complete pixel protection needs registered material')
    sources, references, scales, limits = [], [], [], []
    for owner in owners:
        source,reference=paired_points(owner)
        size = np.asarray(owner['pixel_size'], float)
        if (source.ndim != 2 or source.shape[1:] != (2,) or not len(source)
                or len(source) % 5 or reference.shape != source.shape
                or not np.isfinite(source).all() or not np.isfinite(reference).all()
                or size.shape != (2,) or not np.isfinite(size).all() or (size <= 1).any()
                or owner['kind'] not in ('word', 'unknown')):
            raise ValueError('Invalid immutable pixel footprint pairs')
        sources.append(source)
        references.append(reference)
        scales.append(size-1)
        limits.append(.03 if owner['kind'] == 'unknown' else .1)
    actual = np.split(inverse_retained(MeshSampler(result), np.vstack(sources)),
                      np.cumsum([len(q) for q in sources])[:-1])
    return [dict(owner_index=owner['owner_index'], kind=owner['kind'],
                 footprint_samples=len(source), limit_px=limit,
                 **shape_and_pairs(reference*scale, moved*scale, limit))
            for owner, source, reference, moved, scale, limit in
            zip(owners, sources, references, actual, scales, limits)]
