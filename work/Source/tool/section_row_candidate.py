"""Optional printed-row experiment composed on the accepted full map.

This module is intentionally not the application default. Sections are measured
in the existing result, and full glyph envelopes move rigidly where an ordered
row requires it. Unsupported proposals return the exact prior result.
"""
import json
import sys
from pathlib import Path
import cv2
import numpy as np
import protected_ink_mesh as accepted
from precise_mesh_inverse import inverse as precise_inverse
from printed_rule_sections import measure_sections, word_ink_mask
from ordered_constraint_solver import ordered_shifts, infeasible_cycle
from independent_word_owners import word_groups
from monotone_stock_relaxation import relax,relax_tangent_stock
from joint_stock_projection import crossing_pins, project as project_joint_stock
from composite_mesh_sampling import MeshSampler
from composite_mesh_inverse import inverse_near,inverse_retained
from coordinate_stock import allocate_stock
from normal_composition_guard import validate_normal_composition
from measured_ink_bands import pixel_footprint, column_bands
from printed_rule_component_owners import orthogonal_rule_components
p = accepted.accepted.prior

def associate_rows(sections, width, height):
    """Associate non-overlapping ink fragments using endpoint continuity."""
    chains = []
    for section in sorted(sections, key=lambda g: g[0, 0]):
        choices = []
        for index, chain in enumerate(chains):
            previous = chain[-1]
            gap = section[0, 0] - previous[-1, 0]
            if not 0 < gap < 0.07:
                continue
            n = min(12, len(previous), len(section))
            previous_slope = np.polyfit(previous[-n:, 0], previous[-n:, 1], 1)[0]
            next_slope = np.polyfit(section[:n, 0], section[:n, 1], 1)[0]
            separation = abs(section[0, 1] - previous[-1, 1])
            predicted_error = abs(section[0, 1] - previous[-1, 1] - (previous_slope + next_slope) * gap / 2)
            if separation < 0.012 and predicted_error < max(3 / height, gap * 0.3):
                choices.append((predicted_error, index))
        if choices:
            chains[min(choices)[1]].append(section)
        else:
            chains.append([section])
    rows = []
    for chain in chains:
        points = np.vstack(chain)
        covered = sum((np.ptp(section[:, 0]) for section in chain))
        # Require measured sections on both sides, with substantial coverage.
        # A small bent fragment cannot justify moving a complete layout row.
        if len(chain) < 2 or covered < 0.6 or np.ptp(points[:, 0]) < 0.7:
            continue
        if np.ptp(np.percentile(points[:, 1], [5, 95])) * height < 2 or np.std(points[:, 1]) * height < 0.6:
            continue
        rows.append(chain)
    return rows

def current_measurement(path, base, *, exact_layers=False):
    sampler = MeshSampler(base) if exact_layers else None
    if sampler is None:
        old = accepted.accepted.grid_of(base)
    else:
        layers=[];node=sampler
        while node is not None:
            layers.append((node.rows,node.columns) if node.window is None else tuple(node.window[:2]))
            node=node.parent
        old=sampler.dense(max(s[0] for s in layers),max(s[1] for s in layers))
    raw = cv2.imread(str(path), -1)
    if raw is None:
        raise ValueError('Cannot read the measurement image')
    h, w = raw.shape[:2]
    mask = raw[:, :, 3] if raw.shape[2] == 4 else np.full((h, w), 255, np.uint8)
    alpha = mask[:, :, None] / 255
    source = cv2.cvtColor((raw[:, :, :3] * alpha + 255 * (1 - alpha)).astype(np.uint8), cv2.COLOR_BGR2GRAY)
    uv = np.stack(np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h)), -1)
    mapped = (p.sample(old, uv.reshape(-1, 2)) if sampler is None else sampler.sample(uv.reshape(-1,2))).reshape(h,w,2).astype(np.float32)
    map_x, map_y = (mapped[:, :, 0] * (w - 1), mapped[:, :, 1] * (h - 1))
    gray = cv2.remap(source, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    current_alpha = cv2.remap(mask, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    radius = max(1, round(5 * max(h, w) / 1280))
    safe = cv2.erode(current_alpha, np.ones((radius * 2 + 1, radius * 2 + 1), np.uint8))
    return gray, safe, source, old


def has_coordinate_layer(base):
    while isinstance(base,dict):
        if base.get('coordinate_transform',False):return True
        base=base.get('preserved_mesh')
    return False


def matching_rules(points, mapped_rules, height):
    """Identify the same printed row by supported overlap, not page position."""
    matches = []
    for index, rule in enumerate(mapped_rules):
        rule = rule[np.argsort(rule[:, 0])]
        lo = max(points[0, 0], rule[0, 0])
        hi = min(points[-1, 0], rule[-1, 0])
        if hi - lo < .2:
            continue
        x = np.linspace(lo, hi, 101)
        error = abs(np.interp(x, points[:, 0], points[:, 1]) -
                    np.interp(x, rule[:, 0], rule[:, 1])) * height
        if np.median(error) < 3 and np.percentile(error, 75) < 5:
            matches.append(index)
    return matches

def neighbouring_rule_guards(old,horizontal,vertical,selected,width,height,*,sample_function=None):
    """Keep measured print outside the selected rule's ink fixed normally."""
    guards = []
    sample_function=(lambda points:p.sample(old,points)) if sample_function is None else sample_function
    for guide in horizontal:
        belongs = False
        for target in selected:
            lo = max(guide[0, 0], target[0, 0])
            hi = min(guide[-1, 0], target[-1, 0])
            if (hi - lo) * (width - 1) < 8:
                continue
            x = np.linspace(lo, hi, 65)
            error = abs(np.interp(x, guide[:, 0], guide[:, 1]) -
                        np.interp(x, target[:, 0], target[:, 1])) * (height - 1)
            if np.percentile(error, 95) < 2:
                belongs = True
                break
        if not belongs:
            guards.append({'normal_axis': 1,
                           'points': sample_function(guide).tolist()})
    guards.extend({'normal_axis': 0,
                   'points': sample_function(guide[:,::-1]).tolist()}
                  for guide in vertical)
    return guards

def joined_rule_extent(trace, perpendicular, width, height):
    """Include a supported nearby crossing in a temporary fixed-line band."""
    left, right = float(trace[0, 0]), float(trace[-1, 0])
    radius = 3 * max(width, height) / 1280
    for crossing in perpendicular:
        crossing = crossing[np.argsort(crossing[:, 1])]
        for endpoint, before in [(trace[0], True), (trace[-1], False)]:
            y = endpoint[1]
            if not crossing[0, 1] - radius / height <= y <= crossing[-1, 1] + radius / height:
                continue
            x = float(np.interp(y, crossing[:, 1], crossing[:, 0]))
            gap = (endpoint[0] - x) if before else (x - endpoint[0])
            if 0 <= gap * width <= radius:
                if before:
                    left = min(left, x)
                else:
                    right = max(right, x)
    return left, right

def measurement_aligned_resolution(rows, columns, height, width):
    """Keep measured tangent vertices on actual mesh knots.

    Rule traces use integer tangent pixels and half-pixel ridge centers. A
    uniform mesh with an unrelated denominator averages across their corners,
    leaving a wave even when the requested target is exactly constant.
    """
    normal_denominator = 2 * (height - 1)
    tangent_denominator = width - 1
    if min(rows, columns, height, width) < 2:
        raise ValueError('Measurement and mesh dimensions must exceed one')
    return (((rows - 2) // normal_denominator + 1) * normal_denominator + 1,
            ((columns - 2) // tangent_denominator + 1) * tangent_denominator + 1)


def registered_curve_samples(points,columns,*,tolerance):
    """Fit uniform control values to the complete registered material trace.

    Retained inverse points need not lie on raster knots. Sampling their curve
    onto knots and back averages across short bends. These local linear
    equations instead constrain each original trace point at its actual
    bilinear interpolation weights. Incompatible evidence is declined.
    """
    points=np.asarray(points,dtype=float)
    if (points.ndim!=2 or points.shape[1]!=2 or len(points)<2 or columns<2 or
            not np.isfinite(points).all() or (points<0).any() or (points>1).any() or
            (np.diff(points[:,0])<0).any() or not 0<tolerance<1):
        raise ValueError('Invalid complete registered curve')
    u=np.arange(columns)/(columns-1)
    curve=np.interp(u,points[:,0],points[:,1])
    positions=points[:,0]*(columns-1)
    left=np.minimum(np.floor(positions).astype(int),columns-2)
    right=left+1;fraction=positions-left
    low_weight=1-fraction;high_weight=fraction
    denominator=low_weight**2+high_weight**2
    for _ in range(256):
        error=points[:,1]-(curve[left]*low_weight+curve[right]*high_weight)
        if abs(error).max()<=tolerance:
            if (curve<0).any() or (curve>1).any():break
            return curve
        for lo,hi,a,b,n,target in zip(left,right,low_weight,high_weight,denominator,points[:,1]):
            correction=(target-curve[lo]*a-curve[hi]*b)/n
            curve[lo]+=correction*a;curve[hi]+=correction*b
    raise ValueError('Uniform mesh cannot preserve the complete registered trace')


def refine(path, base, *, guides=None, target_guide=None, include_wide_ink=False, audit_log=None,
           minimum_resolution=(1025, 513), align_to_existing=False, word_ink_groups=None,
           measurement=None, words_mask=None, extra_protected=None, tangent_padding=None,
           separate_structural_ink=False, exact_ordering=False, independent_word_owners=False,
           stock_relaxation=0,stock_coordinate_relaxation=0,
           precise_geometry=False,word_reading_axis=0,joint_stock_iterations=0,
           additional_rulers=None,preserve_neighbouring_rules=False,
           coordinate_composition=False,word_translation_links=None,
           word_owner_envelopes=None,tangent_ink_padding_px=None,
           fixed_word_owners=(),measured_ink_bands=False,free_tangent_rules=False,
           word_flanked_evidence=None,stock_area_reserve=.21,
           registered_guides=False):
    fixed_word_owners=tuple(fixed_word_owners)
    if registered_guides and (not coordinate_composition or guides is None):
        raise ValueError('Registered guides require explicit retained coordinate traces')
    if has_coordinate_layer(base) and not coordinate_composition:
        raise ValueError('Retained coordinate maps need coordinate-aware refinement')
    base_inverse=precise_inverse if precise_geometry or joint_stock_iterations or coordinate_composition else p.inverse
    gray, safe, source, old = current_measurement(path,base,exact_layers=True) if measurement is None and coordinate_composition else (current_measurement(path,base) if measurement is None else measurement)
    parent=MeshSampler(base) if coordinate_composition else None
    if parent is not None and not has_coordinate_layer(base):
        parent._inverse_seed=old
    candidate_sampler=None
    query=None
    if coordinate_composition and (joint_stock_iterations or stock_coordinate_relaxation):
        raise ValueError('Coordinate composition cannot use source-grid projection')
    if measured_ink_bands and not (coordinate_composition and independent_word_owners
                                  and exact_ordering and word_ink_groups):
        raise ValueError('Measured ink bands require exact coordinate word ownership')
    if free_tangent_rules and not measured_ink_bands:
        raise ValueError('Tangent rule proxies require measured ink footprints')
    if word_flanked_evidence is not None and not measured_ink_bands:
        raise ValueError('Word-flanked rows require measured physical ink protection')
    if not np.isfinite(stock_area_reserve) or not .21<=stock_area_reserve<=1:
        raise ValueError('Stock allocation must retain at least the existing area reserve')
    def sample_old(points):
        return p.sample(old,points) if parent is None else parent.sample(points)
    def inverse(grid,points):
        if parent is None:return base_inverse(grid,points)
        before=inverse_retained(parent,points)
        if grid is old:return before
        return inverse_near(candidate_sampler,points,precise_inverse(query,before))
    h, w = gray.shape
    owner_padding=.004
    if tangent_ink_padding_px is not None:
        if (not coordinate_composition or not np.isfinite(tangent_ink_padding_px)
                or tangent_ink_padding_px<1):
            raise ValueError('Measured tangent padding requires coordinate composition and at least one pixel')
        owner_padding=float(tangent_ink_padding_px)/(w-1)
    if guides is None:
        rows = associate_rows(measure_sections(gray, safe), w, h)
        if not rows:
            return base
        guides = max(rows, key=lambda group: np.ptp(np.percentile(np.vstack(group)[:, 1], [5, 95])) * sum((np.ptp(g[:, 0]) for g in group)))
    retargeted_rulers = set()
    shared_row = None
    if word_flanked_evidence is not None:
        from word_flanked_rule_rows import word_flanked_rows
        registered = [(i, inverse(old, np.asarray(r['points'])))
                      for i, r in enumerate(base.get('established_rulers', []))
                      if r['normal_axis'] == 1]
        candidates = word_flanked_rows(guides, word_flanked_evidence['quads'],
                word_ink_groups, word_flanked_evidence['native_count'],
                [q for _, q in registered], w, h)
        if len(candidates) != 1:
            raise ValueError('A shared row needs one unambiguous registered target')
        shared_row = candidates[0]
        level = shared_row['target']
        if target_guide is not None and (np.asarray(target_guide).shape != (2, 2) or
                abs(np.asarray(target_guide)[:, 1] - level).max() * h > 1e-6):
            raise ValueError('Shared row target differs from its immutable registered anchor')
        selected = shared_row['guides'][1 - shared_row['anchor_side']]
        for index, trace in registered:
            if (trace[:, 0].min() < selected[0, 0] - .5 / (w - 1) or
                    trace[:, 0].max() > selected[-1, 0] + .5 / (w - 1) or
                    np.ptp(trace[:, 0]) < .8 * np.ptp(selected[:, 0])):
                continue
            error = abs(trace[:, 1] - np.interp(trace[:, 0], selected[:, 0], selected[:, 1])) * (h - 1)
            if (error.max() <= 2 * max(w, h) / 1280 and
                    abs(np.median(trace[:, 1]) - level) * h > .1):
                retargeted_rulers.add(index)
        measured_guides = [g.tolist() for g in guides]
        guides = list(shared_row['guides'])
        # Preserve the working section's complete registered material trace.
        # Re-measured raster ridge centers can jitter by a fraction of a pixel.
        guides[shared_row['anchor_side']] = registered[shared_row['anchor_indices'][0]][1]
        shared_row = {**shared_row, 'guides': guides, 'measured_guides': measured_guides}
        target_guide = np.array([[min(g[0, 0] for g in guides), level],
                                 [max(g[-1, 0] for g in guides), level]])
    points = np.vstack(guides)
    points = points[np.argsort(points[:, 0])]
    if preserve_neighbouring_rules:
        word_mask = word_ink_mask(gray) if words_mask is None else words_mask
        measured_horizontal = measure_sections(gray, safe, word_mask=word_mask)
        # Vertical stages pass a transposed raster. OpenCV drawing requires
        # a row-contiguous destination even when the source is a valid view.
        crossings = np.zeros_like(gray, order='C')
        for guide in measured_horizontal:
            cv2.polylines(crossings, [np.rint(guide * [w-1, h-1]).astype(np.int32)], False, 1, 9)
        measured_vertical = measure_sections(gray.T, safe.T, word_mask=word_mask.T,
                                             crossing_mask=crossings.T)
        additional_rulers = [*([] if additional_rulers is None else additional_rulers),
                             *neighbouring_rule_guards(old, measured_horizontal,
                                                      measured_vertical,guides,w,h,sample_function=sample_old)]
    rules = p.reference.extract_rules(source)
    mapped_rules = [inverse(old, g) for g in rules] if align_to_existing else []
    primary = matching_rules(points, mapped_rules, h)
    # Anchor a longer measured row to its established main section. A bowed
    # extension must not drag the already working central section to its mean.
    target = (float(np.median(np.vstack([mapped_rules[i] for i in primary])[:, 1]))
              if primary else float(points[:, 1].mean()))
    target_guide = (np.array([[points[0, 0], target], [points[-1, 0], target]])
                    if target_guide is None else np.asarray(target_guide))
    source_rulers = sample_old(points)
    ink = (cv2.GaussianBlur(gray.astype(float), (0, 0), 7) - gray > 10).astype(np.uint8) * 255
    ink[safe == 0] = 0
    for g in guides:
        cv2.polylines(ink, [np.rint(g * [w - 1, h - 1]).astype(np.int32)], False, 0, 5)
    if separate_structural_ink and word_ink_groups:
        independent_words=np.zeros_like(ink)
        for group in word_ink_groups:
            group=np.asarray(group)
            if not len(group):continue
            lo=np.floor(group.min(0)*[w-1,h-1]).astype(int)-2
            hi=np.ceil(group.max(0)*[w-1,h-1]).astype(int)+2
            independent_words[max(0,lo[1]):min(h,hi[1]+1),max(0,lo[0]):min(w,hi[0]+1)]=1
        # Connected frame/letter summaries contain empty stock and cannot be
        # rigid glyph rectangles. Remove only independently measured rule ink
        # outside the full word envelope before forming component summaries.
        structural=np.zeros_like(ink)
        orthogonal=[g[:,::-1] for g in p.reference.extract_rules(source.T)]
        traces=[inverse(old,g) for g in rules+orthogonal]
        traces += [inverse(old,np.asarray(r['points']))
                   for r in base.get('established_rulers',[])]
        measured_horizontal=[g for g in measure_sections(gray,safe,broad_profile=True,
                                  word_mask=independent_words) if np.ptp(g[:,0])>.2]
        crossings=np.zeros_like(ink)
        for trace in measured_horizontal+guides:
            cv2.polylines(crossings,[np.rint(trace*[w-1,h-1]).astype(np.int32)],False,1,7)
        traces += measured_horizontal
        traces += [g[:,::-1] for g in measure_sections(gray.T,safe.T,broad_profile=True,
            word_mask=independent_words.T,crossing_mask=crossings.T) if np.ptp(g[:,0])>.2]
        for trace in traces:
            cv2.polylines(structural,[np.rint(trace*[w-1,h-1]).astype(np.int32)],False,1,5)
        ink[(structural>0)&(independent_words==0)]=0
    _, component_labels, stats, _ = cv2.connectedComponentsWithStats(ink)
    boxes = []
    ink_clouds = []
    actual_words = (word_ink_mask(gray) if words_mask is None else words_mask) if align_to_existing or include_wide_ink else None
    word_mask = actual_words if include_wide_ink else None
    for component_index,(x, y, cw, ch, area) in enumerate(stats[1:],start=1):
        if area < 12 or min(cw, ch) < 3 or (ch > h * 0.05 and ch > cw * 3):
            continue
        if include_wide_ink and ch > h * 0.05:
            # A connected form frame is not a single rigid word envelope.
            # Its measured rules and crossed normals are guarded separately.
            continue
        if word_mask is not None and not word_mask[y:y + ch, x:x + cw].any():
            continue
        if align_to_existing and not actual_words[y:y + ch, x:x + cw].any() and cw >= ch * 1.5 and cw >= 5:
            # A faint remnant of this measured rule is not a glyph. Require
            # actual contrast pixels beside the same trace before excluding
            # it; a word envelope always keeps its full protection.
            cy, cx = np.where(ink[y:y + ch, x:x + cw])
            xx = (cx + x) / (w - 1)
            yy = (cy + y) / (h - 1)
            inside = (xx >= points[0, 0]) & (xx <= points[-1, 0])
            distance = abs(yy - np.interp(xx, points[:, 0], points[:, 1])) * h
            if len(xx) and np.mean(inside & (distance < 3 * max(h, w) / 1280)) > .9:
                continue
        if cw > ch * (30 if include_wide_ink else 5):
            continue
        boxes.append(np.array([[x, y], [x + cw - 1, y], [x + cw - 1, y + ch - 1], [x, y + ch - 1], [(2 * x + cw - 1) / 2, y], [(2 * x + cw - 1) / 2, y + ch - 1]]) / [w - 1, h - 1])
        if measured_ink_bands:
            cy,cx=np.where(component_labels[y:y+ch,x:x+cw]==component_index)
            ink_clouds.append(pixel_footprint(np.c_[cx+x,cy+y]/[w-1,h-1],w,h))
    boxes = np.array(boxes).reshape(-1, 6, 2)
    native_start=len(boxes)
    whole_words = [] if word_ink_groups is None else [np.asarray(g) for g in word_ink_groups if len(g)]
    if fixed_word_owners and not independent_word_owners:
        raise ValueError('Fixed words require independent word ownership')
    if (len(set(fixed_word_owners)) != len(fixed_word_owners) or
            any(not isinstance(i, (int, np.integer)) or isinstance(i, bool)
                or not 0 <= i < len(whole_words) for i in fixed_word_owners)):
        raise ValueError('Invalid fixed word owner')
    if whole_words:
        extra = []
        for index,group in enumerate(whole_words):
            envelope=np.asarray((word_owner_envelopes or {}).get(index,group))
            if envelope.ndim!=2 or envelope.shape[1]!=2 or not np.isfinite(envelope).all():
                raise ValueError('Invalid word owner envelope')
            x0, y0 = np.minimum(group.min(0),envelope.min(0))
            x1, y1 = np.maximum(group.max(0),envelope.max(0))
            extra.append([[x0,y0],[x1,y0],[x1,y1],[x0,y1],[(x0+x1)/2,y0],[(x0+x1)/2,y1]])
        boxes = np.concatenate([boxes, np.asarray(extra)])
        if measured_ink_bands:
            ink_clouds.extend(pixel_footprint(g,w,h) for g in whole_words)
    centers = boxes.mean(1)
    groups = word_groups(boxes,native_start,reading_axis=word_reading_axis) if independent_word_owners and whole_words else []
    for i in [] if independent_word_owners and whole_words else np.argsort(centers[:, 1]):
        for group in groups:
            if boxes[group, :, 0].min() - 0.06 < centers[i, 0] < boxes[group, :, 0].max() + 0.06 and abs(centers[group, 1].mean() - centers[i, 1]) < 0.004 and (max(np.ptp(boxes[i, :, 1]), np.median(np.ptp(boxes[group, :, 1], axis=1))) < min(np.ptp(boxes[i, :, 1]), np.median(np.ptp(boxes[group, :, 1], axis=1))) * 1.7):
                group.append(i)
                break
        else:
            groups.append([int(i)])
    nr, nc = (max(old.shape[0],minimum_resolution[0]),max(old.shape[1],minimum_resolution[1],w if coordinate_composition else 0))
    if coordinate_composition:
        nr,nc=measurement_aligned_resolution(nr,nc,h,w)
    u, v = np.meshgrid(np.linspace(0, 1, nc), np.linspace(0, 1, nr))
    band_low=band_high=None
    free_components=set();free_component_roles=[]
    if measured_ink_bands:
        band_low,band_high=column_bands(ink_clouds,nc,owner_padding+1/(nc-1),1/(nr-1))
    def ink_bands(col,x):
        if measured_ink_bands:
            return [(int(i),float(band_low[i,col]),float(band_high[i,col]))
                    for i in np.flatnonzero(np.isfinite(band_low[:,col])) if i not in free_components]
        active=np.flatnonzero((boxes[:,:,0].min(1)<=x+owner_padding)&
                              (boxes[:,:,0].max(1)>=x-owner_padding))
        return [(int(i),float(boxes[i,:,1].min())-glyph_pad,
                 float(boxes[i,:,1].max())+glyph_pad) for i in active]
    curve = np.interp(u[0], points[:, 0], points[:, 1])
    destination_curve = np.interp(u[0], target_guide[:, 0], target_guide[:, 1])
    observed_columns = None
    if registered_guides or shared_row is not None:
        curve=registered_curve_samples(points,nc,tolerance=.005*max(h,w)/1280/h)
    if shared_row is not None:
        observed_columns = np.zeros(nc, dtype=bool)
        for guide in guides:
            # Bilinear interpolation needs both mesh knots bracketing each
            # observed endpoint. A retained inverse can place an endpoint a
            # fraction below an exact knot; an interval comparison would then
            # leave that complete material endpoint on the old row.
            first=max(0,int(np.floor(guide[0,0]*(nc-1))))
            last=min(nc-1,int(np.ceil(guide[-1,0]*(nc-1))))
            observed_columns[first:last+1]=True
        # A word gap has no printed ridge. Do not impose the interpolated
        # curve there; its real glyph owners supply the constraints instead.
        destination_curve[~observed_columns] = curve[~observed_columns]
    ox0, ox1 = (0.002, 0.998) if tangent_padding is None else (
        max(.002,float(points[:,0].min())-tangent_padding),
        min(.998,float(points[:,0].max())+tangent_padding))
    oy0, oy1 = (max(0.005, points[:, 1].min() - 0.05), min(0.995, points[:, 1].max() + 0.04))
    if joint_stock_iterations:
        # Only the bend and its adjacent blank stock belong to this repair.
        # Fade the displacement past measured endpoints before pinning ink.
        oy0 = max(0., float(points[:, 1].min()) - .05)
        destination_curve = curve + (destination_curve - curve) * p.window(
            u[0], ox0, points[0, 0], points[-1, 0], ox1)
    frozen_rows = []
    # A visible neighbouring rule needs protection even before it has been
    # selected for correction. These temporary guards do not mark its current
    # bend as an established target for subsequent stages.
    ruler_records = [*base.get('established_rulers', []),
                     *([] if additional_rulers is None else additional_rulers)]
    fixed_rows = [inverse(old, np.asarray(ruler['points']))
                  for i, ruler in enumerate(ruler_records)
                  if ruler['normal_axis'] == 1 and i not in retargeted_rulers]
    fixed_rows += [g for i, g in enumerate(mapped_rules) if i not in primary]
    perpendicular = ([inverse(old, np.asarray(ruler['points']))
                      for ruler in ruler_records if ruler['normal_axis'] == 0]
                     if preserve_neighbouring_rules else [])
    for fixed in fixed_rows:
        fixed = fixed[np.argsort(fixed[:, 0])]
        if np.ptp(fixed[:, 0]) > 0:
            left, right = joined_rule_extent(fixed, perpendicular, w, h)
            frozen_rows.append((left, right,
                                np.interp(u[0], fixed[:, 0], fixed[:, 1])))

    def frozen_bands(col, x):
        bands = []
        pad = 3 * max(h, w) / 1280 / h if align_to_existing else .0008
        for left, right, fixed in frozen_rows:
            if left - 2 / (nc - 1) <= x <= right + 2 / (nc - 1):
                low = max(oy0, fixed[col] - pad)
                high = min(oy1, fixed[col] + pad)
                if low < high:
                    bands.append((low, high))
        return bands
    src = source
    vs = [g[:, ::-1] for g in p.reference.extract_rules(src.T)]
    protected = [g for i, g in enumerate(rules) if i not in primary] + accepted.accepted.circular_landmarks(src) + p.reference.barcode_rulers(src)
    protected += [] if extra_protected is None else list(extra_protected)
    protected += [np.asarray(row['points']) for row in base.get('established_text_rows',[])]
    body_sizes=[len(g) for g in ink_clouds] if measured_ink_bands else [6]*len(boxes)
    body_offsets=np.r_[0,np.cumsum(body_sizes)]
    body_positions=np.vstack(ink_clouds) if measured_ink_bands else boxes.reshape(-1,2)
    # Ordering refusals do not need material inversions of every pixel corner.
    # Initialize this unchanged full proof lazily when a candidate reaches it.
    body=word_sources=baseline_body=baseline_boxes=None
    guard_sources=protected+vs+[np.asarray(r['points']) for r in ruler_records]
    guard_sizes=[len(g) for g in guard_sources]
    guard_points=np.vstack(guard_sources) if guard_sources else np.empty((0,2))
    guard_before=inverse(old,guard_points) if len(guard_points) else guard_points
    guard_offsets=np.r_[0,np.cumsum(guard_sizes)]
    result = base
    if not len(boxes):
        return base
    owner = np.zeros(len(boxes), int)
    glyph_pad = 3 * max(h, w) / 1280 / h if align_to_existing else .0006
    if coordinate_composition:
        # If a whole word reaches into the repair, its complete body must
        # remain one owner. A clipped influence region cannot translate half
        # a word or falsely make the entire intersecting body immovable.
        affected_words=[g for g in whole_words if
                g[:,0].min()<=ox1 and g[:,0].max()>=ox0 and
                g[:,1].min()<oy1 and g[:,1].max()>oy0]
        if affected_words:
            # Retain the normal stock margin beyond the complete owner too.
            # A body barely inside the ROI has no stock in which to absorb
            # its necessary rigid translation. Every newly reached owner and
            # neighbouring rule is still constrained below.
            oy0=max(.005,min(oy0,min(g[:,1].min() for g in affected_words)-max(.05,glyph_pad+1/h)))
            oy1=min(.995,max(oy1,max(g[:,1].max() for g in affected_words)+max(.04,glyph_pad+1/h)))
        # Enlarging stock can reach another owner's halo. Close over those
        # complete bodies too; a newly clipped punctuation mark must not pin
        # an otherwise movable word column through overlapping rigid bands.
        for _ in range(len(boxes)+1):
            affected=boxes[(boxes[:,:,0].min(1)<=ox1)&
                           (boxes[:,:,0].max(1)>=ox0)&
                           (boxes[:,:,1].min(1)-glyph_pad<oy1)&
                           (boxes[:,:,1].max(1)+glyph_pad>oy0)]
            if not len(affected):break
            lo=max(.005,min(oy0,float(affected[:,:,1].min())-glyph_pad-1/h))
            hi=min(.995,max(oy1,float(affected[:,:,1].max())+glyph_pad+1/h))
            if lo==oy0 and hi==oy1:break
            oy0,oy1=lo,hi
    preserved_top=oy0
    if precise_geometry and exact_ordering:
        for i,ruler in enumerate(ruler_records,start=len(protected)+len(vs)):
            trace=guard_before[guard_offsets[i]:guard_offsets[i+1]]
            if (ruler['normal_axis']==1 and np.ptp(trace[:,0])>.6 and
                    oy0<trace[:,1].max()<points[:,1].min()-2*glyph_pad):
                # Keep one additional measurement pixel for the raster edge
                # around the already accepted rule, beyond the glyph pad.
                preserved_top=max(preserved_top,np.ceil((trace[:,1].max()+glyph_pad+1/h)*1024)/1024)
    for gi, group in enumerate(groups):
        owner[group] = gi
    if free_tangent_rules:
        orthogonal=[guard_before[guard_offsets[len(protected)+len(vs)+i]:
                                guard_offsets[len(protected)+len(vs)+i+1]]
                    for i,ruler in enumerate(ruler_records)
                    if i<len(base.get('established_rulers',[])) and ruler['normal_axis']==0]
        free_component_roles=orthogonal_rule_components(groups,ink_clouds,native_start,orthogonal,w,h)
        free_components={i for role in free_component_roles for i in role['component_indices']}
        if audit_log is not None:
            audit_log.append({'tangent_rule_proxy_classification':free_component_roles,
                              'native_word_owners_unchanged':True})
    strengths = ([.3] if joint_stock_iterations else [.3,.4,.5,.6])
    if coordinate_composition:
        strengths=[.2,*strengths]
    refused_geometry_shifts=[]
    for strength in strengths:
        # Solve source ordering before constructing the backward map. Every
        # whole glyph band receives one translation, including faint edges.
        bounds = {}

        def constrain(a, b, lower, upper):
            if a == b:
                feasible = lower <= 1e-09 and upper >= -1e-09
                if not feasible and audit_log is not None:
                    audit_log.append({'rejected': 'fixed ordering gap', 'bounds': [lower, upper], 'owner': a})
                return feasible
            if a > b:
                a, b = (b, a)
                lower, upper = (-upper, -lower)
            key = (a, b)
            previous = bounds.get(key, (-np.inf, np.inf))
            bounds[key] = (max(lower, previous[0]), min(upper, previous[1]))
            feasible = bounds[key][0] <= bounds[key][1] + 1e-09
            if not feasible and audit_log is not None:
                positions = []
                for owner_index in (a, b):
                    positions.append(None if owner_index < 0 else
                                     [boxes[groups[owner_index]].min(axis=(0, 1)).tolist(),
                                      boxes[groups[owner_index]].max(axis=(0, 1)).tolist()])
                audit_log.append({'rejected': 'conflicting word translations',
                                  'bounds': bounds[key], 'positions': positions,
                                  'owner_indices':[int(a),int(b)],
                                  'native_word_indices':[
                                      [] if index<0 else [int(j-native_start)
                                          for j in groups[index] if j>=native_start]
                                      for index in (a,b)],
                                  'component_indices':[
                                      [] if index<0 else [int(j)
                                          for j in groups[index] if j<native_start]
                                      for index in (a,b)]})
            return feasible
        possible = True
        for index in fixed_word_owners:
            possible = constrain(-1, int(owner[native_start + index]), 0, 0) and possible
        for link in word_translation_links or []:
            if (len(link)<2 or len(set(link))!=len(link) or
                    any(not isinstance(i,(int,np.integer)) or not 0<=i<len(whole_words) for i in link)):
                raise ValueError('Invalid word translation relation')
            linked=[int(owner[native_start+i]) for i in link]
            for left,right in zip(linked,linked[1:]):
                possible=constrain(left,right,0,0) and possible
        for gi, group in enumerate(groups):
            if boxes[group, :, 0].min() < ox0 or boxes[group, :, 0].max() > ox1 or boxes[group, :, 1].min() < oy0 or (boxes[group, :, 1].max() > oy1):
                constrain(-1, gi, 0, 0)
            else:
                constrain(-1, gi, -0.02, 0.02)
        for col, x in enumerate(u[0]):
            if not ox0 <= x <= ox1:
                continue
            nodes = [(oy0, oy0, -1, 0.0), (oy1, oy1, -1, 0.0)]
            nodes.extend((low, high, -1, 0.0) for low, high in frozen_bands(col, x))
            for i,low,high in ink_bands(col,x):
                b0 = max(oy0, low)
                b1 = min(oy1, high)
                if b1 > b0:
                    nodes.append((b0, b1, int(owner[i]), 0.0))
            ribbon = 0.0008
            if (observed_columns[col] if observed_columns is not None else points[:, 0].min() - 2 / (nc - 1) <= x <= points[:, 0].max() + 2 / (nc - 1)) and (not any((b0 < curve[col] + ribbon and b1 > curve[col] - ribbon for b0, b1, _, _ in nodes))):
                nodes.append((curve[col] - ribbon, curve[col] + ribbon, -1, destination_curve[col] - curve[col]))
            nodes.sort()
            merged = []
            for b0, b1, oi, ci in nodes:
                if merged and b0 <= merged[-1][1] + 1e-09:
                    p0, p1, pi, pc = merged[-1]
                    if not constrain(pi, oi, pc - ci, pc - ci):
                        possible = False
                        break
                    merged[-1] = (p0, max(p1, b1), pi, pc)
                else:
                    merged.append((b0, b1, oi, ci))
            if not possible:
                break
            for aa, bb in zip(merged, merged[1:]):
                gap = bb[0] - aa[1]
                fixed = bb[3] - aa[3]
                if not constrain(aa[2], bb[2], gap / 3 - gap - fixed, gap / strength - gap - fixed):
                    possible = False
                    break
            if not possible:
                break
        if not possible:
            if audit_log is not None:
                audit_log.append({'strength': strength, 'rejected': 'word ordering'})
            continue
        c = np.zeros(len(groups))
        constraints = [(*key, *bound) for key, bound in bounds.items()]
        if exact_ordering:
            relevance=[]
            for group in groups:
                positions=centers[group]
                observed=np.interp(positions[:,0],points[:,0],points[:,1])
                desired=np.interp(positions[:,0],target_guide[:,0],target_guide[:,1])
                relevance.append(float(np.max(abs(desired-observed)/np.maximum(abs(positions[:,1]-observed),2/h))))
            c=ordered_shifts(bounds,len(groups),np.argsort(relevance))
            if c is None:
                if audit_log is not None:
                    cycle=infeasible_cycle(bounds,len(groups))
                    involved=sorted({q[k] for q in cycle for k in ('from_owner','to_owner') if q[k]>=0})
                    audit_log.append({'strength':strength,'rejected':'infeasible ordered translation cycle',
                        'cycle':cycle,'owner_positions':{str(i):{
                            'bounds':[boxes[groups[i]].min(axis=(0,1)).tolist(),boxes[groups[i]].max(axis=(0,1)).tolist()],
                            'native_words':[int(j-native_start) for j in groups[i] if j>=native_start]}
                            for i in involved}})
                continue
        for iteration in range(0 if exact_ordering else 4000):
            error = 0.0
            for aa, bb, lo, hi in constraints:
                value = (c[bb] if bb >= 0 else 0) - (c[aa] if aa >= 0 else 0)
                wanted = float(np.clip(value, lo, hi))
                diff = wanted - value
                error = max(error, abs(diff))
                count = int(aa >= 0) + int(bb >= 0)
                if aa >= 0:
                    c[aa] -= diff / count
                if bb >= 0:
                    c[bb] += diff / count
            if error < 1e-10:
                break
        violation = max((max(lo - ((c[bb] if bb >= 0 else 0) - (c[aa] if aa >= 0 else 0)), (c[bb] if bb >= 0 else 0) - (c[aa] if aa >= 0 else 0) - hi, 0) for aa, bb, lo, hi in constraints))
        if violation > 1e-08:
            if audit_log is not None:
                audit_log.append({'strength': strength, 'rejected': 'translation convergence', 'violation': violation})
            continue
        shifts = c[owner]
        duplicate=next((previous for previous in refused_geometry_shifts
                        if np.array_equal(previous['shifts'],shifts)),None)
        if duplicate is not None:
            if audit_log is not None:
                audit_log.append({'strength':strength,'rejected':'identical refused geometry',
                                  'checked_strength':duplicate['strength']})
            continue
        max_translation = float(np.max(abs(shifts)) * h)
        field = v.copy()
        column_knots=[None]*nc
        possible = True
        for col, x in enumerate(u[0]):
            if not ox0 <= x <= ox1:
                continue
            ribbon = 0.0008
            fixed_knots = [(oy0, oy0), (oy1, oy1)]
            for low, high in frozen_bands(col, x):
                fixed_knots.extend([(low, low), (high, high)])
            knots = list(fixed_knots)
            impose_ridge = (observed_columns[col] if observed_columns is not None else
                           joint_stock_iterations or points[:, 0].min() - 2 / (nc - 1) <= x <= points[:, 0].max() + 2 / (nc - 1))
            if shared_row is not None and any(low < curve[col] + ribbon and high > curve[col] - ribbon
                                              for low, high in frozen_bands(col, x)):
                impose_ridge = False
            if impose_ridge:
                knots.extend([(destination_curve[col] - ribbon, curve[col] - ribbon), (destination_curve[col] + ribbon, curve[col] + ribbon)])
            intervals = []
            for i,b0,b1 in ink_bands(col,x):
                c = shifts[i]
                if b1 < oy0 or b0 > oy1:
                    continue
                b0 = max(oy0, b0)
                b1 = min(oy1, b1)
                intervals.append((b0 + c, b1 + c, b0, b1))
            merged = []
            for d0, d1, s0, s1 in sorted(intervals):
                if merged and d0 < merged[-1][1] + 1e-09:
                    if abs(d0 - s0 - (merged[-1][0] - merged[-1][2])) > 1e-09:
                        possible = False
                        break
                    merged[-1] = (merged[-1][0], max(d1, merged[-1][1]), merged[-1][2], max(s1, merged[-1][3]))
                else:
                    merged.append((d0, d1, s0, s1))
            if not possible:
                break
            if any((s0 < curve[col] + ribbon and s1 > curve[col] - ribbon for d0, d1, s0, s1 in merged)):
                knots = list(fixed_knots)
            for d0, d1, s0, s1 in merged:
                knots.extend([(d0, s0), (d1, s1)])
            knots = np.array(sorted(set(knots)))
            dest, source = knots.T
            if np.any(np.diff(dest) <= 0) or np.any(np.diff(source) <= 0):
                if audit_log is not None:
                    bad=(np.diff(dest)<=0)|(np.diff(source)<=0)
                    audit_log.append({'rejected':'crossed stock knots','column':float(x),
                        'knots':[knots[i:i+2].tolist() for i in np.flatnonzero(bad)]})
                possible = False
                break
            field[:, col] = np.interp(v[:, col], dest, source)
            column_knots[col]=knots
        if not possible:
            if audit_log is not None:
                audit_log.append({'strength': strength, 'rejected': 'backward ordering'})
            continue
        if stock_relaxation:
            field=relax(field,v[:,0],column_knots,stock_relaxation)
        affected = boxes[abs(shifts) > 1e-10]
        inner0 = min(float(points[:, 0].min()) - 2 / (nc - 1), float(affected[:, :, 0].min()) - 0.004) if len(affected) else float(points[:, 0].min()) - 2 / (nc - 1)
        inner1 = max(float(points[:, 0].max()) + 2 / (nc - 1), float(affected[:, :, 0].max()) + 0.004) if len(affected) else float(points[:, 0].max()) + 2 / (nc - 1)
        weight = p.window(u, ox0, inner0, inner1, ox1) * ((v >= oy0) & (v <= oy1))
        if joint_stock_iterations:
            weight = ((u >= ox0) & (u <= ox1) & (v >= oy0) & (v <= oy1))
        query=np.stack([u,v+(field-v)*weight],axis=2)
        if coordinate_composition:
            try:
                query=allocate_stock(old,query,column_knots,weight,float(target_guide[:,1].mean()),allow_unused_alias=has_coordinate_layer(base),parent=parent,minimum_area=stock_area_reserve,
                                     include_neighbour_span=shared_row is not None)
            except ValueError as error:
                if audit_log is not None:
                    audit_log.append({'strength':strength,'rejected':'stock allocation',
                                      'reason':str(error)})
                continue
            if preserved_top>oy0:
                transition=np.clip((v-preserved_top)/max(glyph_pad,2/(nr-1)),0,1)
                transition=transition*transition*(3-2*transition)
                query[:,:,1]=v+(query[:,:,1]-v)*transition
            if np.max(abs(query[:,:,1]-v))<1e-10:
                if audit_log is not None:audit_log.append({'rejected':'no measured coordinate change'})
                continue
            changed_y,changed_x=np.nonzero(abs(query[:,:,1]-v)>1e-14)
            coordinate_regions=[[max(0,int(changed_x.min())-1)/(nc-1),
                                 max(0,int(changed_y.min())-1)/(nr-1),
                                 min(nc-1,int(changed_x.max())+1)/(nc-1),
                                 min(nr-1,int(changed_y.max())+1)/(nr-1)]]
            try:
                proof=validate_normal_composition(parent,query,coordinate_regions)
            except ValueError as error:
                if 'Nested coordinate layers' not in str(error):raise
                from retained_composition_guard import validate_retained_composition
                proof=validate_retained_composition(parent,query,coordinate_regions)
            if not proof['accepted']:
                if audit_log is not None:audit_log.append({'rejected':'continuous retained-map area','proof':proof})
                refused_geometry_shifts.append({'strength':strength,'shifts':shifts.copy()})
                continue
            candidate=query
            candidate_sampler=MeshSampler({'rows':nr,'columns':nc,'grid':query.reshape(-1,2),'coordinate_transform':True,'preserved_mesh':base,'refinement_regions':coordinate_regions})
        else:
            candidate=p.sample(old,query.reshape(-1,2)).reshape(nr,nc,2)
        if joint_stock_iterations:
            desired = np.c_[points[:, 0], np.interp(
                points[:, 0], target_guide[:, 0], target_guide[:, 1])]
            axes = [None] * len(protected) + [0] * len(vs) + [
                r['normal_axis'] for r in ruler_records]
            pin_positions, pin_sources = crossing_pins(
                candidate, desired, source_rulers, guard_before,
                guard_points, guard_sizes, axes)
            candidate = project_joint_stock(
                candidate, boxes, shifts, pin_positions, pin_sources,
                (ox0, oy0, ox1, oy1), joint_stock_iterations, audit_log)
        elif stock_coordinate_relaxation:
            actual_field=v+(field-v)*weight
            guarded_traces=[guard_before[lo:hi] for lo,hi in zip(guard_offsets,guard_offsets[1:])]
            guarded_traces.append(points)
            candidate=relax_tangent_stock(candidate,actual_field,boxes,shifts,
                guarded_traces,(ox0,oy0,ox1,oy1),stock_coordinate_relaxation)
        if preserved_top>oy0 and not coordinate_composition:
            prior=p.sample(old,np.c_[u.ravel(),v.ravel()]).reshape(nr,nc,2)
            transition=np.clip((v-preserved_top)/max(glyph_pad,2/(nr-1)),0,1)
            transition=transition*transition*(3-2*transition)
            candidate=prior+(candidate-prior)*transition[:,:,None]
        if not p.valid(candidate):
            if audit_log is not None:
                dx,dy=np.diff(candidate,axis=1),np.diff(candidate,axis=0)
                areas=np.asarray([(a[:,:,0]*b[:,:,1]-a[:,:,1]*b[:,:,0])*(nr-1)*(nc-1)
                    for a in (dx[:-1],dx[1:]) for b in (dy[:,:-1],dy[:,1:])])
                corner,row,col=np.unravel_index(areas.argmin(),areas.shape)
                audit_log.append({'strength': strength, 'rejected': 'mesh topology',
                    'minimum_cell_area':float(areas.min()),
                    'worst_cell':[float(u[row,col]),float(v[row,col])],
                    'normal_field_scale':float((field[row+1,col]-field[row,col])*(nr-1)),
                    'tangent_field_scale':float((field[row,col+1]-field[row,col])*(nc-1))})
            continue
        after = inverse(candidate, source_rulers)
        target_y = np.interp(points[:, 0], target_guide[:, 0], target_guide[:, 1])
        error = float(np.max(abs(after[:, 1] - target_y)) * h)
        worst_index = int(np.argmax(abs(after[:, 1] - target_y)))
        if body is None:
            body=sample_old(body_positions)
            baseline_body=inverse(old,body)
            baseline_boxes=None if measured_ink_bands else baseline_body.reshape(-1,6,2)
            word_sources=sample_old(np.vstack(whole_words)) if whole_words else None
        new_body=inverse(candidate,body)
        if measured_ink_bands:
            delta=new_body-baseline_body
            parts=[delta[a:b] for a,b in zip(body_offsets,body_offsets[1:])]
            height=max(float(np.ptp(q[:,1]))*h for i,q in enumerate(parts) if i not in free_components)
            strict_groups=[[i for i in group if i not in free_components] for group in groups]
            rigidity=max(float(np.ptp(np.vstack([parts[i] for i in group])[:,1]))*h
                         for group in strict_groups if group)
            xshift=float(abs(delta[:,0]).max())*w
        else:
            new_boxes = new_body.reshape(-1, 6, 2)
            delta = new_boxes - baseline_boxes
            height = float(np.max(abs(new_boxes[:, 2, 1] - new_boxes[:, 1, 1] - (baseline_boxes[:, 2, 1] - baseline_boxes[:, 1, 1]))) * h)
            rigidity = max((float(np.max(np.ptp(delta[group, :, 1], axis=None))) * h for group in groups))
            xshift = float(np.max(abs(delta[:, :, 0])) * w)
        guard_delta=(inverse(candidate,guard_points)-guard_before)*[w,h] if len(guard_points) else guard_points
        protected_end=int(guard_offsets[len(protected)])
        vertical_end=int(guard_offsets[len(protected)+len(vs)])
        other=float(abs(guard_delta[:protected_end]).max()) if protected_end else 0.
        normal=float(abs(guard_delta[protected_end:vertical_end,0]).max()) if vertical_end>protected_end else 0.
        established_shift = 0.0
        retargeted_checks = []
        for i,ruler in enumerate(ruler_records,start=len(protected)+len(vs)):
            axis = ruler['normal_axis']
            rule_index = i - len(protected) - len(vs)
            if rule_index in retargeted_rulers:
                before = guard_before[guard_offsets[i]:guard_offsets[i+1], axis] * h
                actual = before + guard_delta[guard_offsets[i]:guard_offsets[i+1], axis]
                before_range, after_range = float(np.ptp(before)), float(np.ptp(actual))
                target_distance = float(abs(np.median(actual) - shared_row['target'] * h))
                retargeted_checks.append({'index':rule_index,
                    'before_normal_range_px':before_range,
                    'after_normal_range_px':after_range,
                    'target_distance_px':target_distance,
                    'accepted':bool(after_range <= before_range + .05 and target_distance < 2)})
                continue
            shift=abs(guard_delta[guard_offsets[i]:guard_offsets[i+1],axis])
            established_shift = max(established_shift, float(shift.max()))
        whole_word_rigidity = 0.0
        if whole_words:
            delta_words = inverse(candidate, word_sources) - np.vstack(whole_words)
            offset = 0
            for group in whole_words:
                part = delta_words[offset:offset+len(group)] * [w,h]
                offset += len(group)
                whole_word_rigidity = max(whole_word_rigidity, float(np.ptp(part,axis=0).max()))
        rule_tolerance = .03 if coordinate_composition else .6
        take = error < rule_tolerance * max(h, w) / 1280 and height < 0.1 and (rigidity < 0.1) and (xshift < 0.1) and (other < 0.1) and (normal < 0.1) and established_shift < 0.1 and whole_word_rigidity < .1 and all(q['accepted'] for q in retargeted_checks)
        audit = {'accepted': take, 'strength': strength, 'row_target': target,
                 'measured_ink_bands':bool(measured_ink_bands),
                 'stock_area_reserve':float(stock_area_reserve),
                 'stock_neighbour_span':shared_row is not None,
                 'tangent_free_rule_components':[int(i) for i in sorted(free_components)],
                 'tangent_free_rule_normal_shift_px':max((float(abs(delta[body_offsets[i]:body_offsets[i+1],0]).max())*w
                                                         for i in free_components),default=0.),
                 'fixed_word_owners': [int(i) for i in fixed_word_owners],
                 'destination_row_target':float(target_guide[:,1].mean()),
                 'matched_primary_rules': len(primary), 'full_rule_error_px': error,
                 'retargeted_rule_checks':retargeted_checks,
                 'worst_rule_point': points[worst_index].tolist(),
                 'glyph_height_change_px': height, 'row_rigidity_change_px': rigidity,
                 'maximum_word_translation_px': max_translation,
                 'whole_word_ink_rigidity_px': whole_word_rigidity,
                 'glyph_x_shift_px': xshift, 'other_print_shift_px': other,
                 'crossed_normal_shift_px': normal, 'established_normal_shift_px': established_shift}
        desired=np.c_[points[:,0],target_y]
        forward_error=abs((p.sample(candidate,desired) if candidate_sampler is None else candidate_sampler.sample(desired))-source_rulers)*[w,h]
        audit['desired_rule_forward_error_px']=forward_error.max(0).tolist()
        audit['worst_forward_rule_point']=points[int(np.argmax(forward_error.max(1)))].tolist()
        if not take and error >= .6 * max(h,w) / 1280:
            wx,wy=points[worst_index]
            blocked=(boxes[:,:,0].min(1)<=wx+owner_padding)&(boxes[:,:,0].max(1)>=wx-owner_padding)&(
                boxes[:,:,1].min(1)-glyph_pad<wy+.0008)&(
                boxes[:,:,1].max(1)+glyph_pad>wy-.0008)
            audit['rule_overlapping_ink_envelopes']=[
                [q.min(0).tolist(),q.max(0).tolist()] for q in boxes[blocked]]
            col=int(np.argmin(abs(u[0]-wx)))
            audit['rule_overlapping_frozen_bands']=[
                [lo,hi] for lo,hi in frozen_bands(col,wx)
                if lo<wy+.0008 and hi>wy-.0008]
        if not take and other >= .1 and protected_end:
            point,axis=np.unravel_index(abs(guard_delta[:protected_end]).argmax(),
                                       guard_delta[:protected_end].shape)
            entity=int(np.searchsorted(guard_offsets,point,side='right')-1)
            audit['blocking_protected_entity']={'index':entity,'axis':int(axis),
                'before':guard_before[point].tolist(),
                'delta_px':guard_delta[point].tolist(),
                'bounds':[guard_before[guard_offsets[entity]:guard_offsets[entity+1]].min(0).tolist(),
                          guard_before[guard_offsets[entity]:guard_offsets[entity+1]].max(0).tolist()]}
        if audit_log is not None:
            audit_log.append(audit)
        if not take:
            refused_geometry_shifts.append({'strength':strength,'shifts':shifts.copy()})
        if take:
            result = {**base, 'preserved_mesh': base, 'refinement_regions': [[np.floor(ox0 * 512) / 512, np.floor(preserved_top * 1024) / 1024, np.ceil(ox1 * 512) / 512, np.ceil(oy1 * 1024) / 1024]], 'rows': nr, 'columns': nc, 'grid': candidate.reshape(-1, 2).tolist(), 'method': 'Experimental supported printed-row correction', 'joint_row_audit': audit}
            result['established_rulers'] = [*base.get('established_rulers', []),
                                            {'normal_axis': 1, 'points': source_rulers.tolist()}]
            if coordinate_composition:
                result['coordinate_transform']=True
                result['continuous_jacobian_audit']=proof
                result['refinement_regions']=coordinate_regions
                result['method']='Supported printed-row correction through retained coordinate maps'
                if shared_row is not None:
                    result['method']='Supported shared word and rule row through retained coordinate maps'
                    result['word_flanked_rule_row_audit']={
                        **{k:v for k,v in shared_row.items() if k!='guides'},
                        'guides':[g.tolist() for g in shared_row['guides']],
                        'retargeted_rules':[{'index':i,
                            'source_points':np.asarray(base['established_rulers'][i]['points']).tolist(),
                            'reference_points':inverse(old,np.asarray(base['established_rulers'][i]['points'])).tolist()}
                            for i in sorted(retargeted_rulers)]}
                result.pop('grid_window',None)
                if free_component_roles:
                    records=[]
                    for role in free_component_roles:
                        indices=np.concatenate([np.arange(body_offsets[i],body_offsets[i+1])
                                                for i in role['component_indices']])
                        records.append({**role,'points':body_positions[indices].tolist(),
                                        'source_points':body[indices].tolist()})
                    result['tangent_free_rule_proxy_roles']={
                        'roles':records,'input_basis':'Current component summaries; native owners unchanged',
                        'earlier_roles':base.get('tangent_free_rule_proxy_roles')}
            break
    return result

def solve(path, working=None):
    original = accepted.solve(path, working)
    try:
        return refine(path, original)
    except (OSError, ValueError, ArithmeticError, AttributeError, IndexError, KeyError, TypeError, cv2.error):
        return original
if __name__ == '__main__':
    result = solve(sys.argv[1], sys.argv[3] if len(sys.argv) > 3 else None)
    Path(sys.argv[2]).write_text(json.dumps(result, allow_nan=False))
