"""Audit the accepted content map and correct supported residuals anywhere.

Every proposal is local, preserves the prior map outside its measured region,
and must improve measured ink while protecting other content and the perimeter.
"""
from pathlib import Path
import json
import sys
import cv2
import numpy as np
import local_receipt_mesh as prior
from mesh_ruler import circular_landmarks
from content_column_rulers import correct_columns


def character_anchors(gray, guide):
    h, w = gray.shape
    scale = max(h, w)/1280
    pad = max(12, round(30*scale))
    lo = max(0, int(guide[:, 1].min()*h)-pad)
    hi = min(h, int(guide[:, 1].max()*h)+pad)
    crop = gray[lo:hi].astype(float)
    ink = cv2.GaussianBlur(crop, (0, 0), max(2, 7*scale))-crop
    binary = (ink > 10).astype(np.uint8)*255
    _, labels, stats, centers = cv2.connectedComponentsWithStats(binary)
    points, heights = [], []
    for index, (s, center) in enumerate(zip(stats, centers)):
        x, y, cw, ch, area = s
        if not (max(3, 7*scale) < ch < max(12, 35*scale) and
                max(2, 3*scale) < cw < w*.09 and area > max(8, 20*scale*scale) and
                guide[0, 0]-.015 < center[0]/(w-1) < guide[-1, 0]+.015):
            continue
        expected = np.interp(center[0]/(w-1), guide[:, 0], guide[:, 1])*(h-1)
        if abs(y+lo+ch-1-expected) > max(5, 15*scale):
            continue
        component = labels[y:y+ch, x:x+cw] == index
        bottoms = np.array([np.where(component[:, xx])[0][-1]
                            for xx in range(cw) if component[:, xx].any()])
        points.append([center[0]/(w-1), (np.quantile(bottoms, .85)+y+lo)/(h-1)])
        heights.append(ch/(h-1))
    if len(points) < 12:
        return None
    # Accents, ascenders and descenders must not become a false bend. Use the
    # repeated body-height population; abstain when too few letters remain.
    heights = np.array(heights)
    keep = heights <= np.median(heights)*1.15
    if keep.sum() < 12:
        return None
    points = np.array(sorted(np.array(points)[keep].tolist()))
    if np.ptp(points[:, 0]) < .3:
        return None
    return points, float(np.median(heights[keep]))


def residual_evidence(mapped, pixel_height):
    # Font contours vary by letter. Require a coherent changing baseline to
    # explain most observed variation before attempting a local correction.
    x, y = mapped.T
    coefficients = np.polyfit(x, y, 5)
    for _ in range(4):
        errors = np.abs(np.polyval(coefficients, x)-y)*pixel_height
        coefficients = np.polyfit(x, y, 5, w=np.minimum(1, 1.5/(errors+1e-9)))
    fitted = np.polyval(coefficients, x)
    scatter = float(np.mean((fitted-y)**2))
    variance = float(np.var(y))
    explained = max(0, 1-scatter/max(variance, 1e-16))
    excursion = float(np.ptp(fitted)*pixel_height)
    qualifies = excursion > 2 and explained > .6
    return qualifies, {'excursion_px':excursion, 'explained':explained,
                       'scatter_px':float(np.sqrt(scatter)*pixel_height)}


def grid_of(result):
    return np.array(result['grid']).reshape(result['rows'], result['columns'], 2)


def compact_mesh(result):
    return {key:result[key] for key in ('rows', 'columns', 'grid', 'method')}


def canonical_text_rows(texts, grid):
    """Join nearby fragments of the same ink row; deduplicate overlapping traces."""
    canonical=[]
    for guide in sorted(texts,key=lambda p:np.mean(prior.inverse(grid,p)[:,1])):
        q=prior.inverse(grid,guide)
        for index,previous in enumerate(canonical):
            p=prior.inverse(grid,previous)
            if abs(q[:,1].mean()-p[:,1].mean())>=.004:
                continue
            overlap=min(q[:,0].max(),p[:,0].max())-max(q[:,0].min(),p[:,0].min())
            if overlap>min(np.ptp(q[:,0]),np.ptp(p[:,0]))*.5:
                if np.ptp(guide[:,0])>np.ptp(previous[:,0]):canonical[index]=guide
                break
            if -.07<overlap<0:
                left,right=(previous,guide) if previous[-1,0]<guide[0,0] else (guide,previous)
                if left[-1,0]<right[0,0] and abs(left[-1,1]-right[0,1])<.005:
                    canonical[index]=np.vstack([left,right])
                    break
        else:
            canonical.append(guide)
    return canonical


def orthogonal_guards(gray, mask):
    """Protect sideways text, rules and barcode geometry in the same plane."""
    rotated = cv2.rotate(gray, cv2.ROTATE_90_CLOCKWISE)
    rotated_mask = cv2.rotate(mask, cv2.ROTATE_90_CLOCKWISE)
    features = prior.reference.base.extract_text_features(rotated, mask=rotated_mask)
    texts = prior.reference.text_rulers(rotated, features)
    rules = prior.reference.extract_rules(rotated)
    texts += prior.reference.supplemental_text(rotated,texts+rules)
    bars = prior.reference.barcode_rulers(rotated)
    envelopes = prior.reference.measured_envelopes(rotated,bars)
    guards,structures = [],[]
    for group in texts+rules+bars+envelopes:
        # Dense sampling protects short intersections between opposite axes.
        distance = np.r_[0,np.cumsum(np.linalg.norm(np.diff(group,axis=0),axis=1))]
        samples = np.linspace(0,distance[-1],max(len(group),257))
        p = np.column_stack([np.interp(samples,distance,group[:,axis]) for axis in (0,1)])
        guards.append(np.column_stack([p[:,1],1-p[:,0]]))
        if any(group is p for p in rules+bars+envelopes):
            structures.append(guards[-1])
    for line in features.high_confidence_lines:
        if len(line.centers)>=6 and line.width>rotated.shape[1]*.15:
            p=line.centers/[rotated.shape[1]-1,rotated.shape[0]-1]
            guards.append(np.column_stack([p[:,1],1-p[:,0]]))
    return guards,structures, {'text_rows':len(texts),'rules':len(rules),'barcode_bars':len(bars)}


def joint_text_proposals(grid, measured, texts, rules, bars, envelopes, circles,
                         orthogonal, width, height):
    """Treat independently supported neighbouring bends as one surface patch."""
    records = []
    for guide, data in measured:
        if data is None:
            records.append((float(np.mean(prior.inverse(grid,guide)[:,1])),guide,None))
            continue
        mapped = prior.inverse(grid,data[0])
        qualifies,evidence = residual_evidence(mapped,height)
        records.append((float(np.mean(mapped[:,1])),guide,
                        (data,mapped,evidence) if qualifies else None))
    records.sort(key=lambda row:row[0])
    groups,current = [],[]
    for row in records:
        spacing=(float(np.median(np.diff([p[0] for p in current])))
                 if len(current)>1 else .035/1.5)
        gap_limit=min(.035,max(.02,spacing*1.5))
        if row[2] is None or (current and row[0]-current[-1][0]>gap_limit):
            if len(current)>=3:groups.append(current)
            current=[]
        if row[2] is not None:current.append(row)
    if len(current)>=3:groups.append(current)
    proposals,regions = [],[]
    for group in groups:
        mapped = [row[2][1] for row in group]
        common_lo=max(p[:,0].min() for p in mapped)
        common_hi=min(p[:,0].max() for p in mapped)
        if common_hi-common_lo<.25:continue
        x=np.linspace(common_lo,common_hi,65)
        curves=[np.polyval(np.polyfit(p[:,0],p[:,1]-p[:,1].mean(),5),x) for p in mapped]
        coherent=all(np.corrcoef(a,b)[0,1]>.8 for a,b in zip(curves,curves[1:]))
        if not coherent:continue
        rows,columns=grid.shape[:2]
        nr=max(rows,513);nc=max(columns,513)
        if (nr-1)%(rows-1):nr=(rows-1)*4+1
        if (nc-1)%(columns-1):nc=(columns-1)*4+1
        u,v=np.meshgrid(np.linspace(0,1,nc),np.linspace(0,1,nr))
        dense=prior.sample(grid,np.column_stack([u.ravel(),v.ravel()])).reshape(nr,nc,2)
        targets=np.array([row[0] for row in group]);shifts=[]
        for target,row in zip(targets,group):
            anchors=row[2][0][0]
            along=prior.sample(grid,np.column_stack([u[0],np.full(nc,target)]))
            shifts.append(np.interp(along[:,0],anchors[:,0],anchors[:,1])-along[:,1])
        shifts=np.array(shifts)
        glyph_height=max(row[2][0][1] for row in group)
        x0=min(p[:,0].min() for p in mapped);x1=max(p[:,0].max() for p in mapped)
        inner_lo=targets[0]-glyph_height*1.25
        inner_hi=targets[-1]+glyph_height*.35
        outer_lo=max(.005,inner_lo-glyph_height*.6)
        outer_hi=min(.995,inner_hi+glyph_height*.8)
        above=[row for row in records if row[0]<targets[0]-.001]
        below=[row for row in records if row[0]>targets[-1]+.001]
        if above:
            previous=above[-1];size=(previous[2][0][1] if previous[2] is not None else glyph_height)
            outer_lo=max(outer_lo,previous[0]+size*.5+.0005)
        if below:
            following=below[0];size=(following[2][0][1] if following[2] is not None else glyph_height)
            outer_hi=min(outer_hi,following[0]-size*1.5-.0005)
        if not outer_lo<inner_lo<inner_hi<outer_hi:continue
        outer_x0=max(.005,x0-.02);outer_x1=min(.995,x1+.02)
        # Interpolate an ordered backward surface between fixed patch edges
        # and measured row baselines. A narrow displacement fade can fold a
        # mesh even though the text constraints themselves are ordered.
        top=prior.sample(grid,np.column_stack([u[0],np.full(nc,outer_lo)]))[:,1]
        bottom=prior.sample(grid,np.column_stack([u[0],np.full(nc,outer_hi)]))[:,1]
        source_rows=np.vstack([top,shifts+np.array([
            prior.sample(grid,np.column_stack([u[0],np.full(nc,t)]))[:,1]
            for t in targets]),bottom])
        target_rows=np.r_[outer_lo,targets,outer_hi]
        proposed=np.column_stack([np.interp(v[:,0],target_rows,source_rows[:,col]) for col in range(nc)])
        weight=(prior.window(u,outer_x0,x0-.006,x1+.006,outer_x1)*
                ((v>=outer_lo)&(v<=outer_hi)))
        candidate=dense.copy();candidate[:,:,1]+=(proposed-dense[:,:,1])*weight
        group_ids={id(row[1]) for row in group}
        protected_groups=rules+bars+envelopes+circles+orthogonal
        protected_groups += [p for p in texts if id(p) not in group_ids]
        for guide,data in measured:
            if id(guide) not in group_ids and data is not None:
                protected_groups += [data[0],data[0]+[0,-data[1]*1.5],data[0]+[0,data[1]*.5]]
        protected=np.vstack(protected_groups) if protected_groups else np.empty((0,2))
        movement=float(np.max(np.abs((prior.inverse(candidate,protected)-
                                      prior.inverse(grid,protected))*[width,height]))) if len(protected) else 0.
        before=np.array([np.std(p[:,1])*height for p in mapped])
        after=np.array([np.std(prior.inverse(candidate,row[2][0][0])[:,1])*height for row in group])
        topology=prior.valid(candidate)
        accepted=bool(topology and movement<.1 and
                      np.all(after<np.minimum(.3,before*.25)))
        proposals.append({'kind':'coherent_text_group','rows':len(group),
                          'position':[float((x0+x1)/2),float(targets.mean())],
                          'accepted':accepted,'topology_valid':topology,'before_max_px':float(before.max()),
                          'after_max_px':float(after.max()),'protected_shift_px':movement})
        if accepted:
            grid=candidate
            regions.append([float(np.floor(outer_x0*(nc-1))/(nc-1)),
                            float(np.floor(outer_lo*(nr-1))/(nr-1)),
                            float(np.ceil(outer_x1*(nc-1))/(nc-1)),
                            float(np.ceil(outer_hi*(nr-1))/(nr-1))])
    return grid,proposals,regions


def refine(path, original):
    rgba = cv2.imread(str(path), -1)
    h, w = rgba.shape[:2]
    mask = rgba[:, :, 3] if rgba.shape[2] == 4 else np.full((h, w), 255, np.uint8)
    alpha = mask[:, :, None]/255
    gray = cv2.cvtColor((rgba[:, :, :3]*alpha+255*(1-alpha)).astype(np.uint8), cv2.COLOR_BGR2GRAY)
    features = prior.reference.base.extract_text_features(gray, mask=mask)
    texts = prior.reference.text_rulers(gray, features)
    rules = prior.reference.extract_rules(gray)
    texts += prior.reference.supplemental_text(gray, texts+rules)
    bars = prior.reference.barcode_rulers(gray)
    envelopes = prior.reference.measured_envelopes(gray, bars)
    # Shape landmarks remain constraints, not targets for forced straightening.
    circles = circular_landmarks(gray)
    orthogonal,orthogonal_structures,orthogonal_audit = orthogonal_guards(gray, mask)
    grid = grid_of(original)
    # Different ink passes can describe the same physical row. They are not
    # independent constraints, and must not block each other's correction.
    texts=canonical_text_rows(texts,grid)
    measured = [(guide, character_anchors(gray, guide)) for guide in texts]
    grid,proposals,regions = joint_text_proposals(
        grid,measured,texts,rules,bars,envelopes,circles,orthogonal,w,h)
    audit = {'text_rows':len(texts), 'printed_rules':len(rules),
             'barcode_bars':len(bars), 'shape_landmarks':len(circles),
             'orthogonal_content':orthogonal_audit, 'proposals':proposals}
    audit['structure_residuals'] = {
        'rules_px':[float(np.std(prior.inverse(grid,p)[:,1])*h) for p in rules],
        'barcode_bars_px':[float(np.std(prior.inverse(grid,p)[:,0])*w) for p in bars],
        'barcode_envelopes_px':[float(np.std(prior.inverse(grid,p)[:,1])*h) for p in envelopes],
    }
    # The same bounded proposal works for a traced printed rule. Its observed
    # ridge supplies anchors; no page-position or receipt-name condition exists.
    targets = [('text',guide,data) for guide,data in measured]
    targets += [('printed_rule',guide,(guide,max(2/h,.003))) for guide in rules]
    for kind, guide, measurement in targets:
        if measurement is None:
            continue
        anchors, letter_height = measurement
        mapped = prior.inverse(grid, anchors)
        qualifies, evidence = residual_evidence(mapped, h)
        if not qualifies:
            continue
        entry = {'kind':kind, 'position':np.mean(mapped, axis=0).tolist(), **evidence}
        proposals.append(entry)
        # Subdivide existing cells with aligned vertices. Outside the patch
        # Dart evaluates the exact preserved map rather than reinterpolating it.
        rows, columns = grid.shape[:2]
        target_rows = max(rows,129)
        if (target_rows-1) % (rows-1):
            target_rows = (rows-1)*4+1
        target_columns = max(columns, 513)
        if (target_columns-1) % (columns-1):
            target_columns = (columns-1)*4+1
        u, v = np.meshgrid(np.linspace(0, 1, target_columns), np.linspace(0, 1, target_rows))
        dense = prior.sample(grid, np.column_stack([u.ravel(),v.ravel()])).reshape(target_rows,target_columns,2)
        target = float(np.mean(mapped[:, 1]))
        along = prior.sample(grid, np.column_stack([u[0], np.full(target_columns, target)]))
        shift = np.interp(along[:, 0], anchors[:, 0], anchors[:, 1])-along[:, 1]
        x0, y0 = mapped.min(axis=0); x1, y1 = mapped.max(axis=0)
        # The plateau covers the full glyph height, so each letter keeps its
        # vertical shape. Compact support excludes adjacent established rows.
        inner_lo = y0-letter_height*1.25
        inner_hi = y1+letter_height*.3
        outer_lo = max(.005, inner_lo-letter_height*.6)
        outer_hi = min(.995, inner_hi+letter_height*.8)
        outer_x0, outer_x1 = max(.005,x0-.02), min(.995,x1+.02)
        weight = (prior.window(u,outer_x0,x0-.006,x1+.006,outer_x1) *
                  prior.window(v,outer_lo,inner_lo,inner_hi,outer_hi))
        candidate = dense.copy()
        candidate[:, :, 1] += shift[None, :]*weight
        protected_groups = [p for p in rules+bars+envelopes+circles+texts if p is not guide]+orthogonal
        protected_groups += [data[0] for other,data in measured if other is not guide and data is not None]
        for other,data in measured:
            if other is not guide and data is not None:
                # Preserve upper letter bodies and descenders as well as the
                # baselines; a stationary baseline alone can hide glyph bends.
                protected_groups += [data[0]+[0,-data[1]*1.5],
                                     data[0]+[0,data[1]*.5]]
        protected = np.vstack(protected_groups) if protected_groups else np.empty((0,2))
        movement = (float(np.max(np.abs((prior.inverse(candidate, protected)-
                                        prior.inverse(grid, protected))*[w,h])))
                    if len(protected) else 0.)
        after = prior.inverse(candidate, anchors)
        before_error, after_error = np.std(mapped[:,1])*h, np.std(after[:,1])*h
        accepted = bool(prior.valid(candidate) and movement < .1 and
                        after_error < min(.3, before_error*.25))
        entry.update(accepted=accepted, before_px=float(before_error),
                     after_px=float(after_error), protected_shift_px=movement)
        if accepted:
            grid = candidate
            regions.append([float(np.floor(outer_x0*(target_columns-1))/(target_columns-1)),
                            float(np.floor(outer_lo*(target_rows-1))/(target_rows-1)),
                            float(np.ceil(outer_x1*(target_columns-1))/(target_columns-1)),
                            float(np.ceil(outer_hi*(target_rows-1))/(target_rows-1))])
    grid,column_proposals,column_regions=correct_columns(
        grid,gray,texts,rules,bars,envelopes,circles,orthogonal_structures)
    proposals.extend(column_proposals);regions.extend(column_regions)
    original['residual_audit'] = audit
    if regions:
        preserved = compact_mesh(original)
        original.update(rows=grid.shape[0],columns=grid.shape[1],grid=grid.reshape(-1,2).tolist(),
                        method='Protected content residual correction',
                        preserved_mesh=preserved, refinement_regions=regions)
    return original


def solve(path, working=None):
    original = prior.solve(path, working)
    try:
        return refine(path, original)
    except (OSError, ValueError, ArithmeticError, AttributeError,
            IndexError, KeyError, TypeError, cv2.error) as error:
        original['residual_audit'] = {'accepted':False,'error':type(error).__name__}
        return original


if __name__ == '__main__':
    result = solve(sys.argv[1], sys.argv[3] if len(sys.argv)>3 else None)
    Path(sys.argv[2]).write_text(json.dumps(result, allow_nan=False))
