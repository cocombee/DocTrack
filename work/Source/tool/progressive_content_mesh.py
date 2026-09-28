"""Evidence-driven local stages composed on the accepted DocTrack mesh.

This optional research path uses the app's native word bridge. It reads no
fixture geometry and accepts no image-index or page-position exceptions.
Every completed stage is published atomically; a failed addition keeps the
last checkpoint and its exact parent maps.
"""
import sys
import time
from pathlib import Path

import cv2
import numpy as np

import section_row_candidate as rows
import shared_row_candidate as shared
import frame_section_candidate as frame
import normal_row_candidate as normals
import rigid_word_row as word_rows
from printed_rule_sections import measure_sections, word_ink_mask
from native_word_evidence import request_words,ink_groups,ink_observations,recover_faint_components,measurement_words,request_words_with_crops
from local_mesh_storage import compact, publish
from precise_mesh_inverse import inverse
from composite_mesh_sampling import MeshSampler,transpose_mesh
from composite_mesh_inverse import inverse_near,inverse_retained
from rule_join_evidence import extend_to_crossings,measured_straight_anchor,measured_chains,continued_thin_sections
from normal_composition_guard import validate_source_composition
from isolated_rule_bends import before_stable_tail,anchored_remainder
from text_column_alignment import aligned_columns,independent_owner_links,alignment_audit
from word_rule_partition import partition_rule_fragments,partition_bracketed_rules,partition_supported_short_sections
from word_component_ownership import partition_shared_ink

ERRORS = (OSError, ValueError, ArithmeticError, AttributeError, IndexError,
          KeyError, TypeError, cv2.error)
OPTIONS = dict(separate_structural_ink=True, exact_ordering=True,
               independent_word_owners=True, stock_relaxation=12,
               precise_geometry=True,preserve_neighbouring_rules=True)


def supported_word_mask(gray, groups=None, *, precise_ink=False):
    words = word_ink_mask(gray)
    if groups:
        h, w = gray.shape
        if precise_ink:
            ink=np.zeros_like(words)
            for group in groups:
                pixels=np.rint(np.asarray(group)*[w-1,h-1]).astype(int)
                valid=((pixels[:,0]>=0)&(pixels[:,0]<w)&
                       (pixels[:,1]>=0)&(pixels[:,1]<h))
                pixels=pixels[valid]
                ink[pixels[:,1],pixels[:,0]]=1
            # Tracking excludes actual faint ink and its localization halo.
            # Complete word bodies are still rigid owners in the solver.
            return words|cv2.dilate(ink,np.ones((5,5),np.uint8))
        for group in groups:
            low = np.floor(group.min(0) * [w-1, h-1]).astype(int) - 1
            high = np.ceil(group.max(0) * [w-1, h-1]).astype(int) + 1
            words[max(0, low[1]):min(h, high[1]+1),
                  max(0, low[0]):min(w, high[0]+1)] = 1
    return words


def sections(gray, safe, groups=None, *, verified_crossings=False,
             repeated_dash_rules=False,audit_log=None,whole_rulings=False,
             reference_guides=None):
    if whole_rulings and not repeated_dash_rules:
        raise ValueError('Whole dashed rulings require repeated raster evidence')
    words=supported_word_mask(gray,groups,precise_ink=verified_crossings)
    horizontal = measure_sections(gray, safe, word_mask=words,
                                  word_safe_ridge=verified_crossings)
    h, w = gray.shape
    if repeated_dash_rules:
        from repeated_rule_mask import recover
        protected=np.zeros_like(gray,np.uint8)
        for group in [] if groups is None else groups:
            points=np.rint(np.asarray(group)*[w-1,h-1]).astype(int)
            good=((points>=0)&(points<[w,h])).all(1)
            points=points[good];protected[points[:,1],points[:,0]]=1
        actual_bodies=cv2.dilate(protected,np.ones((3,3),np.uint8))
        protected=cv2.dilate(protected,np.ones((5,5),np.uint8))
        seeds=horizontal+([] if reference_guides is None else reference_guides)
        words,evidence=recover(gray,safe,words,protected,seeds,
                               whole_rulings=whole_rulings)
        if audit_log is not None:audit_log.extend(evidence)
        horizontal=measure_sections(gray,safe,word_mask=words,
                                    word_safe_ridge=verified_crossings)
        if whole_rulings:
            from repeated_rule_mask import whole_ruling_guides
            horizontal,joined=whole_ruling_guides(gray,safe,protected,horizontal,evidence)
            if audit_log is not None:audit_log.extend(joined)
            if reference_guides is not None:
                from repeated_rule_mask import restore_measured_ruling_tails
                horizontal,tails=restore_measured_ruling_tails(
                    gray,safe,actual_bodies,horizontal,reference_guides)
                if audit_log is not None:audit_log.extend(tails)
    crossings = np.zeros_like(gray)
    for guide in horizontal:
        cv2.polylines(crossings,
                      [np.rint(guide * [w-1, h-1]).astype(np.int32)], False, 1, 9)
    vertical = measure_sections(gray.T, safe.T, word_mask=words.T,
                                crossing_mask=crossings.T,word_safe_ridge=verified_crossings)
    if verified_crossings and horizontal and vertical:
        horizontal, vertical = refill_crossings(gray, safe, horizontal, vertical,
                                               words, groups,precise_ink=True)
    return horizontal, vertical


def refill_crossings(gray, safe, horizontal, vertical, words, groups=None, *,precise_ink=False):
    """Resolve a crossing only after measuring its missing perpendicular ink."""
    h,w=gray.shape
    independent=words if not groups or precise_ink else np.zeros_like(gray)
    for group in [] if not groups or precise_ink else groups:
        lo=np.floor(group.min(0)*[w-1,h-1]).astype(int)-2
        hi=np.ceil(group.max(0)*[w-1,h-1]).astype(int)+2
        independent[max(0,lo[1]):min(h,hi[1]+1),
                    max(0,lo[0]):min(w,hi[0]+1)]=1
    extended=[extend_to_crossings(gray,safe,guide,
                                  [g[:,::-1] for g in vertical],independent)
              for guide in horizontal]
    if all(np.array_equal(a,z) for a,z in zip(horizontal,extended)):
        return horizontal,vertical
    crossings=np.zeros_like(gray)
    for guide in extended:
        cv2.polylines(crossings,
                      [np.rint(guide*[w-1,h-1]).astype(np.int32)],False,1,9)
    measured=measure_sections(np.ascontiguousarray(gray.T),
                              np.ascontiguousarray(safe.T),word_mask=words.T,
                              crossing_mask=crossings.T,word_safe_ridge=True)
    return extended,measured


def printed_grid(horizontal, vertical):
    return (sum(np.ptp(g[:, 0]) > .3 for g in horizontal) >= 3 and
            sum(np.ptp(g[:, 0]) > .2 for g in vertical) >= 3)


def supported_paper_rows(gray, safe, words, horizontal):
    """Require repeated measured rules and glyph rows for non-grid paper.

    The caller selects flexible paper. Native boxes only associate glyphs;
    neither page position nor a receipt filename supplies deformation evidence.
    """
    if sum(np.ptp(g[:, 0]) > .3 for g in horizontal) < 3:
        return False
    from content_text_baselines import capture_text_baselines
    traces = capture_text_baselines(gray, safe, words)
    supported = [np.asarray(t['points']) for t in traces
                 if t['glyph_baseline_support']['native_word_count'] >= 3
                 and np.ptp(np.asarray(t['points'])[:, 0]) > .15]
    return (len(supported) >= 3 and
            np.ptp([p[:, 1].mean() for p in supported]) > .15)


def carry_word_groups(base,groups,sampler):
    """Retain earlier ink when a later recognition box trims the same word."""
    clouds=[np.asarray(g,dtype=float) for g in base.get('retained_word_clouds',[])]
    if any(g.ndim!=2 or g.shape[1]!=2 or not len(g) or not np.isfinite(g).all()
           for g in clouds):raise ValueError('Invalid retained word geometry')
    result=[np.array(g,copy=True) for g in groups]
    before=[]
    if clouds:
        current=inverse_retained(sampler,np.vstack(clouds))
        before=np.split(current,np.cumsum([len(g) for g in clouds])[:-1])
    references=[np.asarray(g,dtype=float) for g in base.get('retained_word_references',[])]
    if clouds and (len(references)!=len(clouds) or any(a.shape!=b.shape or
            not np.isfinite(a).all() for a,b in zip(references,clouds))):
        raise ValueError('Retained words need their original paired geometry')
    for group in before:
        lo,hi=group.min(0),group.max(0);matches=[]
        for index,current in enumerate(groups):
            a,z=current.min(0),current.max(0)
            overlap=np.maximum(0,np.minimum(hi,z)-np.maximum(lo,a)).prod()
            area=min((hi-lo).prod(),(z-a).prod())
            if area>0 and overlap/area>.5:matches.append(index)
        # Ambiguous ownership cannot silently merge two independent words.
        # Keep the earlier observation as its own conservative constraint.
        if len(matches)==1:
            index=matches[0];result[index]=np.unique(np.vstack([result[index],group]),axis=0)
        else:
            # Recognition may omit an earlier complete word. Its ink is still
            # an active owner, not merely an after-the-fact audit constraint.
            result.append(np.array(group,copy=True))
    if not clouds:
        clouds=[sampler.sample(g) for g in groups]
        references=[np.array(g,copy=True) for g in groups]
    else:
        # Freeze newly observed independent words as well. Replacing this list
        # with only the latest OCR would lose protection on a subsequent pass.
        for group in groups:
            lo,hi=group.min(0),group.max(0)
            overlaps=False
            for prior in before:
                a,z=prior.min(0),prior.max(0)
                area=min((hi-lo).prod(),(z-a).prod())
                if area>0 and np.maximum(0,np.minimum(hi,z)-np.maximum(lo,a)).prod()/area>.5:
                    overlaps=True;break
            if not overlaps:
                clouds.append(sampler.sample(group))
                references.append(np.array(group,copy=True))
    return result,clouds,references


def retained_word_audit(base,result,clouds,pixel_size,*,references=None,limit_px=.1):
    if not np.isfinite(limit_px) or not 0 < limit_px <= .1:
        raise ValueError('Retained ink tolerance cannot exceed the existing guard')
    prior,candidate=MeshSampler(base),MeshSampler(result)
    records=[];scale=np.asarray(pixel_size)
    if not clouds:return records
    sources=np.vstack(clouds);offsets=np.cumsum([len(g) for g in clouds])[:-1]
    actual=np.split(inverse_retained(candidate,sources)*scale,offsets)
    before=(np.split(inverse_retained(prior,sources)*scale,offsets) if references is None
            else [np.asarray(g)*scale for g in references])
    if len(before)!=len(actual) or any(a.shape!=z.shape or not np.isfinite(a).all()
                                       for a,z in zip(before,actual)):
        raise ValueError('Word audit needs paired reference geometry')
    for index,(a,z) in enumerate(zip(before,actual)):
        aa,zz=a-a.mean(0),z-z.mean(0);u,_,v=np.linalg.svd(aa.T@zz);rotation=u@v
        error=float(np.linalg.norm(aa@rotation-zz,axis=1).max())
        ids=np.linspace(0,len(a)-1,min(96,len(a)),dtype=int)
        pairs=float(abs(np.linalg.norm(a[ids,None]-a[None,ids],axis=2)-
                        np.linalg.norm(z[ids,None]-z[None,ids],axis=2)).max())
        records.append({'index':index,'rigid_shape_error_px':error,
                        'pair_distance_change_px':pairs,
                        'accepted':bool(np.linalg.det(rotation)>0 and max(error,pairs)<limit_px)})
    return records


def column_constraints(base,sampler,words,groups,native_count,width,height):
    """Detect working columns and retain earlier independent column geometry."""
    quads=[np.asarray(word['points']) for word in measurement_words(words,width,height)]
    links=[];envelopes={};records=list(base.get('established_text_columns',[]))
    for column in aligned_columns(quads,[width-1,height-1]):
        owners=independent_owner_links([column],quads,groups,native_count)
        if not owners:continue
        links.append(owners[0])
        for index,owner in zip(column,owners[0]):envelopes[owner]=quads[index]
        anchors=np.array([quads[index][np.argsort(quads[index][:,0])[:2]].mean(0)
                          for index in column])
        records.append({'normal_axis':0,'points':sampler.sample(anchors).tolist(),
                        'reference':anchors.tolist()})
    return links,envelopes,records


def column_geometry_audit(result,records,pixel_size,*,word_bindings=None):
    sampler=MeshSampler(result);checks=[]
    if word_bindings is not None and len(word_bindings)!=len(records):
        raise ValueError('Every column needs an explicit binding or legacy fallback')
    for index,record in enumerate(records):
        reference=np.asarray(record['reference'])
        source=np.asarray(record['points'])
        if reference.shape!=source.shape or source.ndim!=2 or source.shape[1]!=2 or len(source)<3:
            raise ValueError('Invalid retained text column geometry')
        if word_bindings is not None and word_bindings[index] is not None:
            from column_word_origins import bound_column_audit
            checks.append(bound_column_audit(sampler,record,word_bindings[index],pixel_size))
            continue
        actual=inverse_retained(sampler,source)
        checks.append(alignment_audit(reference,actual,pixel_size,
                                      normal_axis=record['normal_axis']))
    return checks


def fixed_text_word_owners(base, sampler, groups, pixel_size):
    """Keep an established text body's current owners fixed during rule repairs.

    After-the-fact position checks remain mandatory. These constraints prevent
    the stock solver from first translating a word it already knows is fixed.
    Multiple owners of an established row are all held conservatively.
    """
    fixed = set()
    padding = .1 / np.asarray(pixel_size, dtype=float)
    for row in base.get('established_text_rows', []):
        current = inverse_retained(sampler, np.asarray(row['points']))
        low, high = current.min(0) - padding, current.max(0) + padding
        for index, group in enumerate(groups):
            a, z = group.min(0) - padding, group.max(0) + padding
            covered = np.all((current >= a) & (current <= z), axis=1)
            within_row = np.all((group >= low) & (group <= high), axis=1)
            if (covered.sum() >= min(12, len(current)) and
                    (within_row.mean() >= .9 or covered.mean() >= .9)):
                fixed.add(index)
    return sorted(fixed)


def join_audit(old, new, horizontal, vertical, width, height):
    """A locally straight section may not introduce a kink at its joins."""
    report = []
    for axis, guides, scale in [(1, horizontal, height), (0, vertical, width)]:
        guides = sorted(guides, key=lambda g: g[0, 0])
        for a in guides:
            for b in guides:
                gap = b[0, 0] - a[-1, 0]
                if not 0 < gap < .07:
                    continue
                separation = abs(b[0, 1] - a[-1, 1])
                if separation >= .012:
                    continue
                n = min(12, len(a), len(b))
                slope_a = np.polyfit(a[-n:, 0], a[-n:, 1], 1)[0]
                slope_b = np.polyfit(b[:n, 0], b[:n, 1], 1)[0]
                predicted = abs(b[0, 1] - a[-1, 1] - (slope_a+slope_b)*gap/2)
                if predicted >= max(3/scale, gap*.3):
                    continue
                endpoints = np.array([a[-1], b[0]])
                if axis == 0:
                    endpoints = endpoints[:, ::-1]
                after = inverse(new, rows.p.sample(old, endpoints))
                before_gap = abs(endpoints[1, axis]-endpoints[0, axis])*scale
                after_gap = abs(after[1, axis]-after[0, axis])*scale
                report.append({'axis': axis, 'before_gap_px': float(before_gap),
                               'after_gap_px': float(after_gap),
                               'accepted': bool(after_gap <= max(.6, before_gap+.1))})
    return report


def horizontal_step(path, base, groups, gray, safe, horizontal, audit):
    mask=supported_word_mask(gray,groups)
    broad = measure_sections(gray, safe, broad_profile=True,word_mask=mask)
    guides = frame.supported_extensions(horizontal, broad, gray.shape[1], gray.shape[0])
    for guide in guides:
        result = rows.refine(path, base, guides=[guide], word_ink_groups=groups,
                             include_wide_ink=True, align_to_existing=True,
                             minimum_resolution=(4097, 1025), audit_log=audit,
                             words_mask=mask,
                             **OPTIONS)
        if result is not base:
            return result
    for guide in sorted(horizontal, key=lambda g: -np.ptp(g[:, 1])):
        supported, _ = rows.accepted.accepted.residual_evidence(guide, gray.shape[0])
        if not supported or np.ptp(guide[:, 0]) < .15:
            continue
        result = rows.refine(path, base, guides=[guide], word_ink_groups=groups,
                             include_wide_ink=True, align_to_existing=True,
                             minimum_resolution=(4097, 1025), audit_log=audit,
                             words_mask=mask,
                             **OPTIONS)
        if result is not base:
            return result
    return base


def coordinate_shared_row_step(path,base,groups,words,native_count,measurement,
                               horizontal,vertical,audit,*,candidate_guard=None):
    """Align observed rule pieces across fixed words, leaving the ink gap free."""
    from word_flanked_rule_rows import word_flanked_rows,observed_registered_sections
    gray,safe,source,old=measurement;h,w=gray.shape;sampler=MeshSampler(base)
    anchors=[inverse_retained(sampler,np.asarray(q['points']))
             for q in base.get('established_rulers',[]) if q['normal_axis']==1]
    quads=[np.asarray(q['points']) for q in measurement_words(words,w,h)]
    mask=supported_word_mask(gray,groups,precise_ink=True)
    recovered=observed_registered_sections(horizontal,anchors,mask,safe,w,h)
    guides=sorted(recovered+measured_chains(gray,safe,horizontal,mask)+horizontal,
                  key=lambda g:-np.ptp(g[:,0]))
    for row in word_flanked_rows(guides,quads,groups,native_count,anchors,w,h):
        audit.append({'proposal':'coordinate shared word and rule row',
            **{k:v for k,v in row.items() if k!='guides'},
            'bounds':[[g.min(0).tolist(),g.max(0).tolist()] for g in row['guides']]})
        fixed=sorted(set(fixed_text_word_owners(base,sampler,groups,[w,h]))|
                     set(row['word_owners']))
        try:
            result=rows.refine(path,base,guides=row['guides'],measurement=measurement,
                word_ink_groups=groups,words_mask=supported_word_mask(gray,groups),
                include_wide_ink=True,align_to_existing=True,
                minimum_resolution=(4097,w),fixed_word_owners=fixed,
                measured_ink_bands=True,free_tangent_rules=True,tangent_padding=.015,
                tangent_ink_padding_px=1,coordinate_composition=True,audit_log=audit,
                word_flanked_evidence={'quads':quads,'native_count':native_count},
                stock_area_reserve=.21,**OPTIONS)
        except ERRORS as error:
            audit.append({'rejected':'shared row declined','reason':str(error)})
            continue
        if result is not base and (candidate_guard is None or candidate_guard(result)):
            return result
    return base


def coordinate_horizontal_step(path,base,groups,measurement,horizontal,vertical,audit,
                               *,reading_axis=0,minimum_resolution=None,
                               word_translation_links=None,word_owner_envelopes=None,
                               measured_ink_bands=False,words_mask=None,
                               free_tangent_rules=False,candidate_guard=None,
                               complete_guides=None,complete_endpoint_anchors=False):
    """Optional exact-map step with measured word-safe rule endpoints."""
    if complete_endpoint_anchors and complete_guides is None:
        raise ValueError('Endpoint anchors require complete registered stroke guides')
    gray,safe,source,old=measurement;h,w=gray.shape
    sampler=MeshSampler(base)
    fixed_owners=fixed_text_word_owners(base,sampler,groups,[w,h])
    perpendicular=[g[:,::-1] for g in vertical]
    anchors=[]
    for ruler in base.get('established_rulers',[]):
        points=np.asarray(ruler['points'])
        current=inverse_retained(sampler,points)
        if ruler['normal_axis']==0:perpendicular.append(current)
        elif ruler['normal_axis']==1:anchors.append(current)
    independent=np.zeros_like(gray)
    for group in groups:
        lo=np.floor(group.min(0)*[w-1,h-1]).astype(int)-2
        hi=np.ceil(group.max(0)*[w-1,h-1]).astype(int)+2
        independent[max(0,lo[1]):min(h,hi[1]+1),max(0,lo[0]):min(w,hi[0]+1)]=1
    mask=(supported_word_mask(gray,groups) if words_mask is None else
          np.asarray(words_mask,dtype=np.uint8))
    if mask.shape!=gray.shape:raise ValueError('Word mask must use the measurement coordinates')
    anchors=[measured_straight_anchor(gray,safe,anchor,independent)
             for anchor in anchors]
    anchors=[anchor for anchor in anchors if anchor is not None]
    candidates=[]
    continuations=([] if complete_guides is not None else
                   continued_thin_sections(gray,safe,horizontal,
                                           [g[:,::-1] for g in vertical],independent))
    joined=[] if complete_guides is not None else measured_chains(gray,safe,horizontal,independent)
    for guide in (complete_guides if complete_guides is not None else
                  continuations+joined+horizontal):
        if np.ptp(guide[:,0])<.15:continue
        extended=(guide if complete_guides is not None else
                  extend_to_crossings(gray,safe,guide,perpendicular,independent,
                                      collinear=anchors))
        amplitude=np.ptp(np.percentile(extended[:,1],[5,95]))*h
        scatter=np.std(extended[:,1])*h
        # A long ink-backed trace supports local bends even when one global
        # polynomial cannot explain several opposite bends along that trace.
        if amplitude>2*max(h,w)/1280 and scatter>.6*max(h,w)/1280:
            continuation=None if complete_guides is not None else anchored_remainder(extended,anchors,w,h)
            if continuation is not None:
                remainder,target=continuation
                for part in np.split(remainder,np.flatnonzero(np.diff(remainder[:,0])>2.5/(w-1))+1):
                    if len(part)>=16:candidates.append((0,-amplitude,part,target))
            for reverse in (() if complete_guides is not None else (False,True)):
                measured=extended if not reverse else np.c_[1-extended[::-1,0],extended[::-1,1]]
                bend=before_stable_tail(measured,h)
                if bend is not None:
                    part,target=bend
                    if reverse:part=np.c_[1-part[::-1,0],part[::-1,1]]
                    if len(part)>=16:candidates.append((1,-amplitude,part,target))
            if complete_endpoint_anchors:
                # Each target keeps one actual observed material endpoint at
                # its current level. No missing ink or corner is invented.
                for level in dict.fromkeys(float(extended[i,1]) for i in (0,-1)):
                    candidates.append((2,-amplitude,extended,level))
                candidates.append((3,-amplitude,extended,None))
            else:
                candidates.append((2,-amplitude,extended,None))
    for priority,__,guide,target in sorted(candidates,key=lambda item:item[:2]):
        audit.append({'proposal':'coordinate section','reading_axis':reading_axis,
                      'priority':priority,'measured_points':len(guide),
                      'bounds':[guide.min(0).tolist(),guide.max(0).tolist()],
                      'target':target})
        try:
            result=rows.refine(path,base,guides=[guide],measurement=measurement,
                               target_guide=None if target is None else
                                  np.c_[guide[[0,-1],0],np.full(2,target)],
                               word_ink_groups=groups,include_wide_ink=True,
                               align_to_existing=True,
                               minimum_resolution=minimum_resolution or (4097,w),
                               word_reading_axis=reading_axis,
                               word_translation_links=word_translation_links,
                               word_owner_envelopes=word_owner_envelopes,
                               fixed_word_owners=fixed_owners,
                               measured_ink_bands=measured_ink_bands,
                               free_tangent_rules=free_tangent_rules,
                               tangent_padding=.015,
                               words_mask=mask,audit_log=audit,coordinate_composition=True,
                               tangent_ink_padding_px=1,
                               registered_guides=complete_guides is not None,**OPTIONS)
        except ERRORS as error:
            audit.append({'rejected':'coordinate section declined','reason':str(error),
                          'bounds':[guide.min(0).tolist(),guide.max(0).tolist()]})
            continue
        if result is not base and (candidate_guard is None or candidate_guard(result)):
            return result
    return base


def coordinate_vertical_step(path,base,groups,measurement,horizontal,vertical,audit,
                             *,word_translation_links=None,word_owner_envelopes=None,
                             measured_ink_bands=False,free_tangent_rules=False,
                             candidate_guard=None,complete_guides=None,
                             complete_endpoint_anchors=False):
    """Apply the same exact normal correction with both axes transposed."""
    gray,safe,source,old=measurement
    transposed=transpose_mesh(base)
    rotated_measurement=(np.ascontiguousarray(gray.T),np.ascontiguousarray(safe.T),
                         None if source is None else np.ascontiguousarray(source.T),
                         old.transpose(1,0,2)[:,:,::-1])
    rotated_groups=[group[:,::-1] for group in groups]
    def restored_guard(result):
        restored=transpose_mesh(result)
        restored['preserved_mesh']=base
        return candidate_guard(restored)
    result=coordinate_horizontal_step(path,transposed,rotated_groups,rotated_measurement,
                                      vertical,horizontal,audit,reading_axis=1,
                                      words_mask=supported_word_mask(gray,groups).T,
                                      word_translation_links=word_translation_links,
                                      measured_ink_bands=measured_ink_bands,
                                      free_tangent_rules=free_tangent_rules,
                                      candidate_guard=None if candidate_guard is None else restored_guard,
                                      complete_guides=complete_guides,
                                      complete_endpoint_anchors=complete_endpoint_anchors,
                                      word_owner_envelopes={i:np.asarray(q)[:,::-1]
                                          for i,q in (word_owner_envelopes or {}).items()},
                                      minimum_resolution=(max(2049,old.shape[1],gray.shape[1]),
                                                          max(old.shape[0],gray.shape[0])))
    if result is transposed:return base
    restored=transpose_mesh(result)
    restored['preserved_mesh']=base
    restored['method']='Supported vertical-rule correction through retained coordinate maps'
    return restored


def coordinate_query_grid(result):
    """Reconstruct only the coordinate primitive at its original global knots.

    A stored local window is not a normalized full-page seed. Its primitive
    forwards unchanged coordinates outside its owned region; the actual parent
    stays separate and is never sampled into this inverse seed.
    """
    if 'grid_window' not in result:
        return rows.accepted.accepted.grid_of(result)
    identity={'rows':2,'columns':2,'grid':[[0,0],[1,0],[0,1],[1,1]]}
    primitive=MeshSampler({**result,'preserved_mesh':identity})
    return primitive.dense(*result['grid_window'][:2])


def coordinate_stage_audit(base,result,old,groups,horizontal,vertical,width,height,*,measurement=None):
    """Audit the true composition; never inverse a collapsed approximation."""
    prior,candidate=MeshSampler(base),MeshSampler(result)
    query=coordinate_query_grid(result)
    scale=np.array([width,height]);current=np.vstack(groups)
    after=inverse_near(candidate,prior.sample(current),inverse(query,current))*scale
    words=[];offset=0
    for index,group in enumerate(groups):
        before=group*scale;actual=after[offset:offset+len(group)];offset+=len(group)
        a,z=before-before.mean(0),actual-actual.mean(0)
        u,_,v=np.linalg.svd(a.T@z);rotation=u@v
        error=float(np.linalg.norm(a@rotation-z,axis=1).max())
        ids=np.linspace(0,len(group)-1,min(96,len(group)),dtype=int)
        pairs=float(abs(np.linalg.norm(before[ids,None]-before[None,ids],axis=2)-np.linalg.norm(actual[ids,None]-actual[None,ids],axis=2)).max())
        words.append({'index':index,'rigid_shape_error_px':error,'pair_distance_change_px':pairs,'accepted':bool(np.linalg.det(rotation)>0 and max(error,pairs)<.1)})
    texts=[]
    for row in base.get('established_text_rows',[]):
        source=np.asarray(row['points']);before=inverse_retained(prior,source)
        actual=inverse_near(candidate,source,inverse(query,before))
        error=float(abs((actual-before)*scale).max())
        texts.append({'position_change_px':error,'accepted':bool(error<.1)})
    rulers=[];retargeted={};join_registry=[];observed_extensions={}
    if result.get('method')=='Supported shared word and rule row through retained coordinate maps':
        if result.get('preserved_mesh') is not base:
            raise ValueError('A shared row must directly retain its evidence frame')
        ledger=result['word_flanked_rule_row_audit']
        for record in ledger['retargeted_rules']:
            index=record['index']
            if (not 0<=index<len(base.get('established_rulers',[])) or
                    base['established_rulers'][index]['normal_axis']!=1 or
                    base['established_rulers'][index]['points']!=record['source_points']):
                raise ValueError('Shared row changed an immutable ruler registration')
            retargeted[index]=record
    for index,rule in enumerate(base.get('established_rulers',[])):
        source=np.asarray(rule['points']);before=inverse_retained(prior,source)
        actual=inverse_retained(candidate,source);axis=rule['normal_axis']
        if axis==1:
            join_registry.append({'index':index,'source':source,'reference':before})
        delta=(actual-before)*scale
        error=float(abs(delta).max() if axis is None else abs(delta[:,axis]).max())
        if index in retargeted:
            reference=np.asarray(retargeted[index]['reference_points'])
            if reference.shape!=before.shape or abs((reference-before)*scale).max()>1e-6:
                raise ValueError('Shared row reset its ruler reference frame')
            selected=np.asarray(ledger['guides'][1-ledger['anchor_side']])
            outside=((selected[:,0]<before[:,0].min()-1e-10)|
                     (selected[:,0]>before[:,0].max()+1e-10))
            if outside.any():
                # An old registration must never be extrapolated into the
                # native tail. Reconstruct that continuation from the same
                # measured sections, safe stock and complete word mask.
                if measurement is None:
                    raise ValueError('Observed rule tails require fresh raster evidence')
                gray,safe=measurement
                from word_flanked_rule_rows import observed_registered_sections
                mask=supported_word_mask(gray,groups,precise_ink=True)
                observed=observed_registered_sections(horizontal,[before],mask,safe,width,height)
                if (len(observed)!=1 or observed[0].shape!=selected.shape or
                        abs(observed[0]-selected).max()>1e-10):
                    raise ValueError('Shared row continuation differs from its observed native stroke')
                positions=np.clip(np.searchsorted(selected[:,0],before[:,0]),0,len(selected)-1)
                previous=np.maximum(0,positions-1)
                positions=np.where(abs(selected[previous,0]-before[:,0])<
                                   abs(selected[positions,0]-before[:,0]),previous,positions)
                if ((np.diff(positions)<=0).any() or
                        abs(selected[positions]-before).max()>1e-10):
                    raise ValueError('Shared row continuation changed old material points')
                extended_source=prior.sample(selected)
                extended_source[positions]=source
                join_registry[-1]={'index':index,'source':extended_source,'reference':selected}
                observed_extensions[index]={'observed_continuation_points':int(outside.sum()),
                                            'original_registration_points':len(before)}
            prior_range=float(np.ptp(before[:,axis])*scale[axis])
            current_range=float(np.ptp(actual[:,axis])*scale[axis])
            target_distance=float(abs(np.median(actual[:,axis])-ledger['target'])*scale[axis])
            rulers.append({'index':index,'normal_axis':axis,
                'normal_position_change_px':error,'before_normal_range_px':prior_range,
                'after_normal_range_px':current_range,'target_distance_px':target_distance,
                'basis':'Explicit word-flanked shared row; complete earlier ruler geometry retained',
                'accepted':bool(current_range<=prior_range+.05 and target_distance<2)})
            continue
        rulers.append({'index':index,'normal_axis':axis,
                       'normal_position_change_px':error,'accepted':bool(error<.1)})
    joins=[]
    for axis,guides in [(1,horizontal),(0,vertical)]:
        for a in guides:
            for b in guides:
                gap=b[0,0]-a[-1,0]
                if not 0<gap<.07 or abs(b[0,1]-a[-1,1])>=.012:continue
                n=min(12,len(a),len(b))
                slopes=[np.polyfit(a[-n:,0],a[-n:,1],1)[0],np.polyfit(b[:n,0],b[:n,1],1)[0]]
                predicted=abs(b[0,1]-a[-1,1]-sum(slopes)*gap/2)
                if predicted>=max(3/scale[axis],gap*.3):continue
                endpoints=np.array([a[-1],b[0]])
                if axis==0:endpoints=endpoints[:,::-1]
                actual=inverse_near(candidate,prior.sample(endpoints),inverse(query,endpoints))
                before_gap=float(abs(endpoints[1,axis]-endpoints[0,axis])*scale[axis])
                after_gap=float(abs(actual[1,axis]-actual[0,axis])*scale[axis])
                record={'axis':axis,'before_gap_px':before_gap,'after_gap_px':after_gap,
                        'accepted':bool(after_gap<=max(.6,before_gap+.1))}
                if axis==1 and retargeted:
                    from rule_join_evidence import registered_material_join
                    binding=registered_material_join(join_registry,a,b,groups,width,height)
                    if binding is not None and binding['ruler_index'] in retargeted:
                        physical_after=inverse_retained(candidate,binding['source_points'])
                        physical_before=binding['reference_points']
                        physical_gap_before=float(abs(physical_before[1,1]-physical_before[0,1])*height)
                        physical_gap_after=float(abs(physical_after[1,1]-physical_after[0,1])*height)
                        record={**record,'raster_before_gap_px':before_gap,'raster_after_gap_px':after_gap,
                            'raster_estimate_accepted':record['accepted'],
                            'before_gap_px':physical_gap_before,'after_gap_px':physical_gap_after,
                            'basis':'Complete immutable source stroke with uniquely matched native sections',
                            **{k:v for k,v in binding.items() if k not in ('source_points','reference_points')},
                            **observed_extensions.get(binding['ruler_index'],{}),
                            'accepted':bool(physical_gap_after<=max(.6,physical_gap_before+.1))}
                joins.append(record)
    return words,texts,joins,rulers


def coordinate_word_rulers(base,old,horizontal,vertical,source,parent,gray,safe,groups):
    """Hold complete measured strokes during a text-only coordinate repair."""
    from measured_ink_bands import rule_ink_footprints
    rulers=word_rulers(base,old,horizontal,vertical,source,sampler=parent)
    current=[{**rule,'points':inverse_retained(parent,
                         np.asarray(rule['points'])).tolist()} for rule in rulers]
    # Fix their full positions during text correction. Rule repair remains
    # responsible for any separately authorized tangent movement.
    clouds=rule_ink_footprints(gray,safe,horizontal,vertical,groups)
    current.extend({'normal_axis':None,'geometry':'ink_cloud','points':cloud['points'].tolist()}
                   for cloud in clouds)
    return current


def coordinate_word_row_step(base,groups,words,measurement,horizontal,vertical,audit,
                             *,minimum_resolution=(4097,1025),candidate_guard=None,
                             full_pixel_footprints=False):
    """Restore whole native word-row rotations through the true parent map."""
    gray,safe,source,old=measurement;h,w=gray.shape;parent=MeshSampler(base)
    quads=word_rows.find_rows(measurement_words(words,w,h),[w-1,h-1])
    candidates=word_rows.parallel_blocks(quads,[w-1,h-1])+quads
    if not candidates:return base
    current_rulers=coordinate_word_rulers(base,old,horizontal,vertical,
                                         source,parent,gray,safe,groups)
    nr,nc=max(old.shape[0],minimum_resolution[0]),max(old.shape[1],minimum_resolution[1],w)
    u,v=np.meshgrid(np.arange(nc)/(nc-1),np.arange(nr)/(nr-1))
    identity=np.stack([u,v],axis=2)
    fixed_owners=fixed_text_word_owners(base,parent,groups,[w,h])
    for row in sorted(candidates,key=lambda row:np.vstack(row)[:,1].min()):
        fixed_in_row=[i for i in fixed_owners if any(
            np.all(groups[i].mean(0)>=quad.min(0)-.002) and
            np.all(groups[i].mean(0)<=quad.max(0)+.002) for quad in row)]
        if fixed_in_row:
            audit.append({'rejected':'word row includes established text owner',
                          'fixed_word_owners':fixed_in_row});continue
        query,report=word_rows.refine(identity,row,groups,[w-1,h-1],current_rulers,
                                       use_flow=True,stock_padding=32,
                                       ink_padding=2 if full_pixel_footprints else .5,
                                       flow_body_padding=2 if full_pixel_footprints else .5)
        if query is identity:
            audit.append(report);continue
        try:
            try:
                proof=validate_source_composition(parent,query,[report['region']])
            except ValueError as error:
                if 'Nested coordinate layers' not in str(error):raise
                from retained_composition_guard import validate_retained_composition
                proof=validate_retained_composition(parent,query,[report['region']])
        except ERRORS as error:
            audit.append({'rejected':'word-row composition declined','reason':str(error)});continue
        if not proof['accepted']:
            audit.append({'rejected':'word-row continuous area','proof':proof});continue
        audit.append(report)
        owned=np.vstack([groups[i] for i in report['independent_word_owners']])
        result={**base,'rows':nr,'columns':nc,'grid':query.reshape(-1,2).tolist(),
                'coordinate_transform':True,'preserved_mesh':base,
                'refinement_regions':[report['region']],
                'method':'Rigid printed-word row correction through retained maps',
                'rigid_word_row_audit':report,'continuous_jacobian_audit':proof,
                'established_text_rows':[*base.get('established_text_rows',[]),
                                         {'points':parent.sample(owned).tolist()}]}
        result.pop('grid_window',None)
        if candidate_guard is not None and not candidate_guard(result):continue
        return result
    return base


def coordinate_word_step(base,groups,measurement,horizontal,vertical,audit,
                         *,candidate_guard=None,measured_motion_owners=False):
    """Optional glyph-supported word rotation in current coordinates."""
    gray,safe,source,old=measurement;h,w=gray.shape;parent=MeshSampler(base)
    current_rulers=coordinate_word_rulers(base,old,horizontal,vertical,
                                         source,parent,gray,safe,groups)
    nr,nc=max(old.shape[0],4097),max(old.shape[1],w)
    u,v=np.meshgrid(np.arange(nc)/(nc-1),np.arange(nr)/(nr-1))
    identity=np.stack([u,v],axis=2)
    for index in sorted(range(len(groups)),key=lambda i:groups[i][:,1].min()):
        evidence=word_rows.word_rotation_evidence(gray,groups[index],
                                    pixel_size=[w-1,h-1],body_consensus=True)
        if evidence is None:continue
        for padding in (24,28,32):
            query,report=word_rows.refine_word(identity,gray,groups,index,current_rulers,
                                              pixel_size=[w-1,h-1],body_consensus=True,
                                              stock_padding=padding,flow_constraint_padding=4,
                                              flow_body_padding=4,precise_fixed_ink=True,
                                              measured_motion_owners=measured_motion_owners)
            if query is identity:
                audit.append(report);continue
            try:
                try:
                    proof=validate_source_composition(parent,query,[report['region']])
                except ValueError as error:
                    if 'Nested coordinate layers' not in str(error):raise
                    from retained_composition_guard import validate_retained_composition
                    proof=validate_retained_composition(parent,query,[report['region']])
            except ERRORS as error:
                audit.append({'rejected':'word composition declined','reason':str(error),'word_index':index})
                continue
            if not proof['accepted']:
                audit.append({'rejected':'word continuous area','proof':proof,'word_index':index});continue
            audit.append(report)
            result={**base,'rows':nr,'columns':nc,'grid':query.reshape(-1,2).tolist(),
                    'coordinate_transform':True,'preserved_mesh':base,
                    'refinement_regions':[report['region']],
                    'method':'Glyph-supported word rotation through retained maps',
                    'rigid_word_row_audit':report,'continuous_jacobian_audit':proof,
                    'established_text_rows':[*base.get('established_text_rows',[]),
                                             {'points':parent.sample(np.vstack([
                                                 groups[i] for i in report['independent_word_owners']])).tolist()}]}
            result.pop('grid_window',None)
            if candidate_guard is not None and not candidate_guard(result):continue
            return result
    return base


def word_rulers(base, old, horizontal, vertical, source, *,sampler=None):
    """Keep full rules, barcode bars/envelopes and circular marks fixed."""
    rulers = list(base.get('established_rulers', []))
    rulers.extend({'normal_axis':None,'geometry':'ink_cloud','points':r['points']}
                  for r in base.get('established_text_rows',[]))
    # All measured rules stay fixed during a text-only rotation, including
    # rules that have not yet been selected for their own correction.
    for axis, guides in [(1, horizontal), (0, vertical)]:
        for guide in guides:
            uv = guide if axis == 1 else guide[:, ::-1]
            rulers.append({'normal_axis': axis,
                           'points': (sampler.sample(uv) if sampler is not None else
                                      rows.p.sample(old, uv)).tolist()})
    if source is not None:
        bars = rows.p.reference.barcode_rulers(source)
        shapes = (bars + rows.p.reference.measured_envelopes(source, bars) +
                  rows.accepted.accepted.circular_landmarks(source))
        rulers.extend({'normal_axis': None, 'points': q.tolist()} for q in shapes)
    return rulers


def word_result(base, old, field, evidence, groups, method):
    owned=np.vstack([groups[i] for i in evidence['independent_word_owners']])
    return {**base,'preserved_mesh':base,
            'refinement_regions':[evidence['region']],
            'rows':field.shape[0],'columns':field.shape[1],
            'grid':field.reshape(-1,2).tolist(),'method':method,
            'rigid_word_row_audit':evidence,
            'established_text_rows':[*base.get('established_text_rows',[]),
                                     {'points':rows.p.sample(old,owned).tolist()}]}


def established_text_audit(base, old, new, pixel_size):
    checks=[]
    for row in base.get('established_text_rows',[]):
        source=np.asarray(row['points'])
        before=inverse(old,source);after=inverse(new,source)
        error=float(abs((after-before)*pixel_size).max())
        checks.append({'position_change_px':error,'accepted':bool(error<.1)})
    return checks


def word_step(base, groups, words, old, horizontal, vertical, width, height, audit,
              source=None,current_gray=None):
    rulers = word_rulers(base, old, horizontal, vertical, source)
    nr, nc = max(old.shape[0], 4097), max(old.shape[1], 1025)
    u, v = np.meshgrid(np.linspace(0, 1, nc), np.linspace(0, 1, nr))
    dense = rows.p.sample(old, np.c_[u.ravel(), v.ravel()]).reshape(nr, nc, 2)
    rows_to_check=word_rows.find_rows(measurement_words(words,width,height),
                                     [width-1,height-1])
    blocks=word_rows.parallel_blocks(rows_to_check,[width-1,height-1])
    for quads in sorted(blocks+rows_to_check,
                        key=lambda row: np.vstack(row)[:,1].min()):
        field, evidence = word_rows.refine(dense, quads, groups, [width-1, height-1], rulers,
                                           use_flow=True,ink_padding=.5,stock_padding=32)
        audit.append(evidence)
        if field is dense:
            continue
        return word_result(base,dense,field,evidence,groups,
                           'Rigid printed-word row correction')
    if current_gray is not None:
        # Native recognition may return a short word with a horizontal box.
        # Individual measured glyphs must independently support its rotation.
        for index,group in enumerate(groups):
            field,evidence=word_rows.refine_word(dense,current_gray,groups,index,
                                                rulers,pixel_size=[width-1,height-1])
            audit.append(evidence)
            if field is dense:
                continue
            return word_result(base,dense,field,evidence,groups,
                               'Rigid glyph-supported word correction')
    return base


def word_geometry_audit(old, new, groups, width, height, *, fixed=False, rotated=()):
    """Validate against the actual prior grid, including dense resampling."""
    sources = rows.p.sample(old, np.vstack(groups))
    after = inverse(new, sources) * [width, height]
    records, offset = [], 0
    for index, group in enumerate(groups):
        before = group * [width, height]
        actual = after[offset:offset+len(group)]
        offset += len(group)
        a, b = before-before.mean(0), actual-actual.mean(0)
        u, _, v = np.linalg.svd(a.T @ b)
        rotation = u @ v
        error = float(np.linalg.norm(a @ rotation-b, axis=1).max())
        position = float(np.linalg.norm(actual-before, axis=1).max())
        keep = not fixed or index in rotated or position < .1
        records.append({'index': index, 'rigid_shape_error_px': error,
                        'position_change_px': position,
                        'accepted': bool(np.linalg.det(rotation)>0 and error<.1 and keep)})
    return records


def refine_staged(path, output, base, *, word_provider=request_words, max_steps=12,
                  attempt_observer=None,coordinate_mode=False,crop_words=False,
                  partition_words=False,localization_halo=1,bracketed_rules=False,
                  measured_ink_bands=False,supported_short_rules=False,
                  coordinate_word_rows=False,free_tangent_rules=False,
                  physical_column_origins=False,coordinate_shared_rows=False,
                  unrecognized_print=False,initial_evidence_observer=None,
                  complete_strokes=False,whole_stroke_proposals=False,
                  measured_motion_owners=False,complete_endpoint_anchors=False,
                  non_grid_paper=False,full_pixel_footprints=False,
                  whole_rulings=False):
    full_pixel_footprints = (full_pixel_footprints or non_grid_paper
                            or bool(base.get('retained_pixel_footprints')))
    if whole_rulings and not (full_pixel_footprints and whole_stroke_proposals
                             and coordinate_word_rows):
        raise ValueError('Whole rulings require complete original pixel/stroke guards and whole text rows')
    if full_pixel_footprints and not (coordinate_mode and partition_words
                                    and measured_ink_bands and unrecognized_print
                                    and complete_strokes):
        raise ValueError('Full pixel footprints require complete measured original print protection in coordinate mode')
    if full_pixel_footprints and len(base.get('retained_pixel_footprints', [])) != len(base.get('retained_word_clouds', [])):
        raise ValueError('Pixel footprints must be paired before the first content correction')
    if non_grid_paper and not (coordinate_mode and partition_words
                              and measured_ink_bands and unrecognized_print
                              and complete_strokes):
        raise ValueError('Non-grid paper requires complete measured original print protection')
    if measured_motion_owners and not (coordinate_mode and complete_strokes
                                       and measured_ink_bands and unrecognized_print):
        raise ValueError('Measured word motions require original material and complete-stroke guards')
    if whole_stroke_proposals and not complete_strokes:
        raise ValueError('Whole-stroke proposals require complete original stroke protection')
    if complete_endpoint_anchors and not whole_stroke_proposals:
        raise ValueError('Endpoint anchors require complete original stroke proposals')
    if complete_strokes and not coordinate_mode:
        raise ValueError('Complete printed strokes require exact coordinate refinement')
    if bracketed_rules and not (coordinate_mode and partition_words):
        raise ValueError('Bracketed rule roles require paired coordinate word evidence')
    if measured_ink_bands and not (coordinate_mode and partition_words):
        raise ValueError('Measured ink bands require paired coordinate word evidence')
    if supported_short_rules and not (coordinate_mode and partition_words):
        raise ValueError('Supported short rules require paired coordinate word evidence')
    if coordinate_word_rows and not coordinate_mode:
        raise ValueError('Coordinate word rows require retained coordinate refinement')
    if free_tangent_rules and not measured_ink_bands:
        raise ValueError('Free rule tangents require measured physical ink protection')
    if physical_column_origins and not (coordinate_mode and partition_words):
        raise ValueError('Physical column origins require paired immutable word evidence')
    if coordinate_shared_rows and not measured_ink_bands:
        raise ValueError('Coordinate shared rows require measured physical word protection')
    if unrecognized_print and coordinate_mode and not (partition_words and measured_ink_bands):
        raise ValueError('Unrecognized print requires measured retained coordinate owners')
    output = Path(output)
    history = []
    for step in range(max_steps):
        try:
            gray,safe,source,old=(rows.current_measurement(path,base,exact_layers=True) if coordinate_mode else rows.current_measurement(path,base))
            section_mask_evidence=[]
            horizontal, vertical = sections(gray, safe,verified_crossings=coordinate_mode,
                repeated_dash_rules=full_pixel_footprints,audit_log=section_mask_evidence,
                whole_rulings=whole_rulings)
            measured_whole_rulings=[g.copy() for g in horizontal] if whole_rulings else None
            grid_supported = printed_grid(horizontal, vertical)
            if not grid_supported and not non_grid_paper:
                return base
            words=(request_words_with_crops(gray,output.parent,provider=lambda image,folder,**kwargs:word_provider(image,folder)) if crop_words else word_provider(gray,output.parent))
            if not grid_supported and not supported_paper_rows(gray, safe, words, horizontal):
                return base
            groups = []
            word_roles=[];role_audit=[];component_roles=None;faint_roles=None
            bracketed_roles=[];bracketed_audit=[]
            short_roles=[];short_audit=[]
            repeated_roles=[];repeated_audit=[]
            if coordinate_mode and partition_words:
                observations=ink_observations(gray,words,safe)
                if localization_halo>1:
                    observations,faint_roles=recover_faint_components(gray,observations,
                                        words,safe,halo=localization_halo)
                groups,component_roles=partition_shared_ink(gray,
                    [q['ink'] for q in observations],[q['quad'] for q in observations])
                groups,word_roles,role_audit=partition_rule_fragments(
                    groups,horizontal,gray.shape[1],gray.shape[0])
                if bracketed_rules:
                    groups,bracketed_roles,bracketed_audit=partition_bracketed_rules(
                        groups,horizontal,gray.shape[1],gray.shape[0])
                if supported_short_rules:
                    groups,short_roles,short_audit=partition_supported_short_sections(
                        groups,horizontal,vertical,gray.shape[1],gray.shape[0])
                if whole_rulings:
                    from word_rule_partition import partition_measured_repeated_rulings
                    groups,repeated_roles,repeated_audit=partition_measured_repeated_rulings(
                        groups,section_mask_evidence,gray.shape[1],gray.shape[0])
                sampler=MeshSampler(base)
                for role in word_roles+bracketed_roles+short_roles+repeated_roles:
                    role['source_points']=sampler.sample(np.asarray(role['points'])).tolist()
                for role in component_roles['roles']:
                    for transfer in role['transferred']:
                        transfer['source_points']=sampler.sample(np.asarray(transfer['points'])).tolist()
                if faint_roles is not None:
                    for role in faint_roles['components']:
                        role['source_points']=sampler.sample(np.asarray(role['points'])).tolist()
            else:
                groups=ink_groups(gray,words,safe)
            native_count=len(groups)
            if len(groups) < 10:
                return base
            retained=[];references=[]
            if coordinate_mode or unrecognized_print:
                groups,retained,references=carry_word_groups(base,groups,MeshSampler(base))
            missed_audit=None
            new_native_stop=len(retained)
            missed_clouds=base.get('unrecognized_print_clouds',[])
            missed_references=base.get('unrecognized_print_references',[])
            if full_pixel_footprints:
                # Classify new missed print against the measured rule coverage
                # after actual native and prior owners have protected their ink.
                horizontal,vertical=sections(gray,safe,groups,verified_crossings=True,
                    repeated_dash_rules=True,audit_log=section_mask_evidence,
                    whole_rulings=whole_rulings,reference_guides=measured_whole_rulings)
            if unrecognized_print:
                from unrecognized_print_owners import unrecognized_groups
                additional,missed_audit=unrecognized_groups(gray,groups,safe,horizontal,vertical)
                missed_sampler=MeshSampler(base)
                new_sources=[missed_sampler.sample(g) for g in additional]
                groups.extend(additional)
                retained.extend(new_sources)
                references.extend(additional)
                missed_clouds=[*missed_clouds,*[g.tolist() for g in new_sources]]
                missed_references=[*missed_references,*[g.tolist() for g in additional]]
            pixel_owners=list(base.get('retained_pixel_footprints', []))
            if full_pixel_footprints:
                from material_pixel_footprints import capture as capture_pixels
                for owner_index in range(len(pixel_owners), len(retained)):
                    pixel_owners.extend(capture_pixels(MeshSampler(base),
                        [references[owner_index]], [gray.shape[1],gray.shape[0]],
                        first_owner=owner_index,
                        kind='unknown' if owner_index >= new_native_stop else 'word'))
            links=[];envelopes={};columns=[]
            if coordinate_mode:
                links,envelopes,columns=column_constraints(base,MeshSampler(base),words,
                        groups,native_count,gray.shape[1],gray.shape[0])
            bindings=None
            if physical_column_origins:
                from column_word_origins import column_word_bindings
                bindings=column_word_bindings(base,columns,
                        [gray.shape[1],gray.shape[0]],word_sources=retained)
            # Component heuristics alone can mistake connected letter strokes
            # for a rule. Native word envelopes must also stop its tracking.
            horizontal, vertical = sections(gray, safe, groups,verified_crossings=coordinate_mode,
                repeated_dash_rules=full_pixel_footprints,audit_log=section_mask_evidence,
                whole_rulings=whole_rulings,reference_guides=measured_whole_rulings)
            mask=supported_word_mask(gray,groups)
            stroke_owners=base.get('complete_printed_strokes')
            if complete_strokes and stroke_owners is None:
                from complete_stroke_guard import capture_strokes
                stroke_owners=capture_strokes(base,horizontal,vertical,gray,safe,
                    supported_word_mask(gray,groups,precise_ink=True))
            complete_guides=[None,None]
            if whole_stroke_proposals:
                from complete_stroke_guard import current_stroke_guides
                complete_guides=current_stroke_guides(base,stroke_owners)
            if step == 0 and initial_evidence_observer is not None:
                # Read-only diagnostic capture precedes every proposed content
                # correction; it cannot supply or replace material references.
                initial_evidence_observer(gray.copy(), safe.copy(),
                    [g.copy() for g in horizontal], [g.copy() for g in vertical],
                    [g.copy() for g in groups],
                    [g.copy() for g in additional] if unrecognized_print else [])
            attempts = []
            last_stroke_candidate=None
            last_stroke_checks=[]
            last_pixel_checks=[]
            def stroke_candidate_guard(result):
                nonlocal last_stroke_candidate,last_stroke_checks,last_pixel_checks
                if result is not last_stroke_candidate:
                    from complete_stroke_guard import audit_strokes
                    try:
                        last_stroke_checks=(audit_strokes(base,result,stroke_owners,
                            [gray.shape[1],gray.shape[0]]) if complete_strokes else [])
                        if full_pixel_footprints:
                            from material_pixel_footprints import audit as audit_pixels
                            last_pixel_checks=audit_pixels(result,pixel_owners)
                    except ERRORS as error:
                        attempts.append({'rejected':'complete printed stroke audit failed',
                                         'reason':str(error)})
                        return False
                    last_stroke_candidate=result
                valid=all(q['accepted'] for q in last_stroke_checks+last_pixel_checks)
                if not valid:
                    attempts.append({'rejected':('complete original pixel footprint'
                        if any(not q['accepted'] for q in last_pixel_checks)
                        else 'complete printed stroke bending'),
                        'strokes':last_stroke_checks,'pixel_footprints':last_pixel_checks})
                    # Rejected full grids need not stay alive while the next
                    # proposal allocates its field. Cache only a passing map.
                    last_stroke_candidate=None
                return valid
            candidate_guard=stroke_candidate_guard if complete_strokes or full_pixel_footprints else None
            choices = [
                ('continuation', lambda: normals.refine(
                    path, base, word_ink_groups=groups, audit_log=attempts,
                    words_mask=mask,
                    continue_established=True, joint_stock_iterations=5000, **OPTIONS)),
                ('isolated bend', lambda: normals.refine(
                    path, base, word_ink_groups=groups, audit_log=attempts,
                    words_mask=mask,
                    bend_only=True, joint_stock_iterations=5000, **OPTIONS)),
                ('shared row', lambda: shared.refine(
                    path, base, word_ink_groups=groups, audit_log=attempts,
                    words_mask=mask, **OPTIONS)),
                ('word row', lambda: word_step(
                    base, groups, words, old, horizontal, vertical,
                    gray.shape[1], gray.shape[0], attempts, source,gray)),
                ('horizontal section', lambda: horizontal_step(
                    path, base, groups, gray, safe, horizontal, attempts)),
                ('vertical section', lambda: normals.refine(
                    path, base, word_ink_groups=groups, audit_log=attempts,
                    words_mask=mask, **OPTIONS)),
            ]
            if coordinate_mode:
                choices=[('coordinate word',lambda:coordinate_word_step(base,groups,(gray,safe,source,old),horizontal,vertical,attempts,candidate_guard=candidate_guard,measured_motion_owners=measured_motion_owners)),
                         ('coordinate horizontal section',lambda:coordinate_horizontal_step(path,base,groups,(gray,safe,source,old),horizontal,vertical,attempts,
                              measured_ink_bands=measured_ink_bands,free_tangent_rules=free_tangent_rules,candidate_guard=candidate_guard,complete_guides=complete_guides[0],complete_endpoint_anchors=complete_endpoint_anchors)),
                         ('coordinate vertical section',lambda:coordinate_vertical_step(path,base,groups,(gray,safe,source,old),horizontal,vertical,attempts,
                              word_translation_links=links,word_owner_envelopes=envelopes,
                              measured_ink_bands=measured_ink_bands,free_tangent_rules=free_tangent_rules,candidate_guard=candidate_guard,complete_guides=complete_guides[1],complete_endpoint_anchors=complete_endpoint_anchors))]
                if coordinate_shared_rows:
                    choices.insert(1,('coordinate shared word and rule row',
                        lambda:coordinate_shared_row_step(path,base,groups,words,native_count,
                              (gray,safe,source,old),horizontal,vertical,attempts,candidate_guard=candidate_guard)))
                if coordinate_word_rows:
                    # Establish supported row/column rules before fixing a
                    # complete text row, matching the retained recovery order.
                    choices.append(('coordinate word row',lambda:coordinate_word_row_step(
                        base,groups,words,(gray,safe,source,old),horizontal,vertical,attempts,
                        candidate_guard=candidate_guard,full_pixel_footprints=full_pixel_footprints)))
                if whole_rulings:
                    # A complete text row supplies one ordered baseline. Avoid
                    # independent glyph-pose proposals on this paper-only path.
                    choices=[choice for choice in choices if choice[0]!='coordinate word']
            accepted = False
            for name, propose in choices:
                started=time.monotonic();first_check=len(attempts)
                try:
                    result = propose()
                    if result is base:
                        continue
                    stroke_checks=[]
                    if complete_strokes:
                        if not stroke_candidate_guard(result):continue
                        stroke_checks=last_stroke_checks
                    field=rows.accepted.accepted.grid_of(result)
                    if result.get('coordinate_transform',False):
                        word_checks,text_checks,joins,ruler_checks=coordinate_stage_audit(base,result,old,groups,horizontal,vertical,gray.shape[1],gray.shape[0],measurement=(gray,safe))
                        if not all(q['accepted'] for q in ruler_checks):
                            attempts.append({'rejected':'established rule normal','rules':ruler_checks})
                            continue
                    else:
                        ruler_checks=[]
                        word_checks = word_geometry_audit(
                            old,field,groups,gray.shape[1],gray.shape[0],
                            fixed=name=='word row',
                            rotated=result.get('rigid_word_row_audit',{}).get(
                                'independent_word_owners',[]))
                    if not all(q['accepted'] for q in word_checks):
                        attempts.append({'rejected': 'independent word shape',
                                         'words': word_checks})
                        continue
                    if not result.get('coordinate_transform',False):
                        text_checks=established_text_audit(base,old,field,[gray.shape[1],gray.shape[0]])
                    if not all(q['accepted'] for q in text_checks):
                        attempts.append({'rejected':'established text row','rows':text_checks})
                        continue
                    if not result.get('coordinate_transform',False):
                        joins=join_audit(old,field,horizontal,vertical,gray.shape[1],gray.shape[0])
                    if not all(q['accepted'] for q in joins):
                        attempts.append({'rejected': 'physical rule join', 'joins': joins})
                        continue
                    carried_checks=(retained_word_audit(base,result,retained,
                                    [gray.shape[1],gray.shape[0]],references=references)
                                    if coordinate_mode or unrecognized_print else [])
                    if not all(q['accepted'] for q in carried_checks):
                        attempts.append({'rejected':'retained earlier word shape',
                                         'words':carried_checks})
                        continue
                    missed_checks=(retained_word_audit(base,result,
                        [np.asarray(g) for g in missed_clouds],
                        [gray.shape[1],gray.shape[0]],
                        references=[np.asarray(g) for g in missed_references],limit_px=.03)
                        if unrecognized_print else [])
                    if not all(q['accepted'] for q in missed_checks):
                        attempts.append({'rejected':'retained unrecognized print shape',
                                         'words':missed_checks})
                        continue
                    column_checks=column_geometry_audit(result,columns,
                                        [gray.shape[1],gray.shape[0]],
                                        word_bindings=bindings) if coordinate_mode else []
                    if not all(q['accepted'] for q in column_checks):
                        attempts.append({'rejected':'working text column alignment',
                                         'columns':column_checks})
                        continue
                    history.append({'step': step+1, 'stage': name, 'checks': attempts,
                                    'physical_rule_joins': joins,
                                    'established_text_rows':text_checks,
                                    'established_rule_normals':ruler_checks,
                                    'independent_words': word_checks,
                                    'retained_words':carried_checks,
                                    'unrecognized_print':missed_checks,
                                    'complete_printed_strokes':stroke_checks,
                                    **({'complete_pixel_footprints':last_pixel_checks}
                                       if full_pixel_footprints else {}),
                                    'working_text_columns':column_checks})
                    if complete_strokes:
                        result['complete_printed_strokes']=stroke_owners
                    if coordinate_mode or unrecognized_print:
                        result['retained_word_clouds']=[g.tolist() for g in retained]
                        result['retained_word_references']=[g.tolist() for g in references]
                        if full_pixel_footprints:
                            result['retained_pixel_footprints']=pixel_owners
                            result['repeated_dash_rule_evidence']=section_mask_evidence
                        if unrecognized_print:
                            result['unrecognized_print_clouds']=missed_clouds
                            result['unrecognized_print_references']=missed_references
                            result['unrecognized_print_audit']=missed_audit
                        if coordinate_mode:
                            result['established_text_columns']=columns
                        if coordinate_mode and partition_words:
                            result['printed_word_rule_partition']={
                                'roles':word_roles,'word_point_accounting':role_audit,
                                'earlier_roles':base.get('printed_word_rule_partition')}
                            result['printed_word_component_ownership']={
                                **component_roles,'earlier_roles':base.get('printed_word_component_ownership')}
                            if faint_roles is not None:
                                result['faint_word_component_recovery']={
                                    **faint_roles,'earlier_roles':base.get('faint_word_component_recovery')}
                            if bracketed_rules:
                                result['bracketed_word_rule_partition']={
                                    'roles':bracketed_roles,'word_point_accounting':bracketed_audit,
                                    'input_basis':'After separated divider-fragment roles',
                                    'earlier_roles':base.get('bracketed_word_rule_partition')}
                            if supported_short_rules:
                                result['supported_short_word_rule_partition']={
                                    'roles':short_roles,'word_point_accounting':short_audit,
                                    'input_basis':'After separated divider and dense-band roles',
                                    'earlier_roles':base.get('supported_short_word_rule_partition')}
                            if whole_rulings:
                                result['repeated_ruling_word_partition']={
                                    'roles':repeated_roles,'word_point_accounting':repeated_audit,
                                    'input_basis':'Before immutable word and missed-print registration',
                                    'earlier_roles':base.get('repeated_ruling_word_partition')}
                    result['preserved_mesh'] = compact(base)
                    result['progressive_content_audit'] = history.copy()
                    publish(result, output)
                    base = result
                    accepted = True
                    break
                except ERRORS as error:
                    attempts.append({'stage':name,'rejected':type(error).__name__,'reason':str(error)})
                finally:
                    if attempt_observer is not None:
                        try:
                            attempt_observer({'step':step+1,'stage':name,
                                              'elapsed_seconds':time.monotonic()-started,
                                              'accepted':accepted,
                                              'checks':attempts[first_check:]})
                        except ERRORS:
                            # Diagnostic observers never authorize a proposal
                            # or discard an already completed checkpoint.
                            pass
            if not accepted:
                return base
        except ERRORS:
            return base
    return base


def solve_staged(path, output, working=None):
    base = rows.accepted.solve(path, working)
    publish(base, output)
    return refine_staged(path, output, base)


def solve_coordinate_staged(path,output,working=None,*,word_provider=request_words,
                            max_steps=12,attempt_observer=None,localization_halo=1,
                            bracketed_rules=False,measured_ink_bands=False,
                            supported_short_rules=False,coordinate_word_rows=False,
                            free_tangent_rules=False,physical_column_origins=False,
                            coordinate_shared_rows=False,unrecognized_print=False,
                            original_plane=False,initial_evidence_observer=None,
                            complete_strokes=False,whole_stroke_proposals=False,
                            measured_motion_owners=False,complete_endpoint_anchors=False):
    """Optional original-image path; normalize supported print before locking words.

    The accepted receipt/card path remains the default base. The optional
    original-plane path captures native and missed print before any content
    warp, and omits unguarded preparation. Non-grid pages retain their existing
    fallback. No cached page mesh or fixture supplies production geometry.
    """
    if original_plane and not (measured_ink_bands and unrecognized_print):
        raise ValueError('Original-plane correction requires complete measured print protection')
    if complete_endpoint_anchors and not (whole_stroke_proposals and complete_strokes):
        raise ValueError('Endpoint anchors require complete original stroke proposals')
    # Original owners must precede the first content deformation. Registering
    # words after a broad warp makes its collateral shape changes the reference
    # that every later pass then preserves. Keep the established shipping path
    # unchanged; this optional printed-grid path starts with pure perspective.
    base=({'rows':2,'columns':2,
           'grid':[[0.,0.],[1.,0.],[0.,1.],[1.,1.]],
          'method':'Original perspective plane; no prior content warp'}
          if original_plane else rows.accepted.solve(path,working))
    if not original_plane:
        publish(base,output)
    gray,safe,_,_=rows.current_measurement(path,base,exact_layers=True)
    horizontal,vertical=sections(gray,safe,verified_crossings=True)
    if not printed_grid(horizontal,vertical):
        if original_plane:
            base=rows.accepted.solve(path,working)
            publish(base,output)
        return base
    if original_plane:
        publish(base,output)
    preparation=([] if original_plane else
                 [('supported row preparation',rows.refine),
                  ('complete faint curve preparation',frame.refine)])
    for name,prepare in preparation:
        started=time.monotonic()
        try:
            result=prepare(path,base)
            accepted=result is not base
            if accepted:
                publish(result,output)
                base=result
            if attempt_observer is not None:
                attempt_observer({'step':0,'stage':name,
                                  'elapsed_seconds':time.monotonic()-started,
                                  'accepted':accepted,'checks':[]})
        except ERRORS:
            # Preparation failure cannot replace the last completed map.
            continue
    return refine_staged(path,output,base,word_provider=word_provider,
                         max_steps=max_steps,attempt_observer=attempt_observer,
                         coordinate_mode=True,crop_words=True,partition_words=True,
                         localization_halo=localization_halo,bracketed_rules=bracketed_rules,
                         measured_ink_bands=measured_ink_bands,
                         supported_short_rules=supported_short_rules,
                         coordinate_word_rows=coordinate_word_rows,
                         free_tangent_rules=free_tangent_rules,
                         physical_column_origins=physical_column_origins,
                         coordinate_shared_rows=coordinate_shared_rows,
                         unrecognized_print=unrecognized_print,
                         initial_evidence_observer=initial_evidence_observer,
                         complete_strokes=complete_strokes,
                         whole_stroke_proposals=whole_stroke_proposals,
                         measured_motion_owners=measured_motion_owners,
                         complete_endpoint_anchors=complete_endpoint_anchors)


if __name__ == '__main__':
    solve_staged(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv)>3 else None)
