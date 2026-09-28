"""Apply ordered whole-word row constraints to a measured vertical rule."""
import cv2
import numpy as np

import section_row_candidate as rows
from printed_rule_sections import measure_sections, word_ink_mask
from word_rule_membership import separate_rule_ink
from isolated_rule_bends import before_stable_tail, anchored_remainder


def supported_rule_word_mask(words, ink_groups, guides):
    """Allow a supported spine past false frame-as-letter summaries.

    Only a narrow continuation of a long measured rule may replace the
    heuristic mask. Every independent whole-word envelope still wins there.
    The actual section tracker must supply ink, stock and junction evidence.
    """
    h,w=words.shape
    independent=np.zeros_like(words)
    for group in ink_groups:
        group=np.asarray(group)
        if not len(group):continue
        lo=np.floor(group.min(0)*[w,h]).astype(int)
        hi=np.ceil(group.max(0)*[w,h]).astype(int)
        independent[max(0,lo[1]):min(h,hi[1]+1),max(0,lo[0]):min(w,hi[0]+1)]=1
    independent=cv2.dilate(independent,np.ones((3,3),np.uint8))
    corridor=np.zeros_like(words)
    for guide in guides:
        if np.ptp(guide[:,0])<.3:continue
        lo=max(0,guide[0,0]-.04);hi=min(1,guide[-1,0]+.04)
        y=np.arange(max(0,int(lo*(h-1))),min(h,int(hi*(h-1))+1))
        t=y/(h-1);x=np.interp(t,guide[:,0],guide[:,1])
        for before in (True,False):
            samples=guide[:min(20,len(guide))] if before else guide[-min(20,len(guide)):]
            slope=float(np.clip(np.polyfit(samples[:,0],samples[:,1],1)[0],-.45,.45))
            edge=guide[0] if before else guide[-1]
            outside=t<edge[0] if before else t>edge[0]
            x[outside]=edge[1]+slope*(t[outside]-edge[0])
        center=np.rint(x*(w-1)).astype(int)
        radius=max(2,round(3*max(h,w)/1280))
        for offset in range(-radius,radius+1):
            xx=center+offset;valid=(xx>=0)&(xx<w);corridor[y[valid],xx[valid]]=1
    return np.where(corridor,independent,np.maximum(words,independent)).astype(np.uint8)


def refine(path, base, *, word_ink_groups=None, audit_log=None,
           separate_structural_ink=False,exact_ordering=False,independent_word_owners=False,
           stock_relaxation=0,stock_coordinate_relaxation=0,precise_geometry=False,
           bend_only=False,joint_stock_iterations=0,continue_established=False,
           words_mask=None,preserve_neighbouring_rules=False):
    if not word_ink_groups:
        return base
    gray, safe, source, old = rows.current_measurement(path, base)
    h, w = gray.shape
    words = word_ink_mask(gray) if words_mask is None else np.asarray(words_mask,dtype=np.uint8)
    horizontal = measure_sections(gray, safe, word_mask=words)
    crossing = np.zeros_like(gray)
    for guide in horizontal:
        cv2.polylines(crossing, [np.rint(guide * [w-1,h-1]).astype(np.int32)], False, 1, 9)
    vertical = measure_sections(gray.T, safe.T, word_mask=words.T, crossing_mask=crossing.T)
    if (sum(np.ptp(g[:,0]) > .3 for g in horizontal) < 3 or
            sum(np.ptp(g[:,0]) > .2 for g in vertical) < 3):
        return base
    tracking_words=supported_rule_word_mask(words,word_ink_groups,vertical)
    vertical=measure_sections(gray.T,safe.T,word_mask=tracking_words.T,crossing_mask=crossing.T)
    # Transpose geometry only after measuring words in their reading direction.
    # Each independent word stays one rigid ink band in the ordered solver.
    transposed = {**base, 'established_rulers': [
        {**r,'normal_axis':1-r['normal_axis'],
         'points':np.asarray(r['points'])[:,::-1].tolist()}
        for r in base.get('established_rulers',[])],
        'established_text_rows': [{**r,'points':np.asarray(r['points'])[:,::-1].tolist()}
                                 for r in base.get('established_text_rows',[])]}
    measurement = (gray.T, safe.T, source.T, old.transpose(1,0,2)[:,:,::-1])
    ink_groups = [np.asarray(g)[:,::-1] for g in word_ink_groups if len(g)]
    shapes = rows.accepted.accepted.circular_landmarks(source) + rows.p.reference.barcode_rulers(source)
    guards = [g[:,::-1] for g in shapes]
    candidates=[]
    anchors = ([rows.precise_inverse(measurement[3], np.asarray(r['points']))
                for r in transposed['established_rulers'] if r['normal_axis'] == 1]
               if continue_established else [])
    for guide in vertical:
        qualifies,evidence=rows.accepted.accepted.residual_evidence(guide,w)
        if continue_established:
            continuation = anchored_remainder(guide, anchors, h, w)
            if continuation is not None:
                remaining, target = continuation
                excursion = float(abs(remaining[:, 1] - target).max() * w)
                candidates.append((excursion, remaining, target))
        elif qualifies and np.ptp(guide[:,0]) > .1:
            candidates.append((evidence['excursion_px'],guide))
    candidates.sort(key=lambda q:-q[0])
    for candidate in candidates:
        _,guide = candidate[:2]
        audit=[]
        target_guide = (np.c_[guide[[0,-1],0],np.full(2,candidate[2])]
                        if continue_established else None)
        if bend_only and not continue_established:
            bend = before_stable_tail(guide, max(h, w))
            if bend is None:
                continue
            guide, target = bend
            target_guide = np.c_[guide[[0, -1], 0], np.full(2, target)]
        clean_words, rule_ink = separate_rule_ink(ink_groups,guide,h,w)
        proposal=rows.refine(path,transposed,guides=[guide],measurement=measurement,
                            target_guide=target_guide,
                            words_mask=words.T,word_ink_groups=clean_words,
                            include_wide_ink=True,align_to_existing=True,
                            extra_protected=guards,audit_log=audit,tangent_padding=.015,
                            separate_structural_ink=separate_structural_ink,
                            exact_ordering=exact_ordering,independent_word_owners=independent_word_owners,
                            stock_relaxation=stock_relaxation,
                            stock_coordinate_relaxation=stock_coordinate_relaxation,
                            precise_geometry=precise_geometry,
                            word_reading_axis=1,joint_stock_iterations=joint_stock_iterations,
                            preserve_neighbouring_rules=preserve_neighbouring_rules)
        if audit_log is not None:
            audit_log.append({'normal_axis':0,'position':guide.mean(0)[::-1].tolist(),
                              'checks':audit,'accepted':proposal is not transposed,
                              'excluded_supported_rule_ink_points':sum(len(g) for g in rule_ink)})
        if proposal is transposed:
            continue
        field=np.asarray(proposal['grid']).reshape(proposal['rows'],proposal['columns'],2)
        field=field.transpose(1,0,2)[:,:,::-1]
        regions=[[r[1],r[0],r[3],r[2]] for r in proposal['refinement_regions']]
        new_ruler=proposal['established_rulers'][-1]
        return {**base,'preserved_mesh':base,'refinement_regions':regions,
                'rows':field.shape[0],'columns':field.shape[1],
                'grid':field.reshape(-1,2).tolist(),
                'method':'Experimental whole-word vertical rule correction',
                'normal_row_audit':proposal['joint_row_audit'],
                'established_rulers':[*base.get('established_rulers',[]),
                    {'normal_axis':0,'points':np.asarray(new_ruler['points'])[:,::-1].tolist()}]}
    return base
