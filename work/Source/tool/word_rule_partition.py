"""Partition repeated, separated divider dashes from native word ink.

This changes ink roles, never pixels. Every input point remains in either its
original glyph group or a returned printed-rule group with its original index.
Ambiguous punctuation, single fragments and touching glyphs remain word owners.
"""
import cv2
import numpy as np


def partition_measured_repeated_rulings(groups, evidence, width, height):
    """Separate a real repeated dash accidentally included in an OCR word.

    Unlike proximity-only fragment tests, both independent raster profiles of
    the complete ruling must locate this same separated native component.
    Every input point retains its exact index and either glyph or rule role.
    Call only before carrying/capturing the new immutable word references.
    """
    if min(width,height)<2:
        raise ValueError('Repeated-ruling roles need the measurement raster')
    basis=np.array([width-1,height-1]);scale=max(width,height)/1280
    proofs=[q for q in evidence if q.get('thresholds')==[6,16] and
            len(q.get('components',[]))>=8 and
            len(q.get('independently_supported_components',[]))>=8 and
            np.ptp([p['center_px'][0] for p in q['components']])>.3*width]
    clean=[];roles=[];audit=[]
    def coverage(pixels,components):
        covered=np.zeros(len(pixels),bool)
        for item in components:
            x,y,cw,ch=item['bounds_px']
            covered|=((pixels>=[x-1,y-1])&(pixels<=[x+cw,y+ch])).all(1)
        return float(covered.mean())
    for word_index,value in enumerate(groups):
        group=np.asarray(value,dtype=float)
        if (group.ndim!=2 or group.shape[1:]!=(2,) or not len(group)
                or not np.isfinite(group).all() or (group<0).any() or (group>1).any()):
            raise ValueError('Invalid repeated-ruling word material')
        pixels=np.rint(group*basis).astype(int);low=pixels.min(0);high=pixels.max(0)
        mask=np.zeros((high[1]-low[1]+1,high[0]-low[0]+1),np.uint8)
        local=pixels-low;mask[local[:,1],local[:,0]]=1
        _,labels,stats,_=cv2.connectedComponentsWithStats(mask)
        membership=labels[local[:,1],local[:,0]]
        glyph_height=max(stats[1:,cv2.CC_STAT_HEIGHT],default=0)
        tall=np.flatnonzero(stats[:,cv2.CC_STAT_HEIGHT]>=.7*glyph_height)
        tall=tall[tall!=0];remove=np.zeros(len(group),bool)
        if 8*scale<=glyph_height<=38*scale and len(tall):
            for label,(x,y,cw,ch,area) in enumerate(stats[1:],1):
                if (label in tall or cw<5*scale or ch<2*scale or ch>8*scale
                        or cw<1.45*ch or area/(cw*ch)<.4):continue
                gaps=[max(stats[t,1]-(y+ch),y-(stats[t,1]+stats[t,3]),0) for t in tall]
                if min(gaps)<2*scale:continue
                ids=np.flatnonzero(membership==label);actual=pixels[ids]
                matches=[i for i,q in enumerate(proofs)
                         if coverage(actual,q['components'])>=.85 and
                         coverage(actual,q['independently_supported_components'])>=.5]
                if len(matches)!=1:continue
                remove[ids]=True
                roles.append(dict(word_index=word_index,point_indices=ids.tolist(),
                    points=group[ids].tolist(),ruling_evidence_index=matches[0],
                    components=1,evidence='Same separated native dash in both full-ruling contrast profiles'))
        clean.append(group[~remove].copy())
        audit.append(dict(word_index=word_index,input_points=len(group),
            glyph_point_indices=np.flatnonzero(~remove).tolist(),
            glyph_points=int((~remove).sum()),rule_points=int(remove.sum()),
            all_points_accounted=bool((~remove).sum()+remove.sum()==len(group))))
    return clean,roles,audit


def partition_rule_fragments(groups,guides,width,height,*,pixel_basis='samples'):
    if min(width,height)<2 or pixel_basis not in ('samples','edges'):
        raise ValueError('Word partition needs a valid raster coordinate basis')
    scale=max(width,height)/1280
    basis=np.array([width-1,height-1] if pixel_basis=='samples' else [width,height])
    rules=[]
    for guide in guides:
        g=np.asarray(guide,dtype=float)
        if (g.ndim!=2 or g.shape[1]!=2 or len(g)<48 or not np.isfinite(g).all()
                or np.any(np.diff(g[:,0])<=0) or np.ptp(g[:,0])<.04):continue
        rules.append(g)
    clean=[];printed=[];audit=[]
    for word_index,value in enumerate(groups):
        g=np.asarray(value,dtype=float)
        if g.ndim!=2 or g.shape[1]!=2 or not len(g) or not np.isfinite(g).all() or (g<0).any() or (g>1).any() or (pixel_basis=='edges' and (g>=1).any()):
            raise ValueError('Invalid word ink geometry')
        pixels=(np.rint(g*basis) if pixel_basis=='samples' else np.floor(g*basis)).astype(int)
        lo,hi=pixels.min(0),pixels.max(0)
        mask=np.zeros((hi[1]-lo[1]+1,hi[0]-lo[0]+1),np.uint8)
        xy=pixels-lo;mask[xy[:,1],xy[:,0]]=1
        _,labels,stats,_=cv2.connectedComponentsWithStats(mask)
        membership=labels[xy[:,1],xy[:,0]]
        glyph_height=max(stats[1:,cv2.CC_STAT_HEIGHT],default=0)
        tall=np.flatnonzero(stats[:,cv2.CC_STAT_HEIGHT]>=.5*glyph_height)
        tall=tall[tall!=0]
        candidates={}
        if len(tall)>=3 and glyph_height<=height*.035:
            for label,(x,y,cw,ch,area) in enumerate(stats[1:],1):
                if cw<max(4,4*scale) or cw<2*ch or ch>=.3*glyph_height:continue
                gaps=[max(stats[t,1]-(y+ch),y-(stats[t,1]+stats[t,3]),0) for t in tall]
                if min(gaps)<2*scale:continue
                ids=np.flatnonzero(membership==label);points=g[ids]
                # This bounded extension is membership evidence only. It never
                # creates a drive trace or invents ink between observed dashes.
                margin=min(.05,4*glyph_height/(width-1))
                choices=[]
                for rule_index,r in enumerate(rules):
                    if points[:,0].min()<r[0,0]-margin or points[:,0].max()>r[-1,0]+margin:continue
                    errors=(points[:,1]-np.interp(points[:,0],r[:,0],r[:,1]))*height
                    if abs(errors).max()>3*scale:continue
                    choices.append((float(abs(errors).max()),rule_index,float(np.median(errors))))
                if choices:
                    _,rule_index,normal=min(choices)
                    candidates.setdefault(rule_index,[]).append((ids,cw,ch,normal))
        remove=np.zeros(len(g),bool);roles=[]
        for rule_index,parts in candidates.items():
            if len(parts)<2:continue
            if (np.ptp([p[3] for p in parts])>2.5*scale or
                    max(p[1] for p in parts)>2*min(p[1] for p in parts)):
                continue
            ids=np.concatenate([p[0] for p in parts])
            if len(ids)>.15*len(g):continue
            remove[ids]=True
            roles.append({'word_index':word_index,'rule_index':rule_index,
                          'point_indices':ids.tolist(),'points':g[ids].tolist(),
                          'components':len(parts)})
        clean.append(g[~remove].copy())
        printed.extend(roles)
        audit.append({'word_index':word_index,'glyph_point_indices':np.flatnonzero(~remove).tolist(),
                      'glyph_points':int((~remove).sum()),'rule_points':int(remove.sum()),
                      'input_points':len(g),'all_points_accounted':bool(len(g[~remove])+remove.sum()==len(g))})
    return clean,printed,audit


def partition_bracketed_rules(groups, guides, width, height):
    """Separate a dense thin ruling band observed on both sides of a word.

    This is distinct from repeated divider dashes: unequal thin components can
    form one nearly continuous band. Multiple full-height letters and at least
    two pixels of normal stock must separate that band from the glyphs. Both
    measured flanks are mandatory; no pixel is invented or discarded.
    """
    if min(width, height) < 2:
        raise ValueError('Rule ownership needs a valid raster')
    scale = max(width, height) / 1280
    basis = np.array([width - 1, height - 1])
    measured = []
    for value in guides:
        guide = np.asarray(value, dtype=float)
        if (guide.ndim == 2 and guide.shape[1] == 2 and len(guide) >= 48
                and np.isfinite(guide).all() and np.all(np.diff(guide[:, 0]) > 0)
                and np.ptp(guide[:, 0]) >= .04):
            measured.append(guide)
    output, roles, audit = [], [], []
    for word_index, value in enumerate(groups):
        group = np.asarray(value, dtype=float)
        if (group.ndim != 2 or group.shape[1] != 2 or not len(group)
                or not np.isfinite(group).all() or (group < 0).any()
                or (group > 1).any()):
            raise ValueError('Invalid word ink geometry')
        pixels = np.rint(group * basis).astype(int)
        low, high = pixels.min(0), pixels.max(0)
        local = pixels - low
        mask = np.zeros((high[1] - low[1] + 1, high[0] - low[0] + 1), np.uint8)
        mask[local[:, 1], local[:, 0]] = 1
        _, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        membership = labels[local[:, 1], local[:, 0]]
        font_height = max(stats[1:, cv2.CC_STAT_HEIGHT], default=0)
        tall = np.flatnonzero(stats[:, cv2.CC_STAT_HEIGHT] >= .5 * font_height)
        tall = tall[tall != 0]
        parts = []
        if len(tall) >= 3 and 8 * scale <= font_height <= height * .035:
            for label, (x, y, cw, ch, area) in enumerate(stats[1:], 1):
                if (cw < max(4, 4 * scale) or cw < 4 * ch
                        or ch > min(6 * scale, font_height / 3)):
                    continue
                gaps = [max(stats[t, 1] - (y + ch),
                            y - (stats[t, 1] + stats[t, 3]), 0) for t in tall]
                if min(gaps) < 2 * scale:
                    continue
                component = labels[y:y + ch, x:x + cw] == label
                counts = component.sum(0)
                active = np.flatnonzero(counts)
                profiles = [np.flatnonzero(component[:, col]) for col in active]
                if (len(active) / cw < .95 or
                        np.mean([len(q) == np.ptp(q) + 1 for q in profiles]) < .95):
                    continue
                ids = np.flatnonzero(membership == label)
                parts.append((ids, float(np.median(pixels[ids, 1]))))
        remove = np.zeros(len(group), bool)
        used = set()
        for _, center in parts:
            ids = np.concatenate([q for q, y in parts if abs(y - center) <= 6 * scale])
            if any(int(index) in used for index in ids):
                continue
            points = group[ids]
            px = pixels[ids]
            first, last = int(px[:, 0].min()), int(px[:, 0].max())
            columns = np.unique(px[:, 0])
            if (last - first < max(48 * scale, .04 * width)
                    or len(columns) / (last - first + 1) < .95
                    or np.max(np.diff(columns), initial=0) > 3 * scale):
                continue
            flanks = None
            for i, guide in enumerate(measured):
                left = guide[guide[:, 0] * basis[0] < first]
                right = guide[guide[:, 0] * basis[0] > last]
                if (len(left) >= 48 and len(right) >= 48
                        and np.max(abs(points[:, 1] - np.interp(
                            points[:, 0], guide[:, 0], guide[:, 1]))) * basis[1] <= 3 * scale):
                    flanks = (i, i)
                    break
            if flanks is None:
                for i, left in enumerate(measured):
                    left_gap = first - left[-1, 0] * basis[0]
                    if not -3 * scale <= left_gap <= 8 * scale:
                        continue
                    for j, right in enumerate(measured):
                        right_gap = right[0, 0] * basis[0] - last
                        if i == j or not -3 * scale <= right_gap <= 8 * scale:
                            continue
                        if right[0, 0] <= left[-1, 0]:
                            continue
                        expected = np.interp(points[:, 0],
                            [left[-1, 0], right[0, 0]], [left[-1, 1], right[0, 1]])
                        if np.max(abs(points[:, 1] - expected)) * basis[1] <= 3 * scale:
                            flanks = (i, j)
                            break
                    if flanks is not None:
                        break
            if flanks is None:
                continue
            remove[ids] = True
            used.update(int(index) for index in ids)
            roles.append({'word_index': word_index, 'point_indices': ids.tolist(),
                          'points': points.tolist(), 'left_guide_index': flanks[0],
                          'right_guide_index': flanks[1],
                          'evidence': 'Separated dense thin band with two measured flanks'})
        output.append(group[~remove].copy())
        audit.append({'word_index': word_index,
                      'glyph_point_indices': np.flatnonzero(~remove).tolist(),
                      'glyph_points': int((~remove).sum()),
                      'rule_points': int(remove.sum()), 'input_points': len(group),
                      'all_points_accounted': bool((~remove).sum() + remove.sum() == len(group))})
    return output, roles, audit


def partition_supported_short_sections(groups, guides, vertical_guides, width, height):
    """Separate three short divider pieces beside a supported frame junction.

    A short first-cell label can have only two connected glyph bodies. Require
    three similar separated fragments, a long observed row and two real flanks.
    A short flank additionally needs an independently observed frame crossing.
    Every point retains its exact index and glyph or printed-rule ownership.
    """
    if min(width,height)<2:
        raise ValueError('Short row ownership needs a valid raster')
    basis=np.array([width-1,height-1]);scale=max(width,height)/1280
    def valid_trace(value,min_points,min_span):
        trace=np.asarray(value,dtype=float)
        return (trace.ndim==2 and trace.shape[1]==2 and len(trace)>=min_points
                and np.isfinite(trace).all() and np.all(np.diff(trace[:,0])>0)
                and np.ptp(trace[:,0])>=min_span)
    horizontal=[np.asarray(g) for g in guides if valid_trace(g,256,.3)]
    vertical=[np.asarray(g) for g in vertical_guides if valid_trace(g,128,.15)]

    def crossing(guide,begin,end):
        for trace in vertical:
            center=float(np.median(guide[:,1]))
            for _ in range(3):
                x=float(np.interp(center,trace[:,0],trace[:,1]))
                center=float(np.interp(x,guide[:,0],guide[:,1]))
            if (begin<=x*basis[0]<=end and trace[0,0]<=center<=trace[-1,0]
                    and abs(x-np.interp(center,trace[:,0],trace[:,1]))*basis[0]<=3*scale):
                return True
        return False

    output,roles,audit=[],[],[]
    for word_index,value in enumerate(groups):
        group=np.asarray(value,dtype=float)
        if (group.ndim!=2 or group.shape[1]!=2 or not len(group)
                or not np.isfinite(group).all() or (group<0).any() or (group>1).any()):
            raise ValueError('Invalid short-section word geometry')
        pixels=np.rint(group*basis).astype(int);low,high=pixels.min(0),pixels.max(0)
        local=pixels-low;mask=np.zeros((high[1]-low[1]+1,high[0]-low[0]+1),np.uint8)
        mask[local[:,1],local[:,0]]=1
        _,labels,stats,_=cv2.connectedComponentsWithStats(mask)
        membership=labels[local[:,1],local[:,0]]
        font_height=max(stats[1:,cv2.CC_STAT_HEIGHT],default=0)
        tall=np.flatnonzero(stats[:,cv2.CC_STAT_HEIGHT]>=.5*font_height)
        tall=tall[tall!=0];parts=[]
        if len(tall)>=2 and 8*scale<=font_height<=height*.035:
            for label,(x,y,cw,ch,area) in enumerate(stats[1:],1):
                if (cw<4*scale or cw<2*ch or ch>=.3*font_height):continue
                gaps=[max(stats[t,1]-(y+ch),y-(stats[t,1]+stats[t,3]),0) for t in tall]
                if min(gaps)<2*scale:continue
                component=labels[y:y+ch,x:x+cw]==label
                active=np.flatnonzero(component.sum(0))
                if (len(active)/cw<.8 or
                        np.mean([len(q)==np.ptp(q)+1 for q in
                                 [np.flatnonzero(component[:,col]) for col in active]])<.9):continue
                ids=np.flatnonzero(membership==label)
                parts.append((ids,int(cw),float(np.median(pixels[ids,1]))))
        remove=np.zeros(len(group),bool)
        for _,__,center in parts:
            selected=[q for q in parts if abs(q[2]-center)<=2.5*scale]
            if len(selected)<3 or max(q[1] for q in selected)>2*min(q[1] for q in selected):continue
            ids=np.concatenate([q[0] for q in selected])
            if remove[ids].any() or len(ids)>.2*len(group):continue
            px=pixels[ids];first,last=px[:,0].min(),px[:,0].max();columns=np.unique(px[:,0])
            if (last-first<24*scale or len(columns)/(last-first+1)<.7
                    or np.max(np.diff(columns),initial=0)>8*scale):continue
            points=group[ids];support=None
            for rule_index,guide in enumerate(horizontal):
                left=guide[guide[:,0]*basis[0]<first];right=guide[guide[:,0]*basis[0]>last]
                if (len(left)<max(4,4*scale) or len(right)<max(4,4*scale)
                        or max(len(left),len(right))<48):continue
                # Compare in the measured pixel lattice. A true three-pixel
                # difference must not become 3.00000000000001 after subtracting
                # normalized coordinates and scaling the difference back.
                error=abs(px[:,1]-np.interp(px[:,0],guide[:,0]*basis[0],
                                           guide[:,1]*basis[1])).max()
                if error>3*scale:continue
                if len(left)<48 and not crossing(guide,guide[0,0]*basis[0],first):continue
                if len(right)<48 and not crossing(guide,last,guide[-1,0]*basis[0]):continue
                support=rule_index;break
            if support is None:continue
            remove[ids]=True
            roles.append({'word_index':word_index,'rule_index':support,'components':len(selected),
                          'point_indices':ids.tolist(),'points':points.tolist(),
                          'evidence':'Three separated short fragments on a long measured row with supported flanks'})
        output.append(group[~remove].copy())
        audit.append({'word_index':word_index,'glyph_point_indices':np.flatnonzero(~remove).tolist(),
                      'glyph_points':int((~remove).sum()),'rule_points':int(remove.sum()),
                      'input_points':len(group),'all_points_accounted':bool((~remove).sum()+remove.sum()==len(group))})
    return output,roles,audit
