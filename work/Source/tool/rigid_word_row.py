"""Optional rigid word-row rotation with fixed surrounding print geometry.

The caller supplies native quadrilaterals and independent ink measured in the
same current raster. No image identity, page position or fixture file is used.
A failed proposal returns the original grid object.
"""
import numpy as np
import cv2
from local_receipt_mesh import sample, valid
from precise_mesh_inverse import inverse
from joint_stock_projection import project
from rigid_word_flow import backward

def find_rows(observations, pixel_size):
    """Associate adjacent native words by measured overlap and orientation."""
    scale = np.asarray(pixel_size, dtype=float)
    records = []
    for observation in observations:
        if observation.get('confidence', 0) < .75:
            continue
        quad = np.asarray(observation.get('points', []), dtype=float)
        if quad.shape != (4, 2) or not np.isfinite(quad).all():
            continue
        points = quad * scale
        low, high = points.min(0), points.max(0)
        edge = points[2] - points[3]
        if edge[0] <= 0 or high[1] <= low[1]:
            continue
        angle = float(np.arctan2(edge[1], edge[0]))
        records.append((quad, low, high, angle))
    chains = []
    for record in sorted(records, key=lambda item: item[1][0]):
        quad, low, high, angle = record
        choices = []
        for index, chain in enumerate(chains):
            _, before_low, before_high, before_angle = chain[-1]
            height = min(high[1] - low[1], before_high[1] - before_low[1])
            overlap = min(high[1], before_high[1]) - max(low[1], before_low[1])
            gap = low[0] - before_high[0]
            if (0 <= gap < 4 * height and overlap > .45 * height and
                    abs(angle - before_angle) < .006):
                choices.append((gap, index))
        if choices:
            chains[min(choices)[1]].append(record)
        else:
            chains.append([record])
    return [[item[0] for item in chain] for chain in chains if len(chain) >= 3]


def parallel_blocks(rows, pixel_size):
    """Associate nearby aligned rows using their measured word quadrilaterals."""
    scale=np.asarray(pixel_size,dtype=float);records=[]
    for row in rows:
        quads=[np.asarray(q,dtype=float) for q in row];points=np.vstack(quads)*scale
        angles=[np.arctan2(*((q[2]-q[3])*scale)[::-1]) for q in quads]
        height=float(np.median([np.linalg.norm((q[0]-q[3])*scale) for q in quads]))
        angle=float(np.median(angles))
        if abs(angle)<.004 or abs(angle)>.1 or np.ptp(angles)>.006:continue
        records.append((quads,points.min(0),points.max(0),height,angle,angles))
    chains=[]
    for record in sorted(records,key=lambda q:q[1][1]):
        quads,low,high,height,angle,angles=record;choices=[]
        for i,chain in enumerate(chains):
            _,a,z,h,before,previous=chain[-1]
            gap=low[1]-z[1];overlap=min(high[0],z[0])-max(low[0],a[0])
            span=min(high[0]-low[0],z[0]-a[0])
            all_angles=angles+[x for item in chain for x in item[5]]
            if (span>0 and overlap>.6*span and -.2*min(height,h)<gap<2*min(height,h)
                    and .5<height/h<2 and abs(angle-before)<.006
                    and np.ptp(all_angles)<.006):choices.append((abs(gap),i))
        if choices:chains[min(choices)[1]].append(record)
        else:chains.append([record])
    return [[quad for item in chain for quad in item[0]]
            for chain in chains if len(chain)>=2]

def refine(old, quads, groups, pixel_size, rulers, *, iterations=5000,
           use_flow=False,ink_padding=2,stock_padding=16,flow_body_padding=.5):
    return _refine(old, quads, groups, pixel_size, rulers, iterations=iterations,
                   use_flow=use_flow,ink_padding=ink_padding,stock_padding=stock_padding,
                   flow_body_padding=flow_body_padding)


def _refine(old, quads, groups, pixel_size, rulers, *, iterations=5000,
            minimum_owners=3, translation=(0, 0), ink_padding=2,
            stock_padding=16,use_flow=False,flow_constraint_padding=2,flow_body_padding=.5,
            precise_fixed_ink=False,rotation_center_px=None,explicit_owners=None):
    scale = np.asarray(pixel_size, dtype=float)
    quads = [np.asarray(q, dtype=float) for q in quads]
    groups = [np.asarray(g, dtype=float) for g in groups]
    if len(quads) < minimum_owners or not groups or scale.shape != (2,) or (not np.isfinite(scale).all()) or (scale <= 0).any() or any((q.shape != (4, 2) or not np.isfinite(q).all() for q in quads)) or any((g.ndim != 2 or g.shape[1] != 2 or (not len(g)) or (not np.isfinite(g).all()) for g in groups)):
        return (old, {'accepted': False, 'rejected': 'independent row evidence unavailable'})
    (w, h) = scale
    q = np.vstack(quads)
    center = np.median(q * scale, axis=0)
    if rotation_center_px is not None:
        center=np.asarray(rotation_center_px,dtype=float)
        if center.shape!=(2,) or not np.isfinite(center).all():
            return old,{'accepted':False,'rejected':'invalid rigid rotation center'}
    angles = [np.arctan2(*((quads[i][2] - quads[i][3]) * scale)[::-1]) for i in range(len(quads))]
    angle = float(np.median(angles))
    (c, s) = (np.cos(angle), np.sin(angle))
    rotation = np.array([[c, -s], [s, c]])
    if abs(angle) < 0.004 or abs(angle) > 0.1 or np.max(abs(np.asarray(angles) - angle)) > 0.006:
        return (old, {'accepted': False, 'rejected': 'row angle unsupported'})
    owners = []
    if explicit_owners is not None:
        if (not explicit_owners or any(not isinstance(i,(int,np.integer)) or
                isinstance(i,(bool,np.bool_)) or not 0<=i<len(groups) for i in explicit_owners)):
            return old,{'accepted':False,'rejected':'invalid measured word motion binding'}
        owners=list(explicit_owners)
    else:
        for i in range(len(quads)):
            (a, b) = (quads[i].min(0), quads[i].max(0))
            match = [j for (j, g) in enumerate(groups) if np.all(g.mean(0) >= a - [0.002, 0.002]) and np.all(g.mean(0) <= b + [0.002, 0.002])]
            if len(match) != 1:
                return (old, {'accepted': False, 'rejected': 'ambiguous independent word ownership'})
            owners += match
    owners = sorted(set(owners))
    if len(owners) < minimum_owners:
        return (old, {'accepted': False, 'rejected': 'independent word owners unavailable'})
    source = sample(old, np.vstack(groups))
    current = np.vstack(groups)
    off = np.r_[0, np.cumsum([len(g) for g in groups])]

    def destination(points):
        return ((points * scale - center) @ rotation + center + translation) / scale
    transformed = [destination(groups[i]) for i in owners]
    owned = np.vstack(transformed)
    lo = owned.min(0) - ink_padding / scale
    hi = owned.max(0) + ink_padding / scale
    region = np.r_[np.maximum(0, lo - stock_padding / scale), np.minimum(1, hi + stock_padding / scale)]
    (nr, nc) = old.shape[:2]
    uv = np.stack(np.meshgrid(np.linspace(0, 1, nc), np.linspace(0, 1, nr)), -1)
    target = ((uv * scale - center - translation) @ rotation.T + center) / scale
    delta = target - uv
    weight = np.ones((nr, nc))
    for axis in (0, 1):
        d = np.maximum(np.maximum(lo[axis] - uv[:, :, axis], uv[:, :, axis] - hi[axis]), 0) * scale[axis] / stock_padding
        t = np.clip(1 - d, 0, 1)
        weight *= t * t * (3 - 2 * t)
    fixed = []
    for (i, g) in enumerate(groups):
        if i in owners:
            continue
        (a, z) = (g.min(0) - ink_padding / scale, g.max(0) + ink_padding / scale)
        # The proposal grid is Cartesian. Search its actual coordinate arrays
        # to retain exactly the inclusive envelope, including floating-point
        # endpoints, without scanning the whole dense grid for each word.
        left = np.searchsorted(uv[0, :, 0], a[0], side='left')
        right = np.searchsorted(uv[0, :, 0], z[0], side='right')
        top = np.searchsorted(uv[:, 0, 1], a[1], side='left')
        bottom = np.searchsorted(uv[:, 0, 1], z[1], side='right')
        weight[top:bottom, left:right] = 0
        fixed.append(np.array([a, [z[0], a[1]], z, [a[0], z[1]]]))
    guards = [];trace_guards=[];cloud_guards=[]
    for rule in rulers:
        g=inverse(old, np.asarray(rule['points']));guards.append(g)
        is_cloud=rule.get('geometry')=='ink_cloud'
        (cloud_guards if is_cloud else trace_guards).append(g)
        corridor = np.zeros((nr, nc), np.uint8)
        pixels=np.rint(g * [nc - 1, nr - 1]).astype(np.int32)
        if is_cloud:
            # An unordered pixel body has no connecting segments. Drawing
            # chords through its missing ink would falsely block nearby words.
            good=np.all((pixels>=0)&(pixels<[nc,nr]),axis=1)
            pixels=pixels[good];corridor[pixels[:,1],pixels[:,0]]=1
            corridor=cv2.dilate(corridor,np.ones((5,5),np.uint8))
        else:
            cv2.polylines(corridor,[pixels],False,1,5)
        weight[corridor > 0] = 0
    candidate = sample(old, (uv + delta * weight[:, :, None]).reshape(-1, 2)).reshape(old.shape)
    flow_used=False
    if use_flow and not valid(candidate) and not np.any(translation):
        # Directly blending a rotation against fixed envelopes can fold a
        # narrow gap. A smooth local flow keeps those envelopes stationary
        # throughout the motion; the independent final gates remain identical.
        region=np.r_[np.maximum(0,np.minimum(np.vstack([groups[i] for i in owners]).min(0),owned.min(0))-(stock_padding+4)/scale),
                     np.minimum(1,np.maximum(np.vstack([groups[i] for i in owners]).max(0),owned.max(0))+(stock_padding+4)/scale)]
        active=np.all((uv>=region[:2])&(uv<=region[2:]),axis=2)
        coordinates=uv.copy()
        coordinates[active]=backward(uv[active],np.vstack([groups[i] for i in owners]),
                                     [] if precise_fixed_ink else fixed,
                                     trace_guards,center,angle,scale,padding=stock_padding,
                                     constraint_padding=flow_constraint_padding,
                                     body_padding=flow_body_padding,
                                     fixed_ink=[*cloud_guards,*([g for i,g in enumerate(groups)
                                        if i not in owners] if precise_fixed_ink else [])])
        candidate=sample(old,coordinates.reshape(-1,2)).reshape(old.shape)
        flow_used=True
    locked = fixed + [np.array([lo, [hi[0], lo[1]], hi, [lo[0], hi[1]]])]
    audit = []
    if not valid(candidate) and guards:
        candidate = project(candidate, np.array(locked), np.zeros(len(locked)), np.vstack(guards), np.vstack([sample(old, g) for g in guards]), region, iterations, audit)
    if not valid(candidate):
        return (old, {'accepted': False, 'rejected': 'mesh topology', 'stock_audit': audit})
    after = inverse(candidate, source)
    expected = current.copy()
    for i in owners:
        expected[off[i]:off[i + 1]] = destination(groups[i])
    error = np.abs((after - expected) * scale).max()
    rigid_error = max((float(np.ptp((after[off[i]:off[i + 1]] - expected[off[i]:off[i + 1]]) * scale, axis=0).max()) for i in range(len(groups))))
    guard_error = max((float(np.abs((inverse(candidate, np.asarray(rule['points'])) - g) * scale).max()) for (rule, g) in zip(rulers, guards)), default=0.0)
    report = {'accepted': bool(valid(candidate) and error < 0.1 and (rigid_error < 0.1) and (guard_error < 0.1)), 'independent_word_owners': owners, 'angle_degrees': angle * 180 / np.pi, 'region': region.tolist(), 'expected_rigid_geometry_error_px': float(error), 'maximum_shape_error_px': rigid_error, 'established_rule_geometry_shift_px': guard_error, 'stock_audit': audit}
    if use_flow:
        report['bounded_rigid_flow_used']=flow_used
        report['word_geometry_error_px']=[float(abs((after[off[i]:off[i+1]]-
                expected[off[i]:off[i+1]])*scale).max()) for i in range(len(groups))]
    return (candidate if report['accepted'] else old, report)


def glyph_baseline(gray, group, *, pixel_size=None):
    """Measure repeated glyph bodies inside an independently owned word."""
    h,w=gray.shape
    group=np.asarray(group)
    pixels=(np.floor(group*[w,h]) if pixel_size is None else
            np.rint(group*np.asarray(pixel_size))).astype(int)
    low=np.maximum(0,pixels.min(0));high=np.minimum([w-1,h-1],pixels.max(0))
    mask=np.zeros((high[1]-low[1]+1,high[0]-low[0]+1),np.uint8)
    good=np.all((pixels>=low)&(pixels<=high),axis=1);points=pixels[good]-low
    mask[points[:,1],points[:,0]]=1
    _,labels,stats,_=cv2.connectedComponentsWithStats(mask)
    anchors=[]
    scale=max(h,w)/1280
    for index,(x,y,cw,ch,area) in enumerate(stats[1:],1):
        if not (max(5,6*scale)<ch<38*scale and 2<cw<ch*2.4 and area>15*scale):
            continue
        component=labels[y:y+ch,x:x+cw]==index
        bottoms=[np.where(component[:,xx])[0][-1] for xx in range(cw) if component[:,xx].any()]
        anchors.append([x+low[0]+cw/2,y+low[1]+np.quantile(bottoms,.85),ch])
    if len(anchors)<6:
        return None
    anchors=np.asarray(anchors)
    height=np.median(anchors[:,2]);anchors=anchors[abs(anchors[:,2]-height)<height*.25]
    if len(anchors)<6 or np.ptp(anchors[:,0])<height*4:
        return None
    slope,intercept=np.polyfit(anchors[:,0],anchors[:,1],1)
    errors=anchors[:,1]-(slope*anchors[:,0]+intercept)
    explained=1-np.mean(errors**2)/max(np.var(anchors[:,1]),1e-12)
    return {'points':anchors[:,:2].tolist(),'angle':float(np.arctan(slope)),
            'slope':float(slope),'intercept':float(intercept),
            'scatter_px':float(np.std(errors)),'explained':float(explained),
            'excursion_px':float(abs(slope)*np.ptp(anchors[:,0]))}


def glyph_body_consensus(gray,group,*,pixel_size=None):
    """Require repeated six-glyph body evidence at two contrast levels.

    Stronger measured ink can separate letters joined only by faint blur.
    Body centers use both cap and baseline edges; the full faint word cloud
    remains the protected geometry and is never replaced by this subset.
    """
    h,w=gray.shape;scale=max(h,w)/1280
    group=np.asarray(group);basis=np.array([w,h] if pixel_size is None else pixel_size)
    pixels=(np.floor(group*basis) if pixel_size is None else np.rint(group*basis)).astype(int)
    low=np.maximum(0,pixels.min(0));high=np.minimum([w-1,h-1],pixels.max(0))
    mask=np.zeros((high[1]-low[1]+1,high[0]-low[0]+1),np.uint8)
    good=np.all((pixels>=low)&(pixels<=high),axis=1);points=pixels[good]-low;mask[points[:,1],points[:,0]]=1
    contrast=cv2.GaussianBlur(gray.astype(np.float32),(0,0),max(2,7*scale))-gray
    local=contrast[low[1]:high[1]+1,low[0]:high[0]+1]
    profiles=[]
    for threshold in (10,14,16,18,22):
        _,labels,stats,_=cv2.connectedComponentsWithStats(((mask>0)&(local>threshold)).astype(np.uint8))
        anchors=[]
        for index,(x,y,cw,ch,area) in enumerate(stats[1:],1):
            if not (max(5,6*scale)<ch<38*scale and 2<cw<ch*2.4 and area>15*scale):continue
            component=labels[y:y+ch,x:x+cw]==index
            bottoms=[np.where(component[:,xx])[0][-1] for xx in range(cw) if component[:,xx].any()]
            anchors.append([x+low[0]+cw/2,y+low[1],y+low[1]+np.quantile(bottoms,.85),ch])
        if len(anchors)<6:continue
        anchors=np.asarray(anchors);height=np.median(anchors[:,3]);anchors=anchors[abs(anchors[:,3]-height)<height*.25]
        if len(anchors)<6 or np.ptp(anchors[:,0])<height*4:continue
        center=anchors[:,1:3].mean(1);slope,intercept=np.polyfit(anchors[:,0],center,1)
        errors=center-slope*anchors[:,0]-intercept
        explained=1-np.mean(errors**2)/max(np.var(center),1e-12)
        top=np.polyfit(anchors[:,0],anchors[:,1],1)[0];bottom=np.polyfit(anchors[:,0],anchors[:,2],1)[0]
        excursion=abs(slope)*np.ptp(anchors[:,0]);scatter=np.std(errors)
        if explained<.7 or scatter>1.5 or excursion<2 or abs(np.arctan(top)-np.arctan(bottom))>.012:continue
        profiles.append({'threshold':threshold,'points':np.c_[anchors[:,0],center].tolist(),
                         'angle':float(np.arctan(slope)),'slope':float(slope),
                         'intercept':float(intercept),'explained':float(explained),
                         'scatter_px':float(scatter),'excursion_px':float(excursion),
                         'height':float(height)})
    for first in sorted(profiles,key=lambda item:-item['explained']):
        for second in profiles:
            if first is second or abs(first['angle']-second['angle'])>.006:continue
            a,b=np.asarray(first['points']),np.asarray(second['points'])
            correspondence=abs(a[:,0,None]-b[None,:,0]).min(1)<min(first['height'],second['height'])*.5
            if np.mean(correspondence)<.75:continue
            slope=float(np.mean([first['slope'],second['slope']]))
            return {**first,'slope':slope,'angle':float(np.arctan(slope)),
                    'independent_profiles':[first,second],
                    'profile_angle_difference':abs(first['angle']-second['angle'])}
    return None


def word_rotation_evidence(gray,group,*,pixel_size=None,body_consensus=False,
                           whole_word_envelope=False):
    """Use the same supported glyph evidence in preflight and refinement."""
    evidence=glyph_baseline(gray,group,pixel_size=pixel_size)
    if body_consensus and (evidence is None or evidence['explained']<.7 or
            evidence['scatter_px']>1.5 or evidence['excursion_px']<2):
        evidence=glyph_body_consensus(gray,group,pixel_size=pixel_size)
    if whole_word_envelope and (evidence is None or evidence['explained']<.7 or
            evidence['scatter_px']>1.5 or evidence['excursion_px']<2):
        from measured_word_envelope import envelope_consensus
        evidence=envelope_consensus(gray,group,pixel_size=pixel_size)
    if (evidence is None or evidence['explained']<.7 or
            evidence['scatter_px']>1.5 or evidence['excursion_px']<2):
        return None
    return evidence


def shared_motion_owners(gray,groups,index):
    """Bind poses through common observed dark raster cells, never box centers.

    These are motion links only: no owner, material point or reference is
    merged or removed. Sharing blank stock or merely occupying the same native
    envelope supplies no link. Existing material/shape audits remain required.
    """
    gray=np.asarray(gray)
    if gray.ndim!=2 or min(gray.shape)<2 or not np.isfinite(gray).all():
        raise ValueError('Word motion binding needs a finite measurement raster')
    h,w=gray.shape
    if not isinstance(index,(int,np.integer)) or isinstance(index,(bool,np.bool_)) or not 0<=index<len(groups):
        raise ValueError('Invalid selected word owner')
    contrast=cv2.GaussianBlur(gray.astype(np.float32),(0,0),max(2,7*max(h,w)/1280))-gray
    membership={};footprints=[]
    for owner,group in enumerate(groups):
        group=np.asarray(group,dtype=float)
        if group.ndim!=2 or group.shape[1:]!=(2,) or not len(group) or not np.isfinite(group).all():
            raise ValueError('Invalid measured word material')
        pixels=np.rint(group*[w-1,h-1]).astype(int)
        pixels=pixels[((pixels>=0)&(pixels<[w,h])).all(1)]
        pixels=pixels[contrast[pixels[:,1],pixels[:,0]]>6]
        keys=set((pixels[:,1]*w+pixels[:,0]).tolist());footprints.append(keys)
        for key in keys:membership.setdefault(key,[]).append(owner)
    if not footprints[index]:
        raise ValueError('Selected word has no observed dark raster samples')
    found={int(index)};todo=[int(index)]
    while todo:
        current=todo.pop()
        for key in footprints[current]:
            for other in membership[key]:
                if other not in found:found.add(other);todo.append(other)
    return sorted(found)


def refine_word(old,gray,groups,index,rulers,*,iterations=5000,pixel_size=None,
                body_consensus=False,stock_padding=32,flow_constraint_padding=2,
                flow_body_padding=.5,precise_fixed_ink=False,rotation_anchor='center',whole_word_envelope=False,
                measured_motion_owners=False):
    """Rotate a short word only when its individual glyphs confirm the tilt."""
    evidence=word_rotation_evidence(gray,groups[index],pixel_size=pixel_size,
                                    body_consensus=body_consensus,
                                    whole_word_envelope=whole_word_envelope)
    if evidence is None:
        return old,{'accepted':False,'rejected':'independent glyph baseline unsupported'}
    motion_owners=None
    body=np.asarray(groups[index])
    if measured_motion_owners:
        motion_owners=shared_motion_owners(gray,groups,index)
        body=np.unique(np.vstack([groups[i] for i in motion_owners]),axis=0)
        if len(motion_owners)>1:
            evidence=word_rotation_evidence(gray,body,pixel_size=pixel_size,
                body_consensus=body_consensus,whole_word_envelope=whole_word_envelope)
            if evidence is None:
                return old,{'accepted':False,'rejected':'shared material rotation unsupported',
                            'motion_owner_indices':motion_owners}
    h,w=gray.shape
    scale=np.array([w,h] if pixel_size is None else pixel_size,dtype=float)
    points=body*scale
    low,high=points.min(0),points.max(0)
    center=(low+high)/2;slope=evidence['slope']
    if rotation_anchor not in ('center','leading','trailing'):
        return old,{'accepted':False,'rejected':'invalid rigid rotation anchor'}
    pivot=center.copy()
    if rotation_anchor!='center':pivot[0]=(low if rotation_anchor=='leading' else high)[0]
    quad=np.array([[low[0],low[1]+slope*(low[0]-center[0])],
                   [high[0],low[1]+slope*(high[0]-center[0])],
                   [high[0],high[1]+slope*(high[0]-center[0])],
                   [low[0],high[1]+slope*(low[0]-center[0])]])/scale
    # A compact word can sit close to another row. Lock its measured ink
    # envelope without reserving several pixels of blank stock as if they
    # were glyphs. The final inverse checks still cover every owned ink point,
    # every other word and every complete protected rule.
    result,report=_refine(old,[quad],groups,scale,rulers,iterations=iterations,
                          minimum_owners=1,ink_padding=.5,stock_padding=stock_padding,
                          use_flow=True,flow_constraint_padding=flow_constraint_padding,
                          flow_body_padding=flow_body_padding,
                          precise_fixed_ink=precise_fixed_ink,
                          rotation_center_px=pivot if rotation_anchor!='center' else None,
                          explicit_owners=motion_owners)
    report['independent_glyph_baseline']=evidence
    if measured_motion_owners:
        report['motion_owner_binding']={'selected_owner':index,'owners':motion_owners,
            'basis':'Selected ink and owners sharing observed dark raster cells',
            'all_original_material_owners_retained':True}
    if rotation_anchor!='center':report['rotation_anchor']=rotation_anchor
    return result,report
