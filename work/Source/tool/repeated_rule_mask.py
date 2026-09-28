"""Recover repeated rule bodies hidden by the heuristic text mask.

A long independently measured section and two contrast profiles are required.
Protected material is never unmasked. This changes classification before new
registration, not image pixels, camera geometry, or an accepted owner ledger.
"""
import cv2
import numpy as np


def _measured_reference(points, width, scale):
    """Search along observed curvature; extrapolation only locates real ink."""
    x=np.arange(width,dtype=float)
    slope,intercept=np.polyfit(points[:,0],points[:,1],1)
    if abs(points[:,1]-(slope*points[:,0]+intercept)).max()<=3*scale:
        return slope*x+intercept,np.ones(width,bool),[dict(
            model='Observed straight section',slope=float(slope),
            search_extent_px=width)]
    center=np.interp(x,points[:,0],points[:,1])
    domain=(x>=points[0,0])&(x<=points[-1,0])
    origin=points[:,0].mean();span=np.ptp(points[:,0])
    coefficients=np.polyfit((points[:,0]-origin)/span,points[:,1],2)
    fitted=np.polyval(coefficients,(points[:,0]-origin)/span)
    if abs(points[:,1]-fitted).max()<=3*scale:
        supported=(x>=points[0,0]-.25*width)&(x<=points[-1,0]+.25*width)
        for side in (0,1):
            edge=points[-1,0] if side else points[0,0]
            local=points[abs(points[:,0]-edge)<=max(16,32*scale)]
            local_slope=(np.polyfit(local[:,0],local[:,1],1)[0] if len(local)>=16
                         else np.polyval(np.polyder(coefficients),(edge-origin)/span)/span)
            outside=((x>edge) if side else (x<edge))&supported
            delta=x[outside]-edge
            center[outside]=points[-1 if side else 0,1]+local_slope*delta+coefficients[0]*(delta/span)**2
        return center,domain|supported,[dict(model='Observed coherent curvature',
            normalized_coefficients=coefficients.tolist(),origin_px=float(origin),
            span_px=float(span),search_extent_px=.25*width)]
    endpoints=[]
    for side in (0,1):
        edge=points[-1,0] if side else points[0,0]
        local=points[abs(points[:,0]-edge)<=max(16,32*scale)]
        if len(local)<16 or np.ptp(local[:,0])<12*scale:
            continue
        slope,intercept=np.polyfit(local[:,0],local[:,1],1)
        if abs(local[:,1]-(slope*local[:,0]+intercept)).max()>3*scale:
            continue
        outside=((x>edge)&(x<=edge+.1*width) if side
                 else (x<edge)&(x>=edge-.1*width))
        # Continue the measured endpoint itself; the fit supplies direction.
        center[outside]=points[-1 if side else 0,1]+slope*(x[outside]-edge)
        domain|=outside
        endpoints.append(dict(side=side,observed_points=local.tolist(),
                              slope=float(slope),search_extent_px=.1*width))
    return center,domain,endpoints


def recover(gray, safe, heuristic, protected, guides, *, whole_rulings=False):
    gray, safe = np.asarray(gray), np.asarray(safe)
    heuristic, protected = np.asarray(heuristic), np.asarray(protected)
    if (gray.ndim!=2 or min(gray.shape)<16 or any(q.shape!=gray.shape
            for q in (safe,heuristic,protected)) or not np.isfinite(gray).all()):
        raise ValueError('Repeated rules need paired measurement masks')
    h,w=gray.shape;scale=max(h,w)/1280
    contrast=cv2.GaussianBlur(gray.astype(np.float32),(0,0),max(2,7*scale))-gray
    yy=np.arange(h)[:,None]
    clear=np.zeros_like(gray,np.uint8);reports=[]
    for value in guides:
        guide=np.asarray(value,float)
        if (guide.ndim!=2 or guide.shape[1:]!=(2,) or len(guide)<48
                or not np.isfinite(guide).all() or (np.diff(guide[:,0])<=0).any()
                or np.ptp(guide[:,0])<.3):
            continue
        points=guide*[w-1,h-1]
        reference,domain,endpoints=_measured_reference(points,w,scale)
        band=(abs(yy-reference[None,:])<=8*scale)&domain[None,:]
        curved=not endpoints or endpoints[0].get('model')!='Observed straight section'
        profiles=[]
        for threshold in (6,16):
            ink=(contrast>threshold)&band&(safe>250)
            count,labels,stats,_=cv2.connectedComponentsWithStats(ink.astype(np.uint8))
            components=[]
            for component,(x,y,cw,ch,area) in enumerate(stats[1:],1):
                if not (cw>=(5 if whole_rulings else 8)*scale and cw<.12*w and area/(cw*ch)>.4):
                    continue
                low=np.maximum(0,[x-1,y-1]);high=np.minimum([w,h],[x+cw+1,y+ch+1])
                if protected[low[1]:high[1],low[0]:high[0]].any():
                    continue
                sy,sx=np.where(labels[y:y+ch,x:x+cw]==component)
                sy=sy+y;sx=sx+x
                normal_extent=(float(np.ptp(sy-reference[sx]))+1 if curved else ch)
                if not (2*scale<=normal_extent<=8*scale and cw>(1.45 if whole_rulings else 1.7)*normal_extent):
                    continue
                center=np.average(np.c_[sx,sy],weights=contrast[sy,sx],axis=0)
                if abs(center[1]-np.interp(center[0],np.arange(w),reference))>(8 if whole_rulings else 3)*scale:
                    continue
                components.append(dict(bounds_px=[int(x),int(y),int(cw),int(ch)],
                                       center_px=center.tolist(),observed_pixels=int(area),
                                       normal_extent_px=float(normal_extent)))
            components.sort(key=lambda q:q['bounds_px'][0]);chains=[]
            for component in components:
                if chains:
                    before=chains[-1][-1]
                    previous=before['bounds_px'];current=component['bounds_px']
                    gap=current[0]-previous[0]-previous[2]
                    first_height=before['normal_extent_px'];last_height=component['normal_extent_px']
                    gap_low=previous[0]+previous[2];gap_high=current[0]
                    blocked=(whole_rulings and gap_high>gap_low and
                        np.any(((protected[:,gap_low:gap_high]>0)|(safe[:,gap_low:gap_high]<=250))
                               &band[:,gap_low:gap_high]))
                    if (not blocked and 0<gap<(24 if whole_rulings else 6)*scale
                            and max(first_height,last_height)<(2 if whole_rulings else 1.5)*min(first_height,last_height)):
                        chains[-1].append(component);continue
                chains.append([component])
            profiles.append([chain for chain in chains if len(chain)>=(8 if whole_rulings else 4)
                             and np.ptp([q['center_px'][0] for q in chain])>(.3 if whole_rulings else .15)*w])
        for chain in profiles[0]:
            boxes=np.asarray([q['bounds_px'] for q in chain])
            start=boxes[:,0].min();end=(boxes[:,0]+boxes[:,2]).max()
            matches=[other for other in profiles[1] if
                min(end,max(q['bounds_px'][0]+q['bounds_px'][2] for q in other))-
                max(start,min(q['bounds_px'][0] for q in other))>.9*(end-start)]
            if len(matches)!=1:
                continue
            for component in chain+(matches[0] if whole_rulings else []):
                x,y,cw,ch=component['bounds_px']
                clear[max(0,y-1):min(h,y+ch+1),max(0,x-1):min(w,x+cw+1)]=1
            reports.append(dict(thresholds=[6,16],components=chain,
                                independently_supported_components=matches[0],
                                measured_seed_bounds_px=[points[0].tolist(),points[-1].tolist()],
                                measured_endpoint_search=endpoints,
                                search_follows_observed_curvature=True,
                                protected_material_changed=False,ink_pixels_generated=False))
    clear[safe<=250]=0
    return (heuristic & (1-clear)) | protected, reports


def whole_ruling_guides(gray, safe, protected, guides, evidence):
    """One proposal per measured dashed ruling, retaining real blank gaps.

    Only two-profile repeated-ink evidence can join sections or extend tails.
    Curvature fits locate ink; they never supply replacement material points.
    Existing registered print remains protected by the caller's immutable mask.
    """
    h,w=gray.shape;basis=np.array([w-1,h-1]);scale=max(h,w)/1280
    contrast=cv2.GaussianBlur(gray.astype(np.float32),(0,0),max(2,7*scale))-gray
    result=list(guides);reports=[]
    for proof in evidence:
        components=proof['components']+proof['independently_supported_components']
        components=sorted(components,key=lambda q:q['center_px'][0])
        centers=np.asarray([q['center_px'] for q in components],float)
        if len(centers)<8 or np.ptp(centers[:,0])<.3*w:
            continue
        left=min(q['bounds_px'][0] for q in components)
        right=max(q['bounds_px'][0]+q['bounds_px'][2] for q in components)
        members=[]
        for index,guide in enumerate(result):
            points=np.asarray(guide)*basis
            if len(points)<3:continue
            inside=(points[:,0]>=left)&(points[:,0]<right)
            if inside.mean()<.8:continue
            residual=abs(points[inside,1]-np.interp(points[inside,0],centers[:,0],centers[:,1]))
            if np.quantile(residual,.95)<=3*scale:members.append(index)
        if not members:continue
        observed=[]
        for item in components:
            x,y,cw,ch=item['bounds_px']
            patch=contrast[y:y+ch,x:x+cw]
            ink=(patch>6)&(safe[y:y+ch,x:x+cw]>250)&(protected[y:y+ch,x:x+cw]==0)
            for column in range(cw):
                rows=np.flatnonzero(ink[:,column])
                if len(rows):
                    observed.append([x+column,y+float(np.average(rows,weights=patch[rows,column]))])
        if not observed:continue
        # Preserve the original observed samples at every existing tangent.
        originals=[np.asarray(result[i]) for i in members]
        joined=np.vstack([*originals,np.asarray(observed)/basis])
        order=np.argsort(joined[:,0],kind='stable');joined=joined[order]
        joined=joined[np.r_[True,np.diff(joined[:,0])>1e-10]]
        result=[g for i,g in enumerate(result) if i not in members]+[joined]
        reports.append(dict(original_sections=[g.tolist() for g in originals],
                            actual_added_samples=len(observed),
                            complete_guide_bounds_px=(joined*basis)[[0,-1]].tolist(),
                            repeated_ink_profiles=proof['thresholds'],
                            missing_ink_generated=False))
    result.sort(key=lambda g:float(np.asarray(g)[:,1].mean()))
    return result,reports


def restore_measured_ruling_tails(gray,safe,protected,guides,references):
    """Retain real ends measured before an OCR envelope hid nearby stock.

    References and guides must share this exact measurement raster. Nothing
    is extrapolated: every added point is an earlier observed rule sample,
    supported by current ink and clear of the actual glyph pixel bodies.
    """
    gray,safe,protected=map(np.asarray,(gray,safe,protected))
    if (gray.ndim!=2 or min(gray.shape)<16 or safe.shape!=gray.shape
            or protected.shape!=gray.shape or not np.isfinite(gray).all()):
        raise ValueError('Measured ruling tails require paired rasters')
    h,w=gray.shape;basis=np.array([w-1,h-1]);scale=max(h,w)/1280
    contrast=cv2.GaussianBlur(gray.astype(np.float32),(0,0),max(2,7*scale))-gray
    result=list(guides);reports=[]
    for reference in references:
        reference=np.asarray(reference,float)
        if (reference.ndim!=2 or reference.shape[1:]!=(2,) or len(reference)<48
                or not np.isfinite(reference).all() or (reference<0).any()
                or (reference>1).any() or (np.diff(reference[:,0])<=0).any()):continue
        matches=[]
        for index,current in enumerate(result):
            current=np.asarray(current,float)
            if (current.ndim!=2 or current.shape[1:]!=(2,) or len(current)<3
                    or not np.isfinite(current).all() or (current<0).any()
                    or (current>1).any() or (np.diff(current[:,0])<=0).any()):continue
            low=max(current[0,0],reference[0,0]);high=min(current[-1,0],reference[-1,0])
            if high-low<.3:continue
            points=current[(current[:,0]>=low)&(current[:,0]<=high)]
            if len(points)<3:continue
            error=abs(points[:,1]-np.interp(points[:,0],reference[:,0],reference[:,1]))*basis[1]
            if np.quantile(error,.95)<=2*scale:matches.append(index)
        if len(matches)!=1:continue
        index=matches[0];current=np.asarray(result[index])
        added=reference[(reference[:,0]<current[0,0])|(reference[:,0]>current[-1,0])]
        if not len(added):continue
        xy=np.rint(added*basis).astype(int)
        if (np.any(safe[xy[:,1],xy[:,0]]<=250) or protected[xy[:,1],xy[:,0]].any()
                or np.any(contrast[xy[:,1],xy[:,0]]<=3)):continue
        # Blank dash gaps carry no new ink, but may contain a protected glyph
        # or unavailable stock. A retained tail must not reconnect across it.
        blocked=False
        for start,end in ((reference[0,0],current[0,0]),
                          (current[-1,0],reference[-1,0])):
            if end<=start:continue
            columns=np.arange(int(np.ceil(start*basis[0])),int(np.floor(end*basis[0]))+1)
            rows=np.rint(np.interp(columns/basis[0],reference[:,0],reference[:,1])*basis[1]).astype(int)
            for offset in (-1,0,1):
                yy=np.clip(rows+offset,0,h-1)
                if protected[yy,columns].any() or np.any(safe[yy,columns]<=250):
                    blocked=True;break
            if blocked:break
        if blocked:continue
        joined=np.vstack([current,added]);joined=joined[np.argsort(joined[:,0],kind='stable')]
        result[index]=joined
        reports.append(dict(prior_measured_tail_points=added.tolist(),
            added_samples=len(added),current_glyph_overlap=False,
            current_ink_and_safe_stock_verified=True,protected_gaps_crossed=False,
            ink_or_points_generated=False))
    return result,reports
