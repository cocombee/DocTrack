"""Extend a measured rule to a nearby measured crossing using real ink.

No page position or endpoint guess supplies a new segment. The extension must
have coherent contrast, safe stock and no independent word ink in its path.
"""
import cv2
import numpy as np


def registered_material_join(registry,a,b,word_groups,width,height):
    """Bind a small native ridge gap to one complete immutable source stroke.

    Raster ridge centers are estimates, rather than new material registrations.
    Both full sections must match one earlier stroke. Word occlusion, incomplete
    coverage, ambiguous strokes and uncertain tangent origins retain the legacy
    join check. This helper supplies evidence, never an acceptance verdict.
    """
    scale=max(width,height)/1280;gap=b[0,0]-a[-1,0]
    if not 0<gap*(width-1)<=12*scale:return None
    low=min(a[-1,1],b[0,1])-3*scale/(height-1)
    high=max(a[-1,1],b[0,1])+3*scale/(height-1)
    for group in word_groups:
        group=np.asarray(group)
        if len(group) and (group[:,0].min()<b[0,0] and group[:,0].max()>a[-1,0]
                          and group[:,1].min()<high and group[:,1].max()>low):
            return None
    matches=[];endpoints=np.array([a[-1],b[0]])
    for record in registry:
        trace=np.asarray(record['reference']);sources=np.asarray(record['source'])
        if (trace.ndim!=2 or trace.shape[1]!=2 or trace.shape!=sources.shape or
                len(trace)<64 or not np.isfinite(trace).all() or
                not np.isfinite(sources).all()):continue
        order=np.argsort(trace[:,0]);trace=trace[order]
        if (np.diff(trace[:,0])<=0).any():continue
        errors=[];supported=True
        for guide in (a,b):
            coverage=np.mean((guide[:,0]>=trace[0,0])&(guide[:,0]<=trace[-1,0]))
            error=abs(guide[:,1]-np.interp(guide[:,0],trace[:,0],trace[:,1]))*(height-1)
            if coverage<.95 or error.max()>1.5*scale:
                supported=False;break
            errors.append(float(error.max()))
        if not supported:continue
        indices=np.array([int(abs(trace[:,0]-q[0]).argmin()) for q in endpoints])
        distances=abs(trace[indices,0]-endpoints[:,0])*(width-1)
        if (distances>.25*scale).any() or indices[0]>=indices[1]:continue
        matches.append({'ruler_index':record['index'],'source_indices':order[indices].tolist(),
            'source_points':sources[order[indices]].copy(),
            'reference_points':trace[indices].copy(),
            'complete_registration_points':len(trace),
            'section_ridge_error_px':errors,'tangent_registration_distance_px':distances.tolist()})
    return matches[0] if len(matches)==1 else None


def measured_straight_anchor(gray,safe,guide,word_mask):
    """Calibrate a fixed subpixel ruler against its ridge in this raster."""
    h,w=gray.shape;scale=max(h,w)/1280;guide=np.asarray(guide,dtype=float)
    if len(guide)<12 or np.ptp(guide[:,1])*(h-1)>.1:return None
    contrast=np.maximum(0,cv2.GaussianBlur(gray.astype(np.float32),(0,0),max(2,6*scale))-gray)
    observed=[]
    for x,prediction in guide[np.linspace(0,len(guide)-1,min(80,len(guide)),dtype=int)]*[w-1,h-1]:
        x=int(round(x));low=max(2,int(round(prediction))-3);high=min(h-2,int(round(prediction))+4)
        if not 0<=x<w or low>=high:continue
        if word_mask[max(0,low-2):min(h,high+2),x].any():continue
        ys=np.arange(low,high);score=np.minimum(contrast[ys,x],8)-.7*abs(ys-prediction)
        center=int(ys[np.argmax(score)])
        if contrast[center,x]<3 or not (safe[center-2:center+3,x]>250).all():continue
        profile=contrast[center-2:center+3,x]
        hits=np.flatnonzero(profile>max(3,float(profile.max())*.55))+center-2
        if len(hits):observed.append(float(hits.mean()))
    if len(observed)<12:return None
    observed=np.asarray(observed);target=float(np.median(observed))
    if (abs(target-float(np.median(guide[:,1]))*(h-1))>max(.75,1.5*scale) or
            np.quantile(abs(observed-target),.9)>max(.5,.75*scale)):return None
    result=guide.copy();result[:,1]=target/(h-1)
    return result


def extend_to_crossings(gray,safe,guide,perpendicular,word_mask,*,audit=None,collinear=()):
    h,w=gray.shape;scale=max(h,w)/1280
    guide=np.asarray(guide,dtype=float)
    if word_mask.shape!=gray.shape or safe.shape!=gray.shape:raise ValueError('Rule evidence must share one pixel basis')
    contrast=np.maximum(0,cv2.GaussianBlur(gray.astype(np.float32),(0,0),max(2,6*scale))-gray)
    result=guide.copy()
    for before in [True,False]:
        endpoint=result[0] if before else result[-1]
        candidates=[]
        for crossing in perpendicular:
            crossing=np.asarray(crossing);crossing=crossing[np.argsort(crossing[:,1])]
            # Rule tracking can stop a few pixels before the perpendicular
            # stroke obscures its profile. Only actual measured extension ink
            # below can authorize crossing this small endpoint gap.
            margin=6*scale/(h-1)
            if not crossing[0,1]-margin<=endpoint[1]<=crossing[-1,1]+margin:continue
            x=float(np.interp(endpoint[1],crossing[:,1],crossing[:,0]))
            gap=(endpoint[0]-x) if before else (x-endpoint[0])
            if 2/(w-1)<gap<.07:candidates.append((gap,x,None))
        for anchor in collinear:
            anchor=np.asarray(anchor);anchor=anchor[np.argsort(anchor[:,0])]
            if np.ptp(anchor[:,1])*(h-1)>.1:continue
            edge=anchor[-1] if before else anchor[0]
            gap=endpoint[0]-edge[0] if before else edge[0]-endpoint[0]
            if (2/(w-1)<gap<.035 and
                    abs(edge[1]-endpoint[1])*(h-1)<2.5*scale):
                candidates.append((gap,float(edge[0]),anchor))
        if not candidates:
            if audit is not None:audit.append({'before':before,'rejected':'no nearby supported crossing'})
            continue
        _,target,anchor=min(candidates,key=lambda item:item[0])
        measured=result[:min(20,len(result))] if before else result[-min(20,len(result)):]
        velocity=float(np.clip(np.polyfit(measured[:,0]*(w-1),measured[:,1]*(h-1),1)[0],-.45,.45))
        direction=-1 if before else 1;start=int(round(endpoint[0]*(w-1)));stop=int(round(target*(w-1)))
        y=float(endpoint[1]*(h-1));added=[];supported=0;possible=True
        for x in range(start+direction,stop+direction,direction):
            predicted=y+direction*velocity;radius=max(3,round(4*scale))
            low=max(3,int(round(predicted))-radius);high=min(h-3,int(round(predicted))+radius+1)
            if low>=high or word_mask[max(0,low-2):min(h,high+2),x].any():
                if audit is not None:audit.append({'before':before,'rejected':'word or domain boundary','pixel':[x,predicted]})
                possible=False;break
            ys=np.arange(low,high);score=np.minimum(contrast[ys,x],8)-.7*abs(ys-predicted)
            yy=int(ys[np.argmax(score)])
            if not (safe[max(0,yy-3):min(h,yy+4),x]>250).all():
                if audit is not None:audit.append({'before':before,'rejected':'unsafe stock','pixel':[x,yy]})
                possible=False;break
            if contrast[yy,x]<3:
                added.append([x,predicted]);y=predicted;continue
            if abs(yy-predicted)>3.5:
                if audit is not None:audit.append({'before':before,'rejected':'discontinuous ink','pixel':[x,yy],'prediction':predicted})
                possible=False;break
            profile=contrast[max(0,yy-2):min(h,yy+3),x]
            hits=np.flatnonzero(profile>max(3,float(profile.max())*.55))+max(0,yy-2)
            center=float(hits.mean()) if len(hits) else float(yy)
            velocity=float(np.clip(.8*velocity+.2*(center-y)*direction,-.45,.45))
            y=center;added.append([x,center]);supported+=1
        if not possible or not added or supported/len(added)<.85:
            if audit is not None and possible:audit.append({'before':before,'rejected':'insufficient measured ink','supported':supported,'length':len(added)})
            continue
        extra=np.asarray(added,dtype=float)/[w-1,h-1]
        if anchor is not None:
            endpoint=anchor[-1] if before else anchor[0]
            if abs(extra[-1,1]-endpoint[1])*(h-1)>2.5*scale:continue
            if before:
                extra=extra[extra[:,0]>endpoint[0]+1e-12]
                result=np.vstack([anchor,extra[::-1],result])
            else:
                extra=extra[extra[:,0]<endpoint[0]-1e-12]
                result=np.vstack([result,extra,anchor])
            continue
        result=np.vstack([extra[::-1],result]) if before else np.vstack([result,extra])
    return result


def measured_chains(gray, safe, guides, word_mask):
    """Join adjacent fragments only when the intervening stroke is observed.

    An occluding word or an unsupported gap keeps the fragments independent.
    All original section points remain in a joined proposal, so preserving a
    neighbouring section cannot silently remove its geometry from the audit.
    """
    h, w = gray.shape
    if safe.shape != gray.shape or word_mask.shape != gray.shape:
        raise ValueError('Rule evidence must share one pixel basis')
    contrast = np.maximum(0, cv2.GaussianBlur(gray.astype(np.float32),
                           (0, 0), max(2, 6 * max(h, w) / 1280)) - gray)
    chains = []
    for guide in sorted(guides, key=lambda g: g[0, 0]):
        guide = np.asarray(guide)
        choices = []
        for index, chain in enumerate(chains):
            end, start = chain[-1][-1], guide[0]
            gap = (start[0] - end[0]) * (w - 1)
            if not 1 < gap < .035 * (w - 1):
                continue
            slopes = [np.polyfit(g[:, 0], g[:, 1], 1)[0]
                      for g in (chain[-1][-12:], guide[:12])]
            predicted = abs(start[1] - end[1] - sum(slopes) / 2 *
                            (start[0] - end[0])) * (h - 1)
            if predicted > 3 or abs(start[1] - end[1]) * (h - 1) > 4:
                continue
            xs = np.arange(int(round(end[0] * (w - 1))) + 1,
                           int(round(start[0] * (w - 1))))
            centers = np.interp(xs / (w - 1), [end[0], start[0]],
                                [end[1], start[1]]) * (h - 1)
            bridge = []; supported = 0
            for x, center in zip(xs, centers):
                y = int(round(center))
                lo, hi = max(2, y - 3), min(h - 2, y + 4)
                if lo >= hi or word_mask[lo:hi, x].any():
                    break
                ys = np.arange(lo, hi)
                score = np.minimum(contrast[ys, x], 8) - .7 * abs(ys - center)
                yy = int(ys[np.argmax(score)])
                if not (safe[yy-2:yy+3, x] > 250).all():
                    break
                if contrast[yy, x] > 3:
                    supported += 1
                    profile = contrast[yy-2:yy+3, x]
                    hits = np.flatnonzero(profile > max(3, profile.max() * .55))
                    center = float(hits.mean() + yy - 2)
                bridge.append([x / (w - 1), center / (h - 1)])
            if len(bridge) != len(xs) or not len(xs) or supported / len(xs) < .85:
                continue
            choices.append((predicted, index, np.asarray(bridge)))
        if choices:
            _, index, bridge = min(choices, key=lambda c: c[:2])
            chains[index].extend([bridge, guide])
        else:
            chains.append([guide])
    return [np.vstack(chain) for chain in chains if len(chain) > 1]

def continued_thin_sections(gray,safe,horizontal,perpendicular,word_mask):
    """Recover one thin stroke observed in two separated measured fragments.

    A measured perpendicular can connect the stroke to a large frame component.
    Separate only its observed ink outside complete word envelopes, then require
    a thin unbranched component and agreement with both independent fragments.
    No missing pixels or unrecognized word-shaped component supplies a bridge.
    """
    h,w=gray.shape
    if safe.shape!=gray.shape or word_mask.shape!=gray.shape:
        raise ValueError('Rule evidence must share one pixel basis')
    scale=max(h,w)/1280
    contrast=cv2.GaussianBlur(gray.astype(np.float32),(0,0),7)-gray
    ink=((contrast>10)&(safe>250)&(word_mask==0)).astype(np.uint8)
    crossing=np.zeros_like(ink)
    for guide in perpendicular:
        cv2.polylines(crossing,[np.rint(np.asarray(guide)*[w-1,h-1]).astype(np.int32)],False,1,5)
    ink[(crossing>0)&(word_mask==0)]=0
    _,labels,stats,_=cv2.connectedComponentsWithStats(ink)
    result=[]
    for label,(x,y,cw,ch,area) in enumerate(stats[1:],1):
        if cw<64 or cw<w*.045 or ch>h*.04 or cw<ch*10:continue
        component=labels[y:y+ch,x:x+cw]==label
        counts=component.sum(0)
        if (counts>0).mean()<.95 or np.quantile(counts,.95)>6*scale:continue
        xs=np.flatnonzero(counts)
        profiles=[np.flatnonzero(component[:,i]) for i in xs]
        thickness=np.array([np.ptp(q)+1 for q in profiles])
        if np.quantile(thickness,.95)>6*scale or np.mean(thickness==counts[xs])<.95:continue
        yy=np.array([q.mean()+y for q in profiles]);xs=xs+x
        ridge=np.array([np.median(yy[abs(xs-z)<4]) for z in xs])
        trace=np.c_[xs/(w-1),ridge/(h-1)]
        matches=[]
        for guide in horizontal:
            low=max(trace[0,0],guide[0,0]);high=min(trace[-1,0],guide[-1,0])
            if (high-low)*(w-1)<48:continue
            samples=np.linspace(low,high,48)
            error=abs(np.interp(samples,trace[:,0],trace[:,1])-
                      np.interp(samples,guide[:,0],guide[:,1]))*(h-1)
            if np.quantile(error,.95)<1.5:matches.append(guide)
        # Repeated overlapping detections are not two separated fragments.
        if not any(min(a[-1,0],b[-1,0])-max(a[0,0],b[0,0])<=0
                   for i,a in enumerate(matches) for b in matches[i+1:]):continue
        if np.ptp(trace[:,0])<=max(np.ptp(q[:,0]) for q in matches)+16/(w-1):continue
        result.append(trace)
    return result
