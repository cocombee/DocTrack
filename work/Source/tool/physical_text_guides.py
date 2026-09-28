"""Find measured glyph baselines for the shared page mesh.

Native quadrilaterals locate candidate text. The guides themselves come from
two independently thresholded sets of actual glyph components, not the OCR
box edge. A tall OCR box can contain two physical rows; they are split only
when each row has its own supported glyph trace.
"""
import cv2
import numpy as np
from native_word_evidence import measurement_words
from rigid_word_row import find_rows


def associated_rows(words,w,h):
    """Retain short neighbouring words, even when the old row finder omits them."""
    observations=measurement_words(words,w,h)
    rows=find_rows(observations,[w-1,h-1])
    used={tuple(np.round(np.asarray(q).reshape(-1),10)) for row in rows for q in row}
    unassigned=[]
    for word in observations:
        q=np.asarray(word['points'])
        if q.shape!=(4,2):continue
        normalized=q.copy()
        # measurement_words converts Vision pixel edges to this basis.
        key=tuple(np.round(normalized.reshape(-1),10))
        if key in used:continue
        unassigned.append(normalized);used.add(key)
    scale=np.array([w-1,h-1])
    chains=[]
    for quad in sorted(unassigned,key=lambda q:float((q*scale)[:,0].min())):
        low=(quad*scale).min(0);high=(quad*scale).max(0)
        choices=[]
        for i,chain in enumerate(chains):
            prev=chain[-1]*scale;before_low=prev.min(0);before_high=prev.max(0)
            gap=low[0]-before_high[0]
            height=min(high[1]-low[1],before_high[1]-before_low[1])
            overlap=min(high[1],before_high[1])-max(low[1],before_low[1])
            if (-2<=gap<=max(15,w*.02) and overlap>.65*height and
                    abs(high[1]-before_high[1])<max(7,h*.006)):
                choices.append((abs(gap),i))
        if choices:chains[min(choices)[1]].append(quad)
        else:chains.append([quad])
    rows.extend(chains)
    return rows


def _anchors(gray,safe,threshold):
    h,w=gray.shape;scale=max(h,w)/1280
    contrast=cv2.GaussianBlur(gray.astype(np.float32),(0,0),max(2,7*scale))-gray
    mask=((contrast>threshold)&(safe>250)).astype(np.uint8)
    _,labels,stats,_=cv2.connectedComponentsWithStats(mask)
    found=[]
    for index,(x,y,cw,ch,area) in enumerate(stats[1:],1):
        if not (6*scale<ch<38*scale and 2<cw<ch*2.4 and area>15*scale):continue
        body=labels[y:y+ch,x:x+cw]==index
        bottoms=[np.flatnonzero(body[:,col])[-1] for col in range(cw) if body[:,col].any()]
        found.append([x+cw/2,y+np.quantile(bottoms,.85),ch])
    return np.asarray(found,float).reshape(-1,3)


def _fit(points,reference,*,minimum_span,seed_y=None,degree=2,minimum_support=8):
    """Deterministic robust quadratic using only actual observed glyphs."""
    if len(points)<minimum_support:return None
    x=points[:,0];y=points[:,1]
    origin=float(x.mean());span=float(np.ptp(x));scale=max(1.,float(np.median(points[:,2])))
    if span<minimum_span:return None
    z=(x-origin)/span
    design=np.c_[z*z,z,np.ones(len(z))] if degree==2 else np.c_[z,np.ones(len(z))]
    # Candidate triples cover left, middle and right support. A mixture of
    # two lines is resolved by its independently observed center-row seed.
    rng=np.random.default_rng(1);best=None
    for _ in range(min(600,max(100,len(points)*12))):
        picks=rng.choice(len(points),degree+1,replace=False)
        if np.ptp(x[picks])<span*.45:continue
        coefficients=np.linalg.lstsq(design[picks],y[picks],rcond=None)[0]
        fitted=design@coefficients;keep=abs(y-fitted)<=max(2.5,scale*.18)
        if keep.sum()<minimum_support or np.ptp(x[keep])<minimum_span:continue
        if seed_y is not None and abs(float(np.polyval(coefficients,(reference-origin)/span))-seed_y)>scale*.3:
            continue
        for _ in range(3):
            coefficients=np.linalg.lstsq(design[keep],y[keep],rcond=None)[0]
            fitted=design@coefficients;keep=abs(y-fitted)<=max(2.5,scale*.18)
        residual=y[keep]-fitted[keep]
        if keep.sum()<minimum_support or np.ptp(x[keep])<minimum_span or np.std(residual)>1.5:continue
        selected=x[keep];normalized=(selected-origin)/span
        slope=(2*coefficients[0]*normalized+coefficients[1])/span if degree==2 else coefficients[0]/span
        if np.max(abs(slope))>.15:continue
        score=(int(keep.sum()),-float(np.std(residual)))
        if best is None or score>best[0]:best=(score,coefficients,keep,origin,span)
    if best is None:return None
    _,coef,keep,origin,span=best
    domain=[float(x[keep].min()),float(x[keep].max())]
    xx=np.linspace(*domain,65);yy=np.polyval(coef,(xx-origin)/span)
    return dict(points=np.c_[xx,yy],support=int(keep.sum()),observed=int(len(points)),
        domain_px=domain,scatter_px=float(np.std(y[keep]-np.polyval(coef,(x[keep]-origin)/span))),
        height_px=float(np.median(points[:,2])),accepted_glyph_anchors_px=points[keep,:2].tolist())


def _typographic_body_curve(points, height, minimum_span):
    """Locally trace base glyphs without following isolated low marks.

    A connected descender or attached dot may extend below the ordinary
    baseline. The five-neighbour median treats it as a local outlier while
    keeping the actual base component in the row. Short independent marks
    (for example the separate dot under Ặ) never enter ``_anchors``.
    """
    if len(points)<16:return None
    ordered=points[np.argsort(points[:,0])]
    x,y=ordered[:,0],ordered[:,1]
    if np.ptp(x)<minimum_span:return None
    # Do not bridge an unobserved stretch with a synthetic typography guide.
    spacing=np.diff(x)
    if len(spacing) and spacing.max()>max(65.,np.median(spacing)*3.5):return None
    base=np.array([np.median(y[max(0,i-2):min(len(y),i+3)])
                   for i in range(len(y))])
    xx=np.linspace(x[0],x[-1],max(129,int(np.ceil(x[-1]-x[0]))+1))
    interpolated=np.interp(xx,x,base).astype(np.float32)
    sigma=max(8.,min(18.,height*.8))
    fitted=cv2.GaussianBlur(interpolated[None,:],(0,0),sigma)[0].astype(float)
    slope=np.gradient(fitted,xx)
    if np.max(np.abs(slope))>.15:return None
    residual=y-np.interp(x,xx,fitted)
    if np.percentile(np.abs(residual),90)>max(2.5,height*.16):return None
    return dict(points=np.c_[xx,fitted],support=len(ordered),observed=len(ordered),
                domain_px=[float(x[0]),float(x[-1])],
                scatter_px=float(np.std(residual)),height_px=float(height),
                accepted_glyph_anchors_px=ordered[:,:2].tolist(),
                body_anchor_p90_px=float(np.percentile(np.abs(residual),90)))


def _global_fit_misses_base_bodies(selected,fits,height):
    """Require a corroborated interior run of missed *main* glyph bodies."""
    if len(selected)!=2 or any(fit is None for fit in fits):return False
    if min(map(len,selected))<16:return False
    for profile,fit in zip(selected,fits):
        points=np.asarray(fit['points'])
        x=profile[:,0]
        expected=np.interp(x,points[:,0],points[:,1])
        missed=np.sort(x[(profile[:,1]<expected-max(3.,height*.17)) &
                         (x>points[0,0]+height) & (x<points[-1,0]-height)])
        if len(missed)<2 or not np.any(np.diff(missed)<height*2.3):return False
    return True


def measured_text_rows(gray,safe,rows):
    """Each yielded guide is one visible, two-profile corroborated text line."""
    h,w=gray.shape;basis=np.array([w-1,h-1]);scale=max(h,w)/1280
    profiles=[_anchors(gray,safe,t) for t in (10,16)]
    guides=[]
    for row_number,row in enumerate(rows):
        quads=np.asarray(row,float)*basis
        if quads.ndim!=3 or quads.shape[1:]!=(4,2):continue
        bounds=np.c_[quads.min(1),quads.max(1)]
        left,right=float(bounds[:,0].min()),float(bounds[:,2].max())
        singleton=len(row)==1
        minimum_span=w*(.06 if singleton else .09 if len(row)==2 else .12)
        minimum_support=4 if len(row)<3 else 8
        if right-left<minimum_span:continue
        selected=[]
        for profile in profiles:
            if not len(profile):break
            inside=np.zeros(len(profile),bool)
            for x0,y0,x1,y1 in bounds:
                inside|=((profile[:,0]>=x0-2*scale)&(profile[:,0]<=x1+2*scale)&
                         (profile[:,1]>=y0-2*scale)&(profile[:,1]<=y1+4*scale))
            selected.append(profile[inside])
        if len(selected)!=2 or min(map(len,selected))<minimum_support:continue
        heights=np.r_[selected[0][:,2],selected[1][:,2]]
        height=float(np.median(heights))
        selected=[q[abs(q[:,2]-height)<height*.32] for q in selected]
        if min(map(len,selected))<minimum_support:continue
        center=(left+right)/2
        center_y=np.concatenate([q[(abs(q[:,0]-center)<(right-left)*.16),1]
                                 for q in selected])
        bimodal=(len(center_y)>=6 and np.ptp(center_y)>height*1.25
                 and np.median(bounds[:,3]-bounds[:,1])>height*2.4)
        seeds=[]
        if bimodal:
            low,high=np.quantile(center_y,[.25,.75])
            if high-low>height*1.1:seeds=[float(low),float(high)]
        if not seeds:seeds=[None]
        first_curve=None
        for branch,seed in enumerate(seeds):
            candidates=selected
            if branch==1 and first_curve is not None:
                candidates=[]
                for q in selected:
                    ceiling=np.interp(q[:,0],first_curve[:,0],first_curve[:,1])+height*.65
                    candidates.append(q[q[:,1]>ceiling])
            degree=1 if right-left<w*.35 and not bimodal else 2
            fits=[_fit(q,center,minimum_span=max(minimum_span,(right-left)*.3),
                       seed_y=seed if branch==0 else None,degree=degree,
                       minimum_support=minimum_support)
                  for q in candidates]
            if any(q is None for q in fits):continue
            typographic=False
            if _global_fit_misses_base_bodies(candidates,fits,height):
                local=[_typographic_body_curve(q,height,max(minimum_span,(right-left)*.3))
                       for q in candidates]
                if all(q is not None for q in local):
                    common=[max(q['domain_px'][0] for q in local),
                            min(q['domain_px'][1] for q in local)]
                    if common[1]-common[0]>=max(minimum_span,(right-left)*.25):
                        test_x=np.linspace(*common,65)
                        difference=np.max(np.abs(np.interp(test_x,local[0]['points'][:,0],local[0]['points'][:,1])-
                                                 np.interp(test_x,local[1]['points'][:,0],local[1]['points'][:,1])))
                        if difference<=max(2.75*scale,height*.18):
                            fits=local
                            typographic=True
            domain=[max(q['domain_px'][0] for q in fits),min(q['domain_px'][1] for q in fits)]
            if domain[1]-domain[0]<max(minimum_span,(right-left)*.25):continue
            xx=np.linspace(*domain,65)
            curves=[np.interp(xx,q['points'][:,0],q['points'][:,1]) for q in fits]
            agreement=float(np.max(abs(curves[0]-curves[1])))
            if agreement>max(2.75*scale,height*.18):continue
            points=np.c_[xx,np.mean(curves,axis=0)]/basis
            guides.append(dict(row=row_number,branch=branch,points=points,
                profile_supports=[q['support'] for q in fits],agreement_px=agreement,
                domain_px=domain,median_glyph_height_px=height,
                accepted_glyph_anchors_px=[q['accepted_glyph_anchors_px'] for q in fits],
                compact_word_guide=len(row)<3 and min(q['support'] for q in fits)<8,
                model_is_not_registered_ink=True,
                typographic_base_body_guide=typographic,
                below_baseline_marks_not_used_as_guide=typographic))
            if branch==0:first_curve=np.c_[xx,np.mean(curves,axis=0)]
    return guides
