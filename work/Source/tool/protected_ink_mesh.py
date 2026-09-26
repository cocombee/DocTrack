"""Refine supported ink while preserving neighbouring glyphs and prior maps."""
import json
import sys
from pathlib import Path
import cv2
import numpy as np
import residual_content_mesh as accepted


def glyph_boxes(gray,guide):
    h,w=gray.shape;scale=max(h,w)/1280
    lo=max(0,int(guide[:,1].min()*h)-round(30*scale))
    hi=min(h,int(guide[:,1].max()*h)+round(30*scale))
    crop=gray[lo:hi].astype(float)
    ink=cv2.GaussianBlur(crop,(0,0),max(2,7*scale))-crop
    _,_,stats,centers=cv2.connectedComponentsWithStats((ink>10).astype(np.uint8)*255)
    boxes=[]
    for (x,y,cw,ch,area),center in zip(stats,centers):
        if not (5*scale<ch<35*scale and 1*scale<cw<w*.09 and area>8*scale**2):continue
        if not guide[0,0]-.01<center[0]/(w-1)<guide[-1,0]+.01:continue
        expected=np.interp(center[0]/(w-1),guide[:,0],guide[:,1])*(h-1)
        if abs(y+lo+ch-1-expected)>10*scale:continue
        x1,y1=x+cw-1,y+lo+ch-1;y+=lo
        boxes.append(np.array([[x,y],[x1,y],[x1,y1],[x,y1],
                               [(x+x1)/2,y],[(x+x1)/2,y1]])/[w-1,h-1])
    return np.array(boxes).reshape(-1,6,2)


def envelope(boxes,x,pad):
    left=boxes[:,:,0].min(axis=1);right=boxes[:,:,0].max(axis=1)
    top=boxes[:,:,1].min(axis=1);bottom=boxes[:,:,1].max(axis=1)
    active=(x[None,:]>=left[:,None]-pad)&(x[None,:]<=right[:,None]+pad)
    lo=np.where(active,top[:,None],np.inf).min(axis=0)
    hi=np.where(active,bottom[:,None],-np.inf).max(axis=0)
    return lo,hi


def verified_orthogonal_ink(gray,mask):
    p=accepted.prior
    rotated=cv2.rotate(gray,cv2.ROTATE_90_CLOCKWISE)
    h,w=rotated.shape
    _,structures,_=accepted.orthogonal_guards(gray,mask)
    guards=list(structures)
    textline=p.reference.textline
    for binary in (textline.binarize(rotated),textline.binarize_light(rotated)):
        for group in textline._link_ccs_into_lines(textline._extract_ccs(binary)):
            if len(group)<6 or not textline._is_high_confidence_line(group):continue
            gaps=np.array([b.x0-a.x1 for a,b in zip(group,group[1:])])
            size=float(np.median([c.height for c in group]))
            # Rows crossing a paragraph are not a sideways word. The majority
            # of adjacent glyphs must have real inter-letter spacing.
            if np.mean(gaps<size*.5)<.6:continue
            points=[]
            for c in group:
                points.extend([[c.x0,c.y0],[c.x1,c.y0],[c.x1,c.y1],[c.x0,c.y1],c.center])
            q=np.array(points)/[w-1,h-1]
            guards.append(np.column_stack([q[:,1],1-q[:,0]]))
    return guards


def refine(path,result):
    p=accepted.prior
    rgba=cv2.imread(str(path),-1);h,w=rgba.shape[:2]
    mask=rgba[:,:,3] if rgba.shape[2]==4 else np.full((h,w),255,np.uint8)
    alpha=mask[:,:,None]/255
    gray=cv2.cvtColor((rgba[:,:,:3]*alpha+255*(1-alpha)).astype(np.uint8),cv2.COLOR_BGR2GRAY)
    f=p.reference.base.extract_text_features(gray,mask=mask)
    rules=p.reference.extract_rules(gray);texts=p.reference.text_rulers(gray,f)
    texts+=p.reference.supplemental_text(gray,texts+rules)
    grid=accepted.grid_of(result);texts=accepted.canonical_text_rows(texts,grid)
    measured=[(g,accepted.character_anchors(gray,g),glyph_boxes(gray,g)) for g in texts]
    bars=p.reference.barcode_rulers(gray);envelopes=p.reference.measured_envelopes(gray,bars)
    circles=accepted.circular_landmarks(gray);orthogonal=verified_orthogonal_ink(gray,mask)
    audit=[];accepted_any=False
    for guide,data,boxes in measured:
        if data is None or not len(boxes):continue
        anchors,size=data;mapped=p.inverse(grid,anchors)
        qualifies,evidence=accepted.residual_evidence(mapped,h)
        if not qualifies:continue
        target=float(mapped[:,1].mean());x0,y0=mapped.min(axis=0);x1,y1=mapped.max(axis=0)
        target_boxes=p.inverse(grid,boxes.reshape(-1,2)).reshape(-1,6,2)
        x0=float(target_boxes[:,:,0].min());x1=float(target_boxes[:,:,0].max())
        nr=max(grid.shape[0],1025);nc=max(grid.shape[1],513)
        u,v=np.meshgrid(np.linspace(0,1,nc),np.linspace(0,1,nr))
        inner_lo,inner_hi=envelope(target_boxes,u[0],.003)
        centers=target_boxes[:,:,0].mean(axis=1);order=np.argsort(centers)
        missing=~np.isfinite(inner_lo)
        inner_lo[missing]=np.interp(u[0][missing],centers[order],target_boxes[order,:,1].min(axis=1))
        inner_hi[missing]=np.interp(u[0][missing],centers[order],target_boxes[order,:,1].max(axis=1))
        delta=target-np.interp(u[0],mapped[:,0],mapped[:,1])
        inner_lo+=np.minimum(delta,0)-.0002
        inner_hi+=np.maximum(delta,0)+.0002
        outer_lo=np.maximum(.005,inner_lo-size*.8);outer_hi=np.minimum(.995,inner_hi+size*.8)
        protected=rules+bars+envelopes+circles+orthogonal
        labels=(['rule']*len(rules)+['barcode']*len(bars)+['envelope']*len(envelopes)+
                ['shape']*len(circles)+['orthogonal']*len(orthogonal))
        for other,_,other_boxes in measured:
            if other is guide:continue
            protected.append(other)
            labels.append('other_baseline')
            if not len(other_boxes):continue
            points=other_boxes.reshape(-1,2);protected.append(points)
            labels.append('other_glyphs')
            q=p.inverse(grid,points).reshape(-1,6,2)
            low,high=envelope(q,u[0],2/(nc-1))
            if q[:,:,1].mean()<target:outer_lo=np.maximum(outer_lo,high+.0005)
            else:outer_hi=np.minimum(outer_hi,low-.0005)
        entry={'position':mapped.mean(axis=0).tolist(),**evidence}
        audit.append(entry)
        outer_lo=np.ceil(outer_lo*(nr-1))/(nr-1)
        outer_hi=np.floor(outer_hi*(nr-1))/(nr-1)
        allowed=(outer_lo<inner_lo)&(inner_hi<outer_hi)
        ox0=max(.005,x0-.02);ox1=min(.995,x1+.02)
        left=np.clip((v-outer_lo[None,:])/np.maximum(inner_lo-outer_lo,1e-9),0,1)
        right=np.clip((outer_hi[None,:]-v)/np.maximum(outer_hi-inner_hi,1e-9),0,1)
        weight=(p.window(u,ox0,x0-.006,x1+.006,ox1)*
                left**2*(3-2*left)*right**2*(3-2*right)*allowed[None,:])
        coordinates=np.column_stack([u.ravel(),(v-delta[None,:]*weight).ravel()])
        proposal=p.sample(grid,coordinates).reshape(nr,nc,2)
        fixed=np.vstack(protected) if protected else np.empty((0,2))
        fixed_mapped=p.inverse(grid,fixed)
        fixed_shift=np.abs((p.inverse(proposal,fixed)-fixed_mapped)*[w,h])
        movement=float(fixed_shift.max()) if len(fixed) else 0.
        before=float(np.std(mapped[:,1])*h);after=float(np.std(p.inverse(proposal,anchors)[:,1])*h)
        new_boxes=p.inverse(proposal,boxes.reshape(-1,2)).reshape(-1,6,2)
        shape_changes=np.max(np.abs(
            (new_boxes[:,[3,2,5],1]-new_boxes[:,[0,1,4],1])-
            (target_boxes[:,[3,2,5],1]-target_boxes[:,[0,1,4],1])),axis=1)*h
        shape_change=float(shape_changes.max())
        sideways_shift=float(np.max(np.abs(new_boxes[:,:,0]-target_boxes[:,:,0]))*w)
        worst_index=int(np.unravel_index(fixed_shift.argmax(),fixed_shift.shape)[0]) if len(fixed) else 0
        group_index=int(np.searchsorted(np.cumsum([len(g) for g in protected]),worst_index,side='right')) if len(fixed) else 0
        ok=bool(p.valid(proposal) and movement<.1 and shape_change<.25 and sideways_shift<.1
                and after<min(.3,before*.25))
        entry.update(accepted=ok,before_px=before,after_px=after,
                     protected_shift_px=movement,glyph_height_change_px=shape_change,
                     sideways_glyph_shift_px=sideways_shift,
                     available_column_fraction=float(allowed.mean()),topology_valid=p.valid(proposal),
                     worst_fixed_position=(fixed_mapped[np.unravel_index(fixed_shift.argmax(),fixed_shift.shape)[0]].tolist()
                                           if len(fixed) else None),
                     worst_fixed_kind=labels[group_index] if len(fixed) else None,
                     worst_glyph_position=target_boxes[shape_changes.argmax()].mean(axis=0).tolist())
        if ok:
            if not accepted_any:
                result['preserved_mesh']={key:result[key] for key in
                    ('rows','columns','grid','method','preserved_mesh','refinement_regions') if key in result}
                result['refinement_regions']=[]
                accepted_any=True
            grid=proposal
            result['refinement_regions'].append([float(np.floor(ox0*(nc-1))/(nc-1)),float(outer_lo.min()),
                                                float(np.ceil(ox1*(nc-1))/(nc-1)),float(outer_hi.max())])
            result.update(rows=nr,columns=nc,grid=grid.reshape(-1,2).tolist(),method='Protected ink residual correction')
    result['ink_guard_audit']=audit
    return result


def solve(path,working=None):
    original=accepted.solve(path,working)
    proposal={**original}
    try:
        return refine(path,proposal)
    except (OSError,ValueError,ArithmeticError,AttributeError,IndexError,KeyError,TypeError,cv2.error) as error:
        original['ink_guard_audit']={'accepted':False,'error':type(error).__name__}
        return original


if __name__=='__main__':
    result=solve(sys.argv[1],sys.argv[3] if len(sys.argv)>3 else None)
    Path(sys.argv[2]).write_text(json.dumps(result,allow_nan=False))
