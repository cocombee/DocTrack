"""Evidence-gated interior mesh. Geometry only; original pixels render in Dart.
Uses MIT UVDoc and PINTO0309/doc-dewarping. See Dewarping Evaluation provenance.
Input is an isolated, coarsely rectified document; mesh boundaries stay fixed.
"""
from pathlib import Path
import json,sys
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[3]/'Scanner Integration'/'Dewarping Evaluation'
sys.path.insert(0,str(ROOT/'Text Line Dewarp'/'src'))
from dewarping.textline import extract_text_features
N=33
U,V=np.meshgrid(np.linspace(0,1,N),np.linspace(0,1,N))
IDENTITY=np.stack([U,V],-1)

def lock_boundary(grid):
    g=cv2.resize(grid.astype(np.float64),(N,N),interpolation=cv2.INTER_LINEAR)
    d=g-IDENTITY;u=U[:,:,None];v=V[:,:,None]
    corners=(1-u)*(1-v)*d[0,0]+u*(1-v)*d[0,-1]+u*v*d[-1,-1]+(1-u)*v*d[-1,0]
    coons=(1-v)*d[0:1,:]+v*d[-1:,:]+(1-u)*d[:,0:1]+u*d[:,-1:]-corners
    result=IDENTITY+d-coons
    result[0]=IDENTITY[0];result[-1]=IDENTITY[-1];result[:,0]=IDENTITY[:,0];result[:,-1]=IDENTITY[:,-1]
    return result

def calibrate_corners(grid):
    """Retain predicted surface bends while removing global model framing.

    Full boundary locking subtracts real fold displacement along with framing.
    Calibrate only the four corners, then limit exterior extension smoothly.
    """
    raw=cv2.resize(grid.astype(np.float64),(N,N),interpolation=cv2.INTER_LINEAR)
    corners=np.float32([raw[0,0],raw[0,-1],raw[-1,-1],raw[-1,0]])
    matrix=cv2.getPerspectiveTransform(corners,np.float32([[0,0],[1,0],[1,1],[0,1]]))
    calibrated=cv2.perspectiveTransform(raw.reshape(-1,1,2),matrix).reshape(N,N,2)
    extension=max(0.,-float(calibrated.min()),float(calibrated.max())-1)
    if extension>.024:calibrated=IDENTITY+(calibrated-IDENTITY)*(.024/extension)
    return calibrated

def valid(grid, edge_extension=False):
    dx=np.diff(grid,axis=1);dy=np.diff(grid,axis=0)
    jac=[(a[:,:,0]*b[:,:,1]-a[:,:,1]*b[:,:,0])*(N-1)**2
        for a in (dx[:-1],dx[1:]) for b in (dy[:,:-1],dy[:,1:])]
    return bool(np.isfinite(grid).all() and np.max(np.abs(grid-IDENTITY))<.14 and min(j.min() for j in jac)>.2 and grid.min()>=(-.025 if edge_extension else -.001) and grid.max()<=(1.025 if edge_extension else 1.001))

def inverse_points(grid,points):
    # Newton solve the backward map for observed feature positions.
    estimate=points.copy()
    gx=np.gradient(grid,1/(N-1),axis=1);gy=np.gradient(grid,1/(N-1),axis=0)
    def sample(field,p):
        xy=np.clip(p*(N-1),0,N-1-1e-5);lo=xy.astype(int);f=xy-lo;x=lo[:,0];y=lo[:,1];a=f[:,0,None];b=f[:,1,None]
        return (1-a)*(1-b)*field[y,x]+a*(1-b)*field[y,x+1]+(1-a)*b*field[y+1,x]+a*b*field[y+1,x+1]
    for _ in range(10):
        delta=sample(grid,estimate)-points;a=sample(gx,estimate);b=sample(gy,estimate)
        det=a[:,0]*b[:,1]-a[:,1]*b[:,0]
        estimate-=np.stack([(b[:,1]*delta[:,0]-b[:,0]*delta[:,1])/det,(-a[:,1]*delta[:,0]+a[:,0]*delta[:,1])/det],1)
        estimate=np.clip(estimate,0,1)
    return estimate

def line_error(grid,lines):
    errors=[]
    for p in lines:
        q=inverse_points(grid,p)
        if np.ptp(q[:,0])<.08:continue
        design=np.column_stack([q[:,0],np.ones(len(q))]);fit=design@np.linalg.lstsq(design,q[:,1],rcond=None)[0]
        errors.append(float(np.sqrt(np.mean((q[:,1]-fit)**2))))
    return float(np.median(errors)) if errors else None

def solve(path, working_resolution_path=None):
    rgba=cv2.imread(str(path),cv2.IMREAD_UNCHANGED)
    if rgba is None:raise ValueError('Unreadable isolated document')
    h,w=rgba.shape[:2];mask=rgba[:,:,3] if rgba.shape[2]==4 else np.full((h,w),255,np.uint8)
    alpha=mask[:,:,None]/255;bgr=(rgba[:,:,:3]*alpha+255*(1-alpha)).astype(np.uint8)
    features=extract_text_features(cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY),mask=mask)
    lines=[line.centers/np.array([w-1,h-1]) for line in features.high_confidence_lines if len(line.centers)>=6 and line.width>w*.15]
    # Also use the orthogonal layout when a page's text is sideways.
    transposed=False
    if len(lines)<3:
        rotated=cv2.rotate(bgr,cv2.ROTATE_90_CLOCKWISE);rmask=cv2.rotate(mask,cv2.ROTATE_90_CLOCKWISE)
        f=extract_text_features(cv2.cvtColor(rotated,cv2.COLOR_BGR2GRAY),mask=rmask)
        alternate=[line.centers/np.array([h-1,w-1]) for line in f.high_confidence_lines if len(line.centers)>=6 and line.width>h*.15]
        if len(alternate)>len(lines):lines=alternate;transposed=True
    candidates=[]
    # UVDoc proposes a 2D displacement grid, never synthesizes output content.
    sys.path.insert(0,str(ROOT/'UVDoc'))
    import torch
    from model import UVDocnet
    torch.set_num_threads(2)
    model=UVDocnet(num_filter=32,kernel_size=5)
    model.load_state_dict(torch.load(ROOT/'UVDoc'/'model'/'best_model.pkl',map_location='cpu',weights_only=True)['model_state']);model.eval()
    rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB).astype(np.float32)/255
    with torch.inference_mode():g,_=model(torch.from_numpy(cv2.resize(rgb,(488,712)).transpose(2,0,1)).unsqueeze(0))
    raw=(g[0].permute(1,2,0).numpy()+1)/2
    candidate=lock_boundary(raw)
    if valid(candidate):candidates.append(('UVDoc mesh',candidate))
    calibrated=calibrate_corners(raw)
    if valid(calibrated,edge_extension=True):candidates.append(('UVDoc surface mesh',calibrated))
    # The text/line solver is a second proposal. Its automatic detector is not
    # used: the caller's alpha mask is the sole region-of-interest authority.
    import dewarping.dewarp as dm
    from dewarping.linesegs import PaperRegion
    from dewarping.options import make_options
    from dewarping.model import surface_to_rectified,rectified_to_image
    def trusted_region(image):
        m=cv2.resize(mask,(image.shape[1],image.shape[0]),interpolation=cv2.INTER_NEAREST)
        contours,_=cv2.findContours(m,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        return PaperRegion(m,max(contours,key=cv2.contourArea).reshape(-1,2),None) if contours else None
    def capture(image,params,feat_uv,x_clamp=None,grid=65,interp='linear',warp=None,page_uv=None):
        corners=np.array([[0,0],[w-1,0],[w-1,h-1],[0,h-1]],np.float64)
        q=surface_to_rectified(corners,params)
        if warp is not None:q=warp.uv_to_out(q)
        if not np.isfinite(q).all():return image
        matrix=cv2.getPerspectiveTransform(np.array([[0,0],[1,0],[1,1],[0,1]],np.float32),q.astype(np.float32))
        uv=cv2.perspectiveTransform(IDENTITY.reshape(-1,1,2).astype(np.float64),matrix).reshape(-1,2)
        if warp is not None:uv=warp.out_to_uv(uv)
        back=rectified_to_image(uv,params,x_clamp=x_clamp).reshape(N,N,2)/np.array([w-1,h-1])
        back=lock_boundary(back)
        if valid(back):candidates.append(('Text and line mesh',back))
        return image
    dm.detect_paper_region=trusted_region;dm._render=capture
    try:dm.dewarp_image(bgr,opts=make_options('fast',proc_max_side=800,max_nfev=80,n_outlier_iter=1,use_page_boundary=False))
    except (ValueError,ArithmeticError,np.linalg.LinAlgError):pass
    def evidence_grid(g):
        if not transposed:return g
        rotated=np.rot90(g,k=-1);return np.stack([1-rotated[:,:,1],rotated[:,:,0]],-1)
    from printed_line_mesh import extract_rules, rule_mesh
    rules=extract_rules(cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY))
    ruled=rule_mesh(rules,N)
    if ruled is not None and valid(ruled,edge_extension=True):
        candidates.append(('Printed line alignment',ruled))
    from mesh_ruler import text_rulers,ruler_mesh,circular_landmarks,paragraph_rulers
    gray=cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY)
    text_guides=text_rulers(gray,features)
    vertical_rules=[p[:,::-1] for p in extract_rules(gray.T)]
    vertical_text=[p[:,::-1] for p in text_rulers(gray.T,extract_text_features(gray.T,mask=mask.T))]
    from frame_ruler import frame_rulers,band_rulers
    frame_horizontal,frame_vertical=frame_rulers(gray)
    from printed_frame import printed_frame
    closed_h,closed_v=printed_frame(gray)
    def away_from_frame(guide,frames,axis):
        other=1-axis
        for frame in frames:
            lo=max(guide[:,other].min(),frame[:,other].min())
            hi=min(guide[:,other].max(),frame[:,other].max())
            if hi-lo<.4:continue
            sample=np.linspace(lo,hi,65)
            a=np.interp(sample,guide[:,other],guide[:,axis])
            b=np.interp(sample,frame[:,other],frame[:,axis])
            if np.median(np.abs(a-b))<.015:return False
        return True
    rules=[p for p in rules if away_from_frame(p,closed_h,1)]
    vertical_rules=[p for p in vertical_rules if away_from_frame(p,closed_v,0)]
    frame_horizontal+=closed_h
    frame_vertical+=closed_v
    paragraph_vertical=paragraph_rulers(gray,features)
    paragraph_horizontal=[p[:,::-1] for p in paragraph_rulers(gray.T,extract_text_features(gray.T,mask=mask.T))]
    band_vertical=band_rulers(bgr,paragraph_vertical)
    band_horizontal=[p[:,::-1] for p in band_rulers(bgr.transpose(1,0,2).copy(),[p[:,::-1] for p in paragraph_horizontal])]
    frame_vertical+=band_vertical
    frame_horizontal+=band_horizontal
    from barcode_ruler import barcode_rulers
    barcode_vertical=barcode_rulers(gray)
    barcode_horizontal=[p[:,::-1] for p in barcode_rulers(gray.T)]
    horizontal=[(p,4.) for p in barcode_horizontal]+[(p,2.) for p in frame_horizontal]+[(p,1.2) for p in paragraph_horizontal]+[(p,4.) for p in rules]+[(p,3.) for p in text_guides]
    vertical=[(p,4.) for p in barcode_vertical]+[(p,2.) for p in frame_vertical]+[(p,1.2) for p in paragraph_vertical]+[(p,4.) for p in vertical_rules]+[(p,3.) for p in vertical_text]
    # Require multiple independent rulers before changing the surface.
    if len(frame_horizontal)+len(frame_vertical)+len(rules)+len(text_guides)+len(vertical_rules)+len(vertical_text)>=3 or len(vertical_rules)>=2:
        shared=ruler_mesh(horizontal,vertical,prior=ruled if ruled is not None else None,
            prior_weight=.045)
        if valid(shared,edge_extension=True):candidates.append(('Shared mesh rulers',shared))
    circles=circular_landmarks(gray)
    def preserves_shapes(g):
        for guides,axis in ((barcode_horizontal,1),(barcode_vertical,0)):
            for p in guides:
                if np.std(inverse_points(g,p)[:,axis])>np.std(p[:,axis])+.001:
                    return False
        for p in circles:
            q=(inverse_points(g,p)*[w-1,h-1]).astype(np.float32)
            _,axes,_=cv2.fitEllipse(q.reshape(-1,1,2))
            if min(axes)/max(axes)<.92:return False
        return True
    def ruler_errors(g,observed,axis=1):
        return np.array([np.std(inverse_points(g,p)[:,axis]) for p in observed])
    baseline=line_error(IDENTITY,lines)
    chosen=IDENTITY;method='Outline mesh · interior evidence insufficient';best=baseline
    measurements=[]
    for name,g in candidates:
        if not preserves_shapes(g):continue
        score=line_error(evidence_grid(g),lines)
        measurements.append({'method':name,'line_error':score})
        if name in ('Printed line alignment','Shared mesh rulers'):
            observed=rules if name=='Printed line alignment' else rules+text_guides+paragraph_horizontal+frame_horizontal+barcode_horizontal
            before_values=ruler_errors(IDENTITY,observed)
            after_values=ruler_errors(g,observed)
            axis_errors=[(float(np.mean(before_values)),float(np.mean(after_values)))] if len(before_values) else []
            if name=='Shared mesh rulers' and (vertical_rules or vertical_text or paragraph_vertical or frame_vertical or barcode_vertical):
                vb=ruler_errors(IDENTITY,vertical_rules+vertical_text+paragraph_vertical+frame_vertical+barcode_vertical,0)
                va=ruler_errors(g,vertical_rules+vertical_text+paragraph_vertical+frame_vertical+barcode_vertical,0)
                axis_errors.append((float(np.mean(vb)),float(np.mean(va))))
                before_values=np.r_[before_values,vb]
                after_values=np.r_[after_values,va]
            before=float(np.mean(before_values)) if len(before_values) else None
            after=float(np.mean(after_values)) if len(after_values) else None
            # Every measured ruler must be preserved, not merely the median.
            ruler_ok=bool(len(before_values) and np.all(after_values<=before_values+.0015))
            # Preserve text curvature; independent printed rules also constrain
            # tilt, which a fitted straight-line residual cannot detect.
            text_ok=score is None or best is None or score<=best*1.1+.00005
            measurements[-1].update({'alignment_before':before,'alignment_after':after,'ruler_ok':ruler_ok,'text_ok':text_ok,'axis_errors':axis_errors})
            # Many already-flat text rows must not dilute genuine vertical
            # column bending. Require the same gain on an individual axis,
            # while retaining the per-guide and text preservation gates.
            axis_improved=any(b>.001 and a<b*.35 for b,a in axis_errors)
            if axis_improved and text_ok and ruler_ok:
                chosen=g;method=name;best=score
            continue

        # A learned surface must also preserve independently observed rules.
        # Better text-centre residual alone cannot justify bending table lines.
        structure_ok=True
        for guides,axis in ((rules+frame_horizontal+barcode_horizontal,1),(vertical_rules+frame_vertical+barcode_vertical,0)):
            if guides and np.any(ruler_errors(g,guides,axis)>ruler_errors(IDENTITY,guides,axis)+.0015):
                structure_ok=False
        measurements[-1]['structure_ok']=structure_ok
        if not structure_ok:continue

        # No evidence, no extra deformation. Three independent text lines and
        # meaningful improvement are required; flat text should remain flat.
        if len(lines)>=3 and baseline is not None and baseline>.0008 and score is not None and score<baseline*.9 and (best is None or score<best):chosen=g;method=name;best=score
    # Higher resolution can lose previously reliable grouping. Preserve the
    # working-resolution solution as a proposal, judged by the same independent
    # shape and line evidence rather than silently reverting to no correction.
    if method.startswith('Outline') and working_resolution_path is not None:
        import tempfile,subprocess
        legacy=Path(__file__).resolve().parents[1]/'dewarp_mesh.py'
        with tempfile.TemporaryDirectory(prefix='doctrack-legacy-') as temporary:
            output=Path(temporary)/'mesh.json'
            try:
                done=subprocess.run([sys.executable,str(legacy),str(working_resolution_path),str(output)],
                    capture_output=True,timeout=20)
                if done.returncode==0:
                    previous=json.loads(output.read_text())
                    g=np.asarray(previous['grid']).reshape(N,N,2)
                    score=line_error(evidence_grid(g),lines)
                    preserved=valid(g,edge_extension=True) and preserves_shapes(g)
                    for guides,axis in ((rules+frame_horizontal,1),(vertical_rules+frame_vertical,0)):
                        if guides and np.any(ruler_errors(g,guides,axis)>ruler_errors(IDENTITY,guides,axis)+.0015):preserved=False
                    text_ok=score is None or baseline is None or score<=baseline*1.05+.00005
                    accepted=not previous['method'].startswith('Outline') and preserved and text_ok
                    measurements.append({'method':'Working-resolution proposal','previous_method':previous['method'],
                        'line_error':score,'structure_ok':preserved,'text_ok':text_ok,'accepted':accepted})
                    if accepted:chosen=g;method='Working-resolution '+previous['method'];best=score
                else:measurements.append({'method':'Working-resolution proposal','error':'worker failed'})
            except (OSError,ValueError,subprocess.TimeoutExpired) as error:
                measurements.append({'method':'Working-resolution proposal','error':type(error).__name__})
    return {'rows':N,'columns':N,'grid':chosen.reshape(-1,2).tolist(),'method':method,'lines':len(lines),'printed_rules':len(rules),'text_rulers':len(text_guides),'vertical_rules':len(vertical_rules),'vertical_text_rulers':len(vertical_text),'paragraph_rulers':len(paragraph_horizontal)+len(paragraph_vertical),'band_rulers':len(band_horizontal)+len(band_vertical),'barcode_rulers':len(barcode_horizontal)+len(barcode_vertical),'frame_rulers':len(frame_horizontal)+len(frame_vertical),'circular_landmarks':len(circles),'baseline_error':baseline,'selected_error':best,'candidates':measurements}

if __name__=='__main__':
    result=solve(sys.argv[1],sys.argv[3] if len(sys.argv)>3 else None);Path(sys.argv[2]).write_text(json.dumps(result,allow_nan=False))
