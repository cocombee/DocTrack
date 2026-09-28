"""Experimental shared row displacement with complete rigid word bodies.

Native boxes associate a row; two measured glyph profiles establish its bend.
The row supplies one correction. Smooth blank gaps connect rigid word poses.
Existing owners and references are never rewritten by measurement or proposal.
"""
import cv2
import numpy as np


def smooth(value):
    value=np.clip(value,0,1)
    return value*value*(3-2*value)


def measured_row(gray,safe,quads,groups):
    gray,safe=np.asarray(gray),np.asarray(safe)
    if gray.ndim!=2 or min(gray.shape)<16 or safe.shape!=gray.shape or not np.isfinite(gray).all():
        raise ValueError('Whole text row needs paired finite raster evidence')
    h,w=gray.shape;basis=np.array([w-1,h-1]);scale=max(h,w)/1280
    owners=[]
    for quad in quads:
        quad=np.asarray(quad,float)
        if quad.shape!=(4,2) or not np.isfinite(quad).all():return None
        matches=[i for i,g in enumerate(groups) if np.all(g.mean(0)>=quad.min(0)-.002)
                 and np.all(g.mean(0)<=quad.max(0)+.002)]
        if len(matches)!=1:return None
        owners+=matches
    if len(owners)<3 or len(set(owners))!=len(owners):return None
    owners.sort(key=lambda i:float(groups[i][:,0].mean()))
    mask=np.zeros_like(gray,np.uint8)
    xy=np.rint(np.vstack([groups[i] for i in owners])*basis).astype(int)
    if np.any(xy<0) or np.any(xy>basis) or np.any(safe[xy[:,1],xy[:,0]]<=250):return None
    mask[xy[:,1],xy[:,0]]=1
    contrast=cv2.GaussianBlur(gray.astype(np.float32),(0,0),max(2,7*scale))-gray
    profiles=[]
    for threshold in (10,16):
        _,labels,stats,_=cv2.connectedComponentsWithStats(((mask>0)&(contrast>threshold)).astype(np.uint8))
        anchors=[]
        for index,(x,y,cw,ch,area) in enumerate(stats[1:],1):
            if not (6*scale<ch<38*scale and 2<cw<ch*2.4 and area>15*scale):continue
            cc=labels[y:y+ch,x:x+cw]==index
            bottoms=[np.flatnonzero(cc[:,xx])[-1] for xx in range(cw) if cc[:,xx].any()]
            anchors.append([x+cw/2,y+np.quantile(bottoms,.85),ch])
        if len(anchors)<12:return None
        anchors=np.asarray(anchors)
        height=float(np.median(anchors[:,2]))
        anchors=anchors[abs(anchors[:,2]-height)<height*.25]
        if len(anchors)<12 or np.ptp(anchors[:,0])<.3*w:return None
        origin=float(anchors[:,0].mean());span=float(np.ptp(anchors[:,0]))
        xx=(anchors[:,0]-origin)/span
        coefficients=np.polyfit(xx,anchors[:,1],2)
        residual=anchors[:,1]-np.polyval(coefficients,xx)
        # Glyphs retain all their pixels. Natural descenders only affect the
        # fit's support selection, never material membership or preservation.
        support=abs(residual-np.median(residual))<=3*scale
        if (support.mean()<.75 or support.sum()<12
                or np.ptp(anchors[support,0])<.8*span):return None
        coefficients=np.polyfit(xx[support],anchors[support,1],2)
        residual=anchors[support,1]-np.polyval(coefficients,xx[support])
        if np.std(residual)>1.5*scale:return None
        profiles.append(dict(threshold=threshold,points=anchors[:,:2].tolist(),
            support_indices=np.flatnonzero(support).tolist(),origin_px=origin,span_px=span,
            coefficients=coefficients.tolist(),scatter_px=float(np.std(residual)),height_px=height,
            supported_domain_px=[float(anchors[support,0].min()),float(anchors[support,0].max())]))
    x=np.linspace(max(min(p[0] for p in q['points']) for q in profiles),
                  min(max(p[0] for p in q['points']) for q in profiles),128)
    curves=[np.polyval(q['coefficients'],(x-q['origin_px'])/q['span_px']) for q in profiles]
    if np.max(abs(curves[0]-curves[1]))>scale:return None
    points=[np.asarray(q['points']) for q in profiles]
    correspondence=abs(points[0][:,0,None]-points[1][None,:,0]).min(1)<min(q['height_px'] for q in profiles)*.5
    if correspondence.mean()<.75:return None
    centers=np.array([groups[i][:,0].mean()*basis[0] for i in owners])
    if any(np.any(centers<q['supported_domain_px'][0]) or
           np.any(centers>q['supported_domain_px'][1]) for q in profiles):return None
    levels=np.mean([np.polyval(q['coefficients'],(centers-q['origin_px'])/q['span_px']) for q in profiles],axis=0)
    target=float(np.median(levels));backward=levels-target
    if np.ptp(levels)<2*scale or np.max(abs(backward))>24*scale:return None
    return dict(owners=owners,backward_normal_px=backward.tolist(),target_px=target,
                independent_profiles=profiles,shared_row_excursion_px=float(np.ptp(levels)),
                ink_or_owner_points_changed=False)


def query_for_row(groups,evidence,pixel_size,rulers,*,resolution=(4097,1025),padding=48):
    basis=np.asarray(pixel_size,float)
    nr,nc=resolution
    if min(nr,nc)<2 or basis.shape!=(2,) or np.any(basis<=0):
        raise ValueError('Shared row requires a valid raster and grid resolution')
    # Cover every half-pixel body corner and its complete interpolation cell.
    # A fixed box halo can falsely overlap adjacent words across blank stock.
    body_padding=.5+basis[0]/(nc-1)+1e-6
    owners=evidence['owners'];delta=np.asarray(evidence['backward_normal_px'])
    if len(owners)!=len(delta) or not np.isfinite(delta).all():raise ValueError('Invalid shared row motion')
    bands=[];destinations=[]
    for owner,dy in zip(owners,delta):
        body=np.asarray(groups[owner])*basis
        destination=body-[0,dy]
        destinations.append(destination)
        bands.append([body[:,0].min()-body_padding,body[:,0].max()+body_padding])
    bands=np.asarray(bands)
    if np.any(bands[1:,0]<=bands[:-1,1]):return None,{'rejected':'word body intervals overlap'}
    body=np.vstack(destinations);core_low=body.min(0)-2;core_high=body.max(0)+2
    core_low[0]=bands[0,0];core_high[0]=bands[-1,1]
    low=np.maximum(0,core_low-padding);high=np.minimum(basis,core_high+padding)
    # Ordered rulings and actual glyphs outside this row bound free stock.
    fixed=[(np.asarray(g)*basis,'ink_cloud',i) for i,g in enumerate(groups) if i not in owners]
    fixed.extend((np.asarray(q['points'])*basis,q.get('geometry','trace'),i)
                 for i,q in enumerate(rulers))
    stock_constraints=[]
    for cloud,kind,index in fixed:
        if kind!='ink_cloud':
            # Ordered rulings own their connecting strokes; unordered glyph
            # clouds own actual pixels only, never empty rectangle interiors.
            cloud=np.vstack([np.linspace(a,z,max(2,int(np.ceil(abs(z-a).max()))+1))
                             for a,z in zip(cloud[:-1],cloud[1:])])
        cloud=cloud[(cloud[:,0]>=low[0])&(cloud[:,0]<=high[0])]
        if not len(cloud):continue
        inside=(cloud[:,1]>=core_low[1])&(cloud[:,1]<=core_high[1])
        if inside.any() and np.all(cloud[inside,0]<core_low[0]):
            low[0]=max(low[0],cloud[inside,0].max()+1);inside[:]=False
        elif inside.any() and np.all(cloud[inside,0]>core_high[0]):
            high[0]=min(high[0],cloud[inside,0].min()-1);inside[:]=False
        if inside.any() and any(np.all(cloud.min(0)>=[bands[j,0],core_low[1]]) and
                 np.all(cloud.max(0)<=[bands[j,1],core_high[1]]) for j in range(len(bands))):
            # Separate compact unknown ink may move inside the same plateau.
            # Every original unknown body still needs its independent audit.
            continue
        if inside.any():
            return None,dict(rejected='unbound print crosses shared row stock',
                geometry=kind,index=index,actual_crossing_points=int(inside.sum()),
                bounds_px=[cloud.min(0).tolist(),cloud.max(0).tolist()])
        above=cloud[cloud[:,1]<core_low[1],1]
        below=cloud[cloud[:,1]>core_high[1],1]
        if len(above) and above.max()+1>low[1]:
            low[1]=above.max()+1
            stock_constraints.append(dict(index=index,geometry=kind,side='above',boundary_px=float(low[1])))
        if len(below) and below.min()-1<high[1]:
            high[1]=below.min()-1
            stock_constraints.append(dict(index=index,geometry=kind,side='below',boundary_px=float(high[1])))
    if np.any(core_low<=low) or np.any(core_high>=high):
        return None,dict(rejected='row has insufficient safe stock',
            core_px=[core_low.tolist(),core_high.tolist()],
            stock_px=[low.tolist(),high.tolist()],stock_constraints=stock_constraints)
    x=np.arange(nc)/(nc-1)*basis[0];y=np.arange(nr)/(nr-1)*basis[1]
    normal=np.empty(nc)
    normal[x<=bands[0,0]]=delta[0];normal[x>=bands[-1,1]]=delta[-1]
    for j,(a,z) in enumerate(bands):normal[(x>=a)&(x<=z)]=delta[j]
    for j in range(len(bands)-1):
        a,z=bands[j,1],bands[j+1,0];inside=(x>a)&(x<z)
        t=smooth((x[inside]-a)/(z-a))
        normal[inside]=delta[j]*(1-t)+delta[j+1]*t
    normal*=smooth((x-low[0])/(core_low[0]-low[0]))*smooth((high[0]-x)/(high[0]-core_high[0]))
    weight=smooth((y-low[1])/(core_low[1]-low[1]))*smooth((high[1]-y)/(high[1]-core_high[1]))
    u,v=np.meshgrid(x/basis[0],y/basis[1])
    query=np.stack([u,v+weight[:,None]*normal[None,:]/basis[1]],axis=-1)
    if np.any(np.diff(query[:,:,1],axis=0)<=0):return None,{'rejected':'shared row stock is not ordered'}
    return query,dict(accepted=True,region=np.r_[low/basis,high/basis].tolist(),
                     complete_word_plateaus_px=bands.tolist(),smooth_real_blank_gaps=True,
                     body_padding_px=float(body_padding),
                     common_row_target_px=evidence['target_px'])
