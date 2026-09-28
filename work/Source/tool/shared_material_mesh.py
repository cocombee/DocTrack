"""One shared page field with simultaneous material and stroke constraints.

The general page model supplies the poses. Every original word/unknown body
is constrained together in one field; complete rulings supply normal pins.
Only unprinted stock is relaxed. This is not a sequence of local corrections.
The caller still audits every original footprint, stroke and continuous cell.
"""
import cv2
import numpy as np
from local_receipt_mesh import sample
from composite_mesh_inverse import inverse_retained
from composite_mesh_sampling import MeshSampler
from material_pixel_footprints import paired_points,capture
from free_cell_jacobian import repair_free_cells


def barcode_material_owners(gray,safe,ledger):
    """Register actual parallel bar bodies on the initial measurement only.

    Already working bars keep their complete shape. Noisy ridge endpoints
    cannot authorize straightening a rounded or thick printed bar end.
    """
    h,w=gray.shape;basis=np.array([w-1,h-1]);scale=max(h,w)/1280
    traces=[np.asarray(s['source_points'])*basis for s in ledger['complete_printed_strokes'] if s['axis']==1]
    if len(traces)<6:return [],dict(accepted=False,reason='No supported barcode family')
    tops=np.array([q[:,1].min() for q in traces]);bottoms=np.array([q[:,1].max() for q in traces])
    top,bottom=float(np.median(tops)),float(np.median(bottoms));height=bottom-top
    if height<40*scale or np.ptp(tops)>height*.2 or np.ptp(bottoms)>height*.2:
        return [],dict(accepted=False,reason='Parallel bars do not share a supported extent')
    contrast=cv2.GaussianBlur(gray.astype(np.float32),(0,0),max(2,7*scale))-gray
    mask=((contrast>6)&(safe>250)).astype(np.uint8)
    _,labels,stats,_=cv2.connectedComponentsWithStats(mask)
    low=min(q[:,0].min() for q in traces)-height*.75
    high=max(q[:,0].max() for q in traces)+height*.75
    records=[]
    for index,(x,y,cw,ch,area) in enumerate(stats[1:],1):
        if (area<40*scale or ch<height*.75 or ch>height*1.3 or cw>height*.3
                or x<low or x+cw>high or abs(y-top)>height*.2
                or abs(y+ch-bottom)>height*.2):continue
        yy,xx=np.where(labels[y:y+ch,x:x+cw]==index);points=np.c_[xx+x,yy+y]
        if np.count_nonzero(contrast[points[:,1],points[:,0]]>16)<len(points)*.4:continue
        records.append(dict(label=index,points=points.astype(float),traces=[]))
    for trace in traces:
        xy=np.rint(trace).astype(int)
        candidates=[]
        for index,record in enumerate(records):
            overlap=np.mean(labels[xy[:,1],xy[:,0]]==record['label'])
            if overlap>.65:candidates.append((overlap,index))
        if len(candidates)!=1:return [],dict(accepted=False,reason='A complete bar trace has ambiguous material')
        records[candidates[0][1]]['traces'].append(trace)
    clouds=[np.vstack([q['points'],*q['traces']])/basis for q in records]
    identity=dict(rows=2,columns=2,grid=[[0,0],[1,0],[0,1],[1,1]])
    owners=capture(identity,clouds,[w,h],first_owner=len(ledger['retained_pixel_footprints']),kind='unknown')
    return owners,dict(accepted=True,bar_bodies=len(owners),complete_vertical_traces=len(traces),
                       original_bar_pixels=sum(len(q['points']) for q in records),new_registration_basis='initial identity')


def rigid_destination(source,proposed):
    a=source-source.mean(0);b=proposed-proposed.mean(0)
    u,_,v=np.linalg.svd(a.T@b)
    rotation=u@v
    if np.linalg.det(rotation)<=0:raise ValueError('Material pose reflects print')
    destination=a@rotation+proposed.mean(0)
    return destination,rotation,source.mean(0),proposed.mean(0)


def cell_nodes(points,shape):
    nr,nc=shape
    xy=np.clip(points*[nc-1,nr-1],0,[nc-1-1e-9,nr-1-1e-9])
    lo=np.floor(xy).astype(int);ix,iy=lo.T
    return np.unique(np.r_[iy*nc+ix,iy*nc+ix+1,(iy+1)*nc+ix,(iy+1)*nc+ix+1])


def fit_material_constraints(seed,owners,pixel_size,shape,*,halo_cells=0):
    """Merge only bodies that need the same lattice node, before solving."""
    basis=np.asarray(pixel_size,float)-1;nr,nc=shape
    source=[paired_points(owner)[0] for owner in owners]
    halos=[owner.get('guard_halo_cells',halo_cells) for owner in owners]
    if any(not isinstance(value,int) or value<0 for value in halos):
        raise ValueError('Each material guard halo must be a nonnegative cell count')
    proposed=np.split(inverse_retained(MeshSampler(seed),np.vstack(source)),
                      np.cumsum([len(q) for q in source])[:-1])
    groups=[[i] for i in range(len(source))]
    for iteration in range(len(source)):
        node_owner=np.full(nr*nc,-1,int);poses=[];collisions=[]
        for index,group in enumerate(groups):
            original=np.vstack([source[i] for i in group])*basis
            initial=np.vstack([proposed[i] for i in group])*basis
            destination,rotation,origin,center=rigid_destination(original,initial)
            normalized=destination/basis
            if np.any(normalized<0) or np.any(normalized>1):
                raise ValueError('A material pose leaves the page')
            parts=[];cursor=0
            for owner_index in group:
                count=len(source[owner_index])
                nodes=cell_nodes(normalized[cursor:cursor+count],shape)
                cursor+=count
                halo=halos[owner_index]
                if halo:
                    mask=np.zeros((nr,nc),np.uint8);mask.reshape(-1)[nodes]=1
                    mask=cv2.dilate(mask,np.ones((halo*2+1,)*2,np.uint8))
                    nodes=np.flatnonzero(mask.reshape(-1))
                parts.append(nodes)
            nodes=np.unique(np.concatenate(parts))
            overlap=np.unique(node_owner[nodes]);overlap=overlap[overlap>=0]
            collisions.extend((int(other),index) for other in overlap)
            node_owner[nodes]=index
            poses.append((group,nodes,rotation,origin,center))
        if not collisions:return poses,dict(material_clusters=len(groups),
            original_owners=len(owners),shared_node_merges=iteration,
            largest_cluster=max(map(len,groups)))
        roots=list(range(len(groups)))
        def root(i):
            while roots[i]!=i:roots[i]=roots[roots[i]];i=roots[i]
            return i
        for a,b in collisions:roots[root(b)]=root(a)
        merged={}
        for i,group in enumerate(groups):merged.setdefault(root(i),[]).extend(group)
        groups=list(merged.values())
    raise ValueError('Material cluster ownership did not converge')


def shared_material_field(seed,ledger,pixel_size,*,shape=(1025,1025),iterations=96,
                          material_halo_cells=0,protected_vertical=False,
                          repair_near_folds=False):
    nr,nc=shape;basis=np.asarray(pixel_size,float)-1
    if protected_vertical:
        # Never skip a normal constraint merely because a flag claims the
        # bar is protected: each original trace member must have a body.
        centers=np.vstack([paired_points(owner)[0][::5]
            for owner in ledger['retained_pixel_footprints']])
        registered={tuple(p) for p in np.rint(centers*1e10).astype(np.int64)}
        for stroke in ledger['complete_printed_strokes']:
            if stroke['axis']==1 and any(tuple(p) not in registered for p in
                    np.rint(np.asarray(stroke['source_points'])*1e10).astype(np.int64)):
                raise ValueError('A protected vertical stroke lacks complete material ownership')
    u,v=np.meshgrid(np.linspace(0,1,nc),np.linspace(0,1,nr))
    identity=np.stack([u,v],-1)
    initial=sample(np.asarray(seed['grid']).reshape(seed['rows'],seed['columns'],2),
                   identity.reshape(-1,2)).reshape(nr,nc,2)
    poses,ownership=fit_material_constraints(seed,ledger['retained_pixel_footprints'],pixel_size,shape,
                                             halo_cells=material_halo_cells)
    fixed=np.zeros((nr,nc),bool);target=initial.copy()
    for group,nodes,rotation,origin,center in poses:
        destination=identity.reshape(-1,2)[nodes]*basis
        target.reshape(-1,2)[nodes]=((destination-center)@rotation.T+origin)/basis
        fixed.reshape(-1)[nodes]=True
    target[0]=identity[0];target[-1]=identity[-1]
    target[:,0]=identity[:,0];target[:,-1]=identity[:,-1]
    fixed[[0,-1],:]=True;fixed[:,[0,-1]]=True
    constraints=[]
    for stroke in ledger['complete_printed_strokes']:
        if protected_vertical and stroke['axis']==1:continue
        source=np.asarray(stroke['source_points'],float)
        axis=1-int(stroke['axis'])
        destination=source.copy();destination[:,axis]=source[:,axis].mean()
        xy=np.clip(destination*[nc-1,nr-1],0,[nc-1-1e-9,nr-1-1e-9])
        lo=np.floor(xy).astype(int);ix,iy=lo.T;fx,fy=(xy-lo).T
        nodes=np.c_[iy*nc+ix,iy*nc+ix+1,(iy+1)*nc+ix,(iy+1)*nc+ix+1]
        weights=np.c_[(1-fx)*(1-fy),fx*(1-fy),(1-fx)*fy,fx*fy]
        gradients=weights*(~fixed.reshape(-1)[nodes])
        norm=(gradients*gradients).sum(1)
        if np.any(norm<1e-12):raise ValueError('A ruling conflicts with material ownership')
        constraints.append((nodes,weights,gradients,norm,source[:,axis],axis))
    # All normals of the same orientation solve together. Batching avoids a
    # full-page allocation per ruling on each shared-field iteration.
    batched=[]
    for axis in (0,1):
        selected=[q for q in constraints if q[-1]==axis]
        if selected:batched.append((*[np.concatenate([q[i] for q in selected]) for i in range(5)],axis))
    stroke_count=len(constraints);constraints=batched
    field=initial.copy();field[fixed]=target[fixed]
    report={**ownership,'grid_shape':list(shape),'simultaneous_material_constraints':True,
            'complete_stroke_constraints':stroke_count,'local_corrections':0,
            'material_halo_cells':material_halo_cells,
            'protected_vertical_strokes':sum(s['axis']==1 for s in ledger['complete_printed_strokes']) if protected_vertical else 0}
    for iteration in range(iterations):
        correction=field-initial
        relaxed=cv2.GaussianBlur(correction,(5,5),0,borderType=cv2.BORDER_REFLECT_101)
        field=initial+.6*relaxed+.4*correction
        field[fixed]=target[fixed]
        for _ in range(3):
            for nodes,weights,gradients,norm,source,axis in constraints:
                flat=field[:,:,axis].reshape(-1)
                error=(flat[nodes]*weights).sum(1)-source
                delta=np.bincount(nodes.reshape(-1),
                    weights=(-error[:,None]*gradients/norm[:,None]).reshape(-1),minlength=nr*nc)
                count=np.bincount(nodes.reshape(-1),weights=(gradients>0).reshape(-1),minlength=nr*nc)
                flat+=.95*delta/np.maximum(count,1)
                field[:,:,axis]=flat.reshape(nr,nc)
        field[fixed]=target[fixed]
    # Finish all guide equations together, without another smoothing pass
    # reintroducing normal error at their actual material samples.
    for _ in range(64):
        for nodes,weights,gradients,norm,source,axis in constraints:
            flat=field[:,:,axis].reshape(-1)
            error=(flat[nodes]*weights).sum(1)-source
            delta=np.bincount(nodes.reshape(-1),
                weights=(-error[:,None]*gradients/norm[:,None]).reshape(-1),minlength=nr*nc)
            count=np.bincount(nodes.reshape(-1),weights=(gradients>0).reshape(-1),minlength=nr*nc)
            flat+=.95*delta/np.maximum(count,1)
            field[:,:,axis]=flat.reshape(nr,nc)
        field[fixed]=target[fixed]
    if repair_near_folds:
        field,repair=repair_free_cells(field,fixed,pixel_size)
        report['free_cell_repair']=repair
    residual=max((float(abs((field[:,:,axis].reshape(-1)[nodes]*weights).sum(1)-source).max()*basis[axis])
        for nodes,weights,_,_,source,axis in constraints),default=0.)
    du,dv=np.diff(field,axis=1),np.diff(field,axis=0)
    minimum=float(min(((a[...,0]*b[...,1]-a[...,1]*b[...,0])*(nr-1)*(nc-1)).min()
        for a in (du[:-1],du[1:]) for b in (dv[:,:-1],dv[:,1:])))
    report.update(iterations=iterations,maximum_ruling_normal_pin_error_px=residual,
        minimum_continuous_bilinear_jacobian=minimum,
        boundary_maximum_displacement=float(abs((field-identity)[fixed&((u==0)|(u==1)|(v==0)|(v==1))]).max()),
        accepted=bool(np.isfinite(field).all() and minimum>.2 and field.min()>=0 and field.max()<=1 and residual<.03
                      and (not repair_near_folds or report['free_cell_repair']['accepted'])))
    return field,report
