"""Repeated text margins constrain vertical alignment without guessing size."""
import cv2
import numpy as np
import local_receipt_mesh as prior


def margin_tracks(gray,grid):
    h,w=gray.shape
    textline=prior.reference.textline
    groups=textline._link_ccs_into_lines(textline._extract_ccs(textline.binarize(gray)))
    rows=[]
    for group in groups:
        if len(group)<12 or not textline._is_high_confidence_line(group):continue
        first,last=group[0],group[-1]
        points=np.array([[first.x0,first.center[1]],[last.x1,last.center[1]]])/[w-1,h-1]
        mapped=prior.inverse(grid,points)
        if mapped[1,0]-mapped[0,0]<.22:continue
        rows.append({'points':points,'mapped':mapped,
                     'height':float(np.median([cc.height for cc in group])/(h-1))})
    tracks=[]
    for side in (0,1):
        chains=[]
        for row in sorted(rows,key=lambda r:r['mapped'][side,1]):
            q=row['mapped'][side]
            choices=[]
            for i,chain in enumerate(chains):
                old=chain[-1]['mapped'][side]
                if (q[1]-old[1]<.12 and abs(q[0]-np.median([r['mapped'][side,0] for r in chain]))<.014
                        and abs(row['height']/np.median([r['height'] for r in chain])-1)<.35):
                    choices.append((abs(q[0]-old[0]),i))
            if choices:chains[min(choices)[1]].append(row)
            else:chains.append([row])
        for chain in chains:
            if len(chain)<8:continue
            points=np.array([row['points'][side] for row in chain])
            mapped=prior.inverse(grid,points)
            x,y=mapped.T
            if np.ptp(y)<.12:continue
            coefficients=np.polyfit(y,x,3)
            for _ in range(5):
                residual=np.abs(np.polyval(coefficients,y)-x)*w
                coefficients=np.polyfit(y,x,3,w=np.minimum(1,1.5/(residual+1e-9)))
            fit=np.polyval(coefficients,y)
            explained=1-np.mean((fit-x)**2)/max(np.var(x),1e-16)
            excursion=np.ptp(fit)*w
            if explained<.65 or excursion<2:continue
            tracks.append({'side':side,'points':points,'mapped':mapped,'coefficients':coefficients,
                           'height':float(np.median([row['height'] for row in chain])),
                           'row_bounds':np.array([row['mapped'] for row in chain]),
                           'explained':float(explained),'excursion_px':float(excursion)})
    return tracks


def correct_columns(grid,gray,texts,rules,bars,envelopes,circles,orthogonal_structures):
    h,w=gray.shape
    proposals,regions=[],[]
    for track in margin_tracks(gray,grid):
        q=prior.inverse(grid,track['points'])
        target=float(np.mean(q[:,0]));glyph_height=track['height']
        y0,y1=q[:,1].min(),q[:,1].max()
        bounds=track['row_bounds']
        x0,x1=bounds[:,:,0].min(),bounds[:,:,0].max()
        outer_x0=max(.005,x0-.025);outer_x1=min(.995,x1+.025)
        outer_y0=max(.005,y0-glyph_height*2);outer_y1=min(.995,y1+glyph_height*2)
        rows,columns=grid.shape[:2];nr=max(rows,513);nc=max(columns,513)
        if (nr-1)%(rows-1):nr=(rows-1)*4+1
        if (nc-1)%(columns-1):nc=(columns-1)*4+1
        u,v=np.meshgrid(np.linspace(0,1,nc),np.linspace(0,1,nr))
        dense=prior.sample(grid,np.column_stack([u.ravel(),v.ravel()])).reshape(nr,nc,2)
        drift=np.polyval(track['coefficients'],np.clip(v,y0,y1))-target
        # Keep the opposite margin anchored. Two supported sides can therefore
        # recover row width; one side cannot translate an entire layout.
        if track['side']==1:
            across=np.clip((u-x0)/max(target-x0,1e-9),0,1)
        else:
            across=np.clip((x1-u)/max(x1-target,1e-9),0,1)
        weight=(prior.window(u,outer_x0,x0,x1,outer_x1)*
                prior.window(v,outer_y0,y0-glyph_height*.5,y1+glyph_height*.5,outer_y1))
        source=np.column_stack([(u+drift*across*weight).ravel(),v.ravel()])
        candidate=prior.sample(grid,source).reshape(nr,nc,2)
        protected=rules+bars+envelopes+circles+orthogonal_structures
        movement=0.
        if protected:
            p=np.vstack(protected)
            movement=float(np.max(np.abs((prior.inverse(candidate,p)-prior.inverse(grid,p))*[w,h])))
        # Correcting vertical alignment may change x positions of text. Its
        # already accepted horizontal alignment and vertical spacing stay fixed.
        horizontal_movement=0.
        if texts:
            p=np.vstack(texts)
            horizontal_movement=float(np.max(np.abs(prior.inverse(candidate,p)[:,1]-
                                                     prior.inverse(grid,p)[:,1]))*h)
        after=prior.inverse(candidate,track['points'])
        before_fit=np.polyval(np.polyfit(q[:,1],q[:,0],3),q[:,1])
        after_fit=np.polyval(np.polyfit(after[:,1],after[:,0],3),after[:,1])
        before_error=float(np.std(before_fit)*w);after_error=float(np.std(after_fit)*w)
        accepted=bool(prior.valid(candidate) and movement<.1 and horizontal_movement<.1
                      and after_error<before_error*.35 and np.std(after[:,0])<=np.std(q[:,0])+.1/w)
        proposals.append({'kind':'vertical_text_margin','side':track['side'],'rows':len(q),
                          'excursion_px':track['excursion_px'],'explained':track['explained'],
                          'accepted':accepted,'before_px':before_error,'after_px':after_error,
                          'protected_shift_px':movement,'horizontal_shift_px':horizontal_movement})
        if accepted:
            grid=candidate
            regions.append([float(np.floor(outer_x0*(nc-1))/(nc-1)),
                            float(np.floor(outer_y0*(nr-1))/(nr-1)),
                            float(np.ceil(outer_x1*(nc-1))/(nc-1)),
                            float(np.ceil(outer_y1*(nr-1))/(nr-1))])
    return grid,proposals,regions
