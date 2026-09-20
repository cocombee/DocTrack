"""Experimental source matte cleanup with observed inside/outside material."""
import cv2
import numpy as np


def fringe_mask(image, segments):
    h,w=image.shape[:2];remove=np.zeros((h,w),np.uint8)
    for s in segments:
        if s.get('isImageClip'):continue
        def point(k):return np.array([s[k]['x']*w-.5,s[k]['y']*h-.5])
        a,b=point('start'),point('end');c=point('control1') if s.get('control1') else a;z=point('control2') if s.get('control2') else b
        ts=np.linspace(.005,.995,max(12,round(np.linalg.norm(b-a))))
        t=ts[:,None]
        q=(1-t)**3*a+3*(1-t)**2*t*c+3*(1-t)*t*t*z+t**3*b
        tangent=np.gradient(q,axis=0);normal=np.column_stack([-tangent[:,1],tangent[:,0]])
        norm=np.linalg.norm(normal,axis=1);normal/=np.maximum(norm[:,None],1e-8)
        offsets=np.arange(-6,6.01,.25)
        pts=q[:,None,:]+normal[:,None,:]*offsets[None,:,None]
        values=cv2.remap(image,pts[:,:,0].astype('float32'),pts[:,:,1].astype('float32'),cv2.INTER_LINEAR).astype(float)
        exterior=values[:,offsets<-3].mean(1);interior=values[:,offsets>3].mean(1)
        vector=interior-exterior;length=np.sum(vector*vector,axis=1)
        stable=(values[:,offsets<-3].std(1).max(1)<12)&(values[:,offsets>3].std(1).max(1)<12)&(length>75**2)
        stable&=(pts.min(1)[:,0]>=0)&(pts.min(1)[:,1]>=0)&(pts.max(1)[:,0]<w-1)&(pts.max(1)[:,1]<h-1)
        gradient=np.linalg.norm(values[:,2:]-values[:,:-2],axis=2)
        edge=offsets[1:-1][gradient.argmax(1)]
        safe=stable&(edge>.25)&(edge<4.5)
        safe&=np.convolve(safe.astype(int),np.ones(11),mode='same')>=8
        for i in np.flatnonzero(safe):
            x,y=np.rint(q[i]).astype(int)
            for yy in range(max(0,y-5),min(h,y+6)):
                for xx in range(max(0,x-5),min(w,x+6)):
                    delta=np.array([xx,yy])-q[i];inset=np.dot(delta,normal[i]);along=abs(normal[i,0]*delta[1]-normal[i,1]*delta[0])
                    if not -.5<=inset<=4.5 or along>.8:continue
                    pixel=image[yy,xx].astype(float)
                    coverage=np.dot(pixel-exterior[i],vector[i])/length[i]
                    residual=np.linalg.norm(pixel-exterior[i]-coverage*vector[i])
                    if coverage<.85 and residual<20:remove[yy,xx]=255
    return remove


if __name__=='__main__':
    from pathlib import Path
    import json,sys
    root=Path(__file__).resolve().parents[3]
    output=root/'Scanner Integration'/'Source Matte Candidate 49';output.mkdir(exist_ok=True)
    m=json.loads((root/'Data/manifest.json').read_text())
    diagnosis={r['index']:r for r in json.loads((root/'Scanner Integration'/'Outline Diagnosis 48'/'Report.json').read_text())}
    report=[]
    for i in map(int,sys.argv[1].split(',')):
        e=m[i-1];im=cv2.imread(str(root/'Data'/e['image']));d=json.loads((root/'Scanner Integration/Predictions Physical V18'/f"{e['name']}.json").read_text())
        evidence=diagnosis[i]
        protected=bool(evidence['roundedCorners']) or evidence['content']=='identityDocument' or evidence['format'].startswith(('ID /','Passport','Business card'))
        mask=np.zeros(im.shape[:2],np.uint8) if protected else fringe_mask(im,d['boundarySegments'])
        report.append({'index':i,'protected':protected,'removed':int(np.count_nonzero(mask))})
        rgba=cv2.cvtColor(im,cv2.COLOR_BGR2BGRA);rgba[:,:,3]=255-mask
        cv2.imwrite(str(output/(e['name']+'.png')),rgba)
        print(i,int(np.count_nonzero(mask)))
    (output/'Report.json').write_text(json.dumps(report,indent=2))
