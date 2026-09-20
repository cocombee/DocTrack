from pathlib import Path
import json,numpy as np,os,shutil
root=Path(__file__).resolve().parents[2]
p=root/'Data/labels/train/bill_segmentation_002194.json'
backup=root/'Outline Review/First Image Accepted Before Extra Inset.json'
if not backup.exists():shutil.copy2(p,backup)
d=json.loads(backup.read_text());nodes=d['fullPolygon'];a=np.array([[v['x'],v['y']] for v in nodes])*640
hin=np.array([v['hIn'] for v in nodes])*640;hout=np.array([v['hOut'] for v in nodes])*640

def normal(v):
 v=v/np.linalg.norm(v,axis=-1,keepdims=True);return np.stack([-v[...,1],v[...,0]],axis=-1)
ni=normal(a-hin);no=normal(hout-a);new=a+2*(ni+no)/(1+(ni*no).sum(axis=1))[:,None]
t=np.linspace(0,1,301)[:,None];b0=(1-t)**3;b1=3*(1-t)**2*t;b2=3*(1-t)*t*t;b3=t**3
controls=[]
for i in range(len(a)):
 j=(i+1)%len(a);b=hout[i];c=hin[j]
 q=b0*a[i]+b1*b+b2*c+b3*a[j]
 deriv=3*(1-t)**2*(b-a[i])+6*(1-t)*t*(c-b)+3*t*t*(a[j]-c)
 target=q+2*normal(deriv)
 controls.append(np.linalg.lstsq(np.hstack([b1[1:-1],b2[1:-1]]),(target-b0*new[i]-b3*new[j])[1:-1],rcond=None)[0])
for i,node in enumerate(nodes):
 node.update(x=round(new[i,0]/640,8),y=round(new[i,1]/640,8),hIn=(controls[i-1][1]/640).round(8).tolist(),hOut=(controls[i][0]/640).round(8).tolist(),smooth=True,broken=True)
d['corners']=[(new[i]/640).round(8).tolist() for i in [0,2,4,6]]
d['inset_px']=4.0
d['edge_rule']='Simple main paper outline; retain real bends and corners, ignore tiny nicks/tears; additional 2px inward from the accepted outline, approximately 4px total inset.'
d['edge_refined']='Eight-point outline with gentle curves, extra inset applied consistently around all sides.'
s=json.dumps(d,indent=2)+'\n';tmp=p.with_suffix('.json.tmp');tmp.write_text(s);os.replace(tmp,p)
(root/'Outline Review/First Image After.json').write_text(s)
print('First image saved: 8 points, additional 2px inset, tiny nicks simplified.')
