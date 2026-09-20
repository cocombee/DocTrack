from pathlib import Path
from PIL import Image
import numpy as np,json,os
root=Path(__file__).resolve().parents[2]
p=root/'Data/labels/train/filtered_document_segmentation_000995.json'
backup=root/'Outline Review/Sixth Image Rejected Nine Points.json'
if not backup.exists():backup.write_text(p.read_text())
im=np.array(Image.open(root/'Data/originals/train/filtered_document_segmentation_000995.jpg').convert('RGB'),float).mean(2)
# Four corners plus five crease-related direction changes on each side.
a=np.array([(20,87),(531,57),(539,95),(557,220),(561,325),(569,398),(573,485),(584,537),(79,580),(75,432),(69,382),(65,312),(43,232),(38,174)],float)
controls=[];curved=[]
for i in range(len(a)):
 j=(i+1)%len(a);A,B=a[i],a[j];d=B-A
 c1=A+d/3;c2=A+2*d/3
 if i not in [0,7]:
  ys=np.linspace(A[1],B[1],max(12,int(abs(d[1]))+1));xs=[]
  for y in ys:
   yi=int(round(y));expected=A[0]+(y-A[1])/d[1]*d[0]
   scores=[]
   for x in range(max(4,int(expected)-5),min(621,int(expected)+7)):
    g=(np.median(im[yi-3:yi+4,x+1:x+4])-np.median(im[yi-3:yi+4,x-3:x]))*(1 if i>=8 else -1)
    scores.append((g,x))
   vals=np.array([v[0] for v in scores]);k=int(vals.argmax());x=float(scores[k][1])
   if 0<k<len(vals)-1:
    den=vals[k-1]-2*vals[k]+vals[k+1]
    if den<-.01:x+=float(np.clip(.5*(vals[k-1]-vals[k+1])/den,-.5,.5))
   xs.append(x)
  xs=np.array(xs);xs=np.array([np.median(xs[max(0,k-2):k+3]) for k in range(len(xs))])
  t=(ys-A[1])/d[1];b0=(1-t)**3;b1=3*(1-t)**2*t;b2=3*(1-t)*t*t;b3=t**3
  inner=slice(3,-3)
  coef=np.linalg.lstsq(np.stack([b1[inner],b2[inner]],1),(xs-b0*A[0]-b3*B[0])[inner],rcond=None)[0]
  lo,hi=min(A[0],B[0])-1,max(A[0],B[0])+1
  c1[0],c2[0]=np.clip(coef,lo,hi)
  curved.append(i)
 controls.append(np.array([c1,c2]))
hin=np.array([controls[i-1][1] for i in range(len(a))]);hout=np.array([v[0] for v in controls])
def normal(v):
 v=v/np.linalg.norm(v,axis=-1,keepdims=True);return np.stack([-v[...,1],v[...,0]],axis=-1)
ni=normal(a-hin);no=normal(hout-a);new=a+4*(ni+no)/(1+(ni*no).sum(axis=1))[:,None]
t=np.linspace(0,1,301)[:,None];b0=(1-t)**3;b1=3*(1-t)**2*t;b2=3*(1-t)*t*t;b3=t**3
newcontrols=[]
for i in range(len(a)):
 j=(i+1)%len(a);b,c=controls[i];q=b0*a[i]+b1*b+b2*c+b3*a[j]
 deriv=3*(1-t)**2*(b-a[i])+6*(1-t)*t*(c-b)+3*t*t*(a[j]-c)
 target=q+4*normal(deriv)
 newcontrols.append(np.linalg.lstsq(np.hstack([b1[1:-1],b2[1:-1]]),(target-b0*new[i]-b3*new[j])[1:-1],rcond=None)[0])
d=json.loads(p.read_text());d['fullPolygon']=[dict(x=round(q[0]/626,8),y=round(q[1]/626,8),hIn=(newcontrols[i-1][1]/626).round(8).tolist(),hOut=(newcontrols[i][0]/626).round(8).tolist(),smooth=True,broken=True,corner=True) for i,q in enumerate(new)]
d['corners']=[(new[i]/626).round(8).tolist() for i in [0,1,7,8]]
d['inset_px']=4.0
d['edge_rule']='Retain five crease-related directional changes and protrusions along the paper sides. Simplify only tiny tears and speckle; do not smooth away real folds. Approximately 4px inward margin.'
d['edge_refined']='14 independent-handle anchors: four corners plus five fold-related anchors on each side; local RGB-transition fitting between anchors.'
s=json.dumps(d,indent=2)+'\n';tmp=p.with_suffix('.json.tmp');tmp.write_text(s);os.replace(tmp,p)
(root/'Outline Review/Sixth Image After.json').write_text(s)
print('Saved 14-point fold-aware outline; previous rejected shape backed up.')
