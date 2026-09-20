from pathlib import Path
import json,numpy as np,os
root=Path(__file__).resolve().parents[2]
p=root/'Data/labels/train/filtered_document_segmentation_000995.json'
# Restore the prior nine-point perimeter, then offset it 2px farther inward.
nodes=[(20,87,25,123,190,77,True),(531,57,360,67,540,79,True),(560,290,558,232,564,352,False),(573,470,569,410,575,497,False),(584,537,580,520,415,551,True),(79,580,247,566,79,526,True),(75,430,77,485,72,383,False),(64,310,68,345,58,289,False),(43,230,47,265,39,199,False)]
anchors=np.array([[n[0],n[1]] for n in nodes],float)
hin=np.array([[n[2],n[3]] for n in nodes],float)
hout=np.array([[n[4],n[5]] for n in nodes],float)
def normal(v):
 v=v/np.linalg.norm(v,axis=-1,keepdims=True)
 return np.stack([-v[...,1],v[...,0]],axis=-1)
ni=normal(anchors-hin);no=normal(hout-anchors)
new=anchors+4*(ni+no)/(1+(ni*no).sum(axis=1))[:,None]
controls=[];errors=[]
t=np.linspace(0,1,301)[:,None];b0=(1-t)**3;b1=3*(1-t)**2*t;b2=3*(1-t)*t*t;b3=t**3
for i in range(len(nodes)):
 j=(i+1)%len(nodes)
 a,b,c,d=anchors[i],hout[i],hin[j],anchors[j]
 q=b0*a+b1*b+b2*c+b3*d
 deriv=3*(1-t)**2*(b-a)+6*(1-t)*t*(c-b)+3*t*t*(d-c)
 target=q+4*normal(deriv)
 # Preserve corner intersections; fit only the interior samples.
 control=np.linalg.lstsq(np.hstack([b1[1:-1],b2[1:-1]]),
   (target-b0*new[i]-b3*new[j])[1:-1],rcond=None)[0]
 controls.append(control)
 fit=b0*new[i]+b1*control[0]+b2*control[1]+b3*new[j]
 errors.append(float(np.linalg.norm(fit[5:-5]-target[5:-5],axis=1).max()))
backup=root/'Outline Review/Sixth Image Before.json'
if not backup.exists(): backup.write_text(p.read_text())
d=json.loads(backup.read_text())
d['fullPolygon']=[dict(x=round(a[0]/626,8),y=round(a[1]/626,8),hIn=(controls[i-1][1]/626).round(8).tolist(),hOut=(controls[i][0]/626).round(8).tolist(),corner=bool(nodes[i][-1]),smooth=True,broken=True) for i,a in enumerate(new)]
d['corners']=[(new[i]/626).round(8).tolist() for i in [0,1,4,5]]
d['inset_px']=4.0
d['edge_rule']='Simple main paper perimeter with 4-source-pixel inward offset. Preserve broad folds and edge curvature; ignore speckle and tiny edge defects.'
d['edge_refined']='Nine anchors with straight top/bottom and curved side sections, reviewed against 626x626 source.'
text=json.dumps(d,indent=2)+'\n';tmp=p.with_suffix('.json.tmp');tmp.write_text(text);os.replace(tmp,p)
(root/'Outline Review/Sixth Image After.json').write_text(text)
(root/'Outline Review/Sixth Image Precision.json').write_text(json.dumps({'method':'Simplified nine-point outline for folded sheet','anchors':9,'inset_px':4,'estimated_total_paper_inset_px':4,'max_cubic_offset_fit_deviation_px':max(errors)},indent=2)+'\n')
print('Sixth image saved: 9 points, simplified folds, 4px inward offset.')
