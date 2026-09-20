from pathlib import Path
import json,numpy as np,os
root=Path(__file__).resolve().parents[2]
p=root/'Data/labels/train/bill_segmentation_image_project_000091.json'
# Restore the prior nine-point perimeter, then offset it 2px farther inward.
nodes=[(132,24,127,90,190,27,True),(306,24,248,26,358,22,False),(463,31,410,27,468,93,True),(465,304,465,212,465,396,False),(462,584,467,533,410,588,True),(294,591,350,591,234,595,False),(108,594,170,596,108,552,True),(112,446,109,497,115,395,False),(117,309,116,355,120,214,False)]
anchors=np.array([[n[0],n[1]] for n in nodes],float)
hin=np.array([[n[2],n[3]] for n in nodes],float)
hout=np.array([[n[4],n[5]] for n in nodes],float)
def normal(v):
 v=v/np.linalg.norm(v,axis=-1,keepdims=True)
 return np.stack([-v[...,1],v[...,0]],axis=-1)
ni=normal(anchors-hin);no=normal(hout-anchors)
new=anchors+2*(ni+no)/(1+(ni*no).sum(axis=1))[:,None]
controls=[];errors=[]
t=np.linspace(0,1,301)[:,None];b0=(1-t)**3;b1=3*(1-t)**2*t;b2=3*(1-t)*t*t;b3=t**3
for i in range(len(nodes)):
 j=(i+1)%len(nodes)
 a,b,c,d=anchors[i],hout[i],hin[j],anchors[j]
 q=b0*a+b1*b+b2*c+b3*d
 deriv=3*(1-t)**2*(b-a)+6*(1-t)*t*(c-b)+3*t*t*(d-c)
 target=q+2*normal(deriv)
 # Preserve corner intersections; fit only the interior samples.
 control=np.linalg.lstsq(np.hstack([b1[1:-1],b2[1:-1]]),
   (target-b0*new[i]-b3*new[j])[1:-1],rcond=None)[0]
 controls.append(control)
 fit=b0*new[i]+b1*control[0]+b2*control[1]+b3*new[j]
 errors.append(float(np.linalg.norm(fit[5:-5]-target[5:-5],axis=1).max()))
d=json.loads((root/'Outline Review/Second Image Before.json').read_text())
d['fullPolygon']=[dict(x=round(a[0]/640,8),y=round(a[1]/640,8),hIn=(controls[i-1][1]/640).round(8).tolist(),hOut=(controls[i][0]/640).round(8).tolist(),corner=bool(nodes[i][-1]),smooth=True,broken=True) for i,a in enumerate(new)]
d['corners']=[(new[i]/640).round(8).tolist() for i in [0,2,4,6]]
d['inset_px']=4.0
d['edge_rule']='Prior simple nine-point curved outline, offset an additional 2 source-image pixels inward at user request; approximately 4px total paper inset.'
d['edge_refined']='Nine anchors retained; gentle natural curvature preserved; extra inward offset applied to all sides.'
text=json.dumps(d,indent=2)+'\n';tmp=p.with_suffix('.json.tmp');tmp.write_text(text);os.replace(tmp,p)
(root/'Outline Review/Second Image After.json').write_text(text)
(root/'Outline Review/Second Image Precision.json').write_text(json.dumps({'method':'User-selected earlier nine-point outline with additional inward offset','anchors':9,'additional_inset_px':2,'estimated_total_paper_inset_px':4,'max_cubic_offset_fit_deviation_px':max(errors)},indent=2)+'\n')
print('Restored 9 points with additional 2px inset. Maximum curve fit deviation:',max(errors))
