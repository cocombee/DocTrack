"""Fit the reviewed receipt perimeter, then inset 2 source pixels.
The pixel edge is estimated from local RGB transitions. Low-contrast sections
are reviewed estimates; geometric offset precision is not segmentation certainty.
"""
from pathlib import Path
import json, math, numpy as np, os
ROOT=Path(__file__).resolve().parents[2]
label=ROOT/'Data/labels/train/bill_segmentation_image_project_000091.json'
# Source-pixel boundary landmarks, clockwise, including small inward bends.
top=np.array([(130,22),(175,24),(235,24),(275,23),(297,22),(303,19),(307,21),(311,23),(335,21),(350,23),(395,25),(420,26),(438,29),(465,30)],float)
right=np.array([(465,30),(468,65),(468,95),(466.5,130),(467,180),(466,240),(465,305),(465,340),(467,395),(468,445),(469,485),(469,535),(468,550),(467,560),(466.5,570),(464.5,582),(462,585)],float)
bottom=np.array([(106,596),(150,596),(210,596),(235,595),(255,594),(280,592),(289,592),(294,589),(299,592),(325,592),(355,592),(380,590),(398,589),(418,589),(437,588),(449,587),(462,585)],float)
left=np.array([(130,22),(128,30),(126,60),(124,90),(121,120),(120,150),(117.5,180),(119,210),(117,240),(115,270),(114.5,300),(114,330),(113,360),(111.5,390),(111.5,420),(110,450),(108,480),(107,510),(106,540),(106,570),(106,596)],float)

def pchip(x,y):
 h=np.diff(x);delta=np.diff(y)/h;d=np.zeros(len(x))
 d[0]=delta[0];d[-1]=delta[-1]
 for i in range(1,len(x)-1):
  if delta[i-1]*delta[i]>0:
   a=2*h[i]+h[i-1];b=h[i]+2*h[i-1]
   d[i]=(a+b)/(a/delta[i-1]+b/delta[i])
 result=[]
 for i in range(len(x)-1):
  n=max(2,int(math.ceil(h[i]*2)))
  t=np.linspace(0,1,n,endpoint=False)
  yy=(2*t**3-3*t**2+1)*y[i]+(t**3-2*t**2+t)*h[i]*d[i]+(-2*t**3+3*t**2)*y[i+1]+(t**3-t**2)*h[i]*d[i+1]
  result.extend(zip(x[i]+t*h[i],yy))
 result.append((x[-1],y[-1]));return np.array(result)
def side(a,axis):
 if axis==0:return pchip(a[:,0],a[:,1])
 return pchip(a[:,1],a[:,0])[:,::-1]
sides=[side(top,0),side(right,1),side(bottom,0)[::-1],side(left,1)[::-1]]
starts=np.cumsum([0]+[len(s)-1 for s in sides])
outer=np.concatenate([s[:-1] for s in sides])
v=outer-np.roll(outer,1,axis=0);v/=np.linalg.norm(v,axis=1)[:,None]
w=np.roll(outer,-1,axis=0)-outer;w/=np.linalg.norm(w,axis=1)[:,None]
n=np.stack([-v[:,1],v[:,0]],axis=1);m=np.stack([-w[:,1],w[:,0]],axis=1)
offset=outer+2*(n+m)/(1+(n*m).sum(axis=1))[:,None]

def curve(c,t):
 t=t[:,None];return (1-t)**3*c[0]+3*(1-t)**2*t*c[1]+3*(1-t)*t*t*c[2]+t**3*c[3]
segments=[];errors=[]
def fit(a):
 dist=np.linalg.norm(np.diff(a,axis=0),axis=1);t=np.r_[0,np.cumsum(dist)];t/=t[-1]
 start=(a[1]-a[0]);start/=np.linalg.norm(start)
 end=(a[-1]-a[-2]);end/=np.linalg.norm(end)
 b1=3*(1-t)**2*t;b2=3*(1-t)*t*t
 baseline=((1-t)**3+b1)[:,None]*a[0]+(t**3+b2)[:,None]*a[-1]
 design=np.stack([(b1[:,None]*start).reshape(-1),(-b2[:,None]*end).reshape(-1)],axis=1)
 lengths=np.linalg.lstsq(design,(a-baseline).reshape(-1),rcond=None)[0]
 lengths=np.maximum(lengths,0)
 c=np.array([a[0],a[0]+lengths[0]*start,a[-1]-lengths[1]*end,a[-1]])
 err=np.linalg.norm(curve(c,t)-a,axis=1);i=int(err.argmax())
 if err[i]>.35 and len(a)>6:
  i=max(2,min(len(a)-3,i));fit(a[:i+1]);fit(a[i:])
 else:segments.append(c);errors.append(float(err.max()))
closed=np.vstack([offset,offset[0]])
for i in range(4):fit(closed[starts[i]:starts[i+1]+1])
# Explicitly preserve true corners and independent incoming/outgoing curvature.
points=[]
for i,c in enumerate(segments):
 p=c[0];incoming=segments[i-1][2];outgoing=c[1]
 iscorner=any(np.linalg.norm(p-offset[int(k)%len(offset)])<1e-6 for k in starts[:-1])
 points.append(dict(x=round(p[0]/640,8),y=round(p[1]/640,8),hIn=(incoming/640).round(8).tolist(),hOut=(outgoing/640).round(8).tolist(),corner=bool(iscorner),smooth=True,broken=True))
d=json.loads(label.read_text());d['fullPolygon']=points
d['corners']=[(offset[int(k)]/640).round(8).tolist() for k in starts[:-1]]
d['inset_px']=2.0
d['edge_rule']='Entire reviewed paper perimeter, including small bends and indents; 2 source-pixel inward normal offset; adaptive cubic curves.'
d['edge_refined']='All four edges fitted to reviewed pixel-transition landmarks. Maximum curve-fit error to offset polyline: %.4f source pixels. Boundary estimates are uncertain where contrast is low.'%max(errors)
d['uncertain']=True
d['notes']='Reviewed all four edges. Low-contrast bottom-right boundary around source x=410..445 is estimated; exact physical edge cannot be proven from compressed image.'
text=json.dumps(d,indent=2)+'\n';tmp=label.with_suffix('.json.tmp');tmp.write_text(text);os.replace(tmp,label)
(ROOT/'Outline Review/Second Image After.json').write_text(text)
report={'source_size':[640,640],'inset_px':2,'anchors':len(points),'max_curve_fit_error_px':max(errors),'boundary_status':'Reviewed RGB-transition landmarks; bottom-right low-contrast segment remains estimated','profiles':{'top':top.tolist(),'right':right.tolist(),'bottom':bottom.tolist(),'left':left.tolist()}}
(ROOT/'Outline Review/Second Image Precision.json').write_text(json.dumps(report,indent=2)+'\n')
print({k:v for k,v in report.items() if k!='profiles'})
