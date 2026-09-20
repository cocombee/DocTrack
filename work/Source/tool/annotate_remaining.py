from pathlib import Path
import json, numpy as np, shutil, hashlib
from PIL import Image
r=Path(__file__).resolve().parents[2]; data=r/'Data'; review=r/'Outline Review/Remaining 47'; backup=review/'Before'; backup.mkdir(exist_ok=True)
m=json.loads((data/'manifest.json').read_text())
# Visually located boundaries in aspect-preserving preview coordinates, before source-pixel inset.
Q={
3:([(179,92),(456,92),(456,267),(178,268)],8),
4:([(139,86),(441,84),(444,281),(135,282)],4),
5:([(118,87),(539,126),(499,558),(79,550)],0),
7:([(23,269),(321,258),(342,453),(10,461)],12),
8:([(20,185),(210,169),(236,459),(52,483)],10),
9:([(41,273),(354,269),(356,489),(36,467)],10),
10:([(75,269),(294,309),(292,454),(53,438)],12),
11:([(26,241),(319,253),(335,456),(22,473)],[0,0,12,12]),
12:([(57,157),(326,204),(291,392),(24,344)],[0,0,10,10]),
13:([(27,288),(304,284),(321,480),(14,484)],[0,0,8,8]),
14:([(20,219),(309,205),(329,385),(22,402)],12),
15:([(51,289),(324,287),(327,453),(53,470)],11),
16:([(16,99),(345,99),(345,329),(14,328)],[0,0,10,10]),
19:([(64,126),(585,109),(596,505),(70,530)],16),
20:([(222,178),(445,179),(454,313),(220,315)],0),
21:([(66,99),(560,109),(551,447),(75,447)],0),
23:([(75,97),(566,98),(567,406),(74,407)],20),
24:([(27,8),(586,18),(574,368),(22,354)],22),
29:([(0,245),(350,202),(355,407),(27,454)],[0,10,10,10]),
30:([(14,323),(276,254),(322,414),(53,490)],8),
31:([(37,220),(332,220),(344,403),(32,410)],10),
32:([(59,205),(354,233),(332,434),(48,409)],[0,0,12,8]),
33:([(32,72),(587,39),(602,544),(47,578)],0),
34:([(44,166),(530,177),(527,447),(38,439)],0),
38:([(26,146),(325,143),(335,326),(29,338)],10),
40:([(212,118),(534,173),(437,530),(83,469)],0),
41:([(0,180),(135,196),(142,489),(0,500)],[0,8,8,0]),
42:([(237,100),(394,144),(316,298),(188,263)],2),
43:([(272,284),(425,343),(390,463),(230,421)],5),
44:([(311,165),(578,267),(481,396),(205,331)],5),
45:([(12,199),(343,191),(340,438),(15,423)],[0,0,12,10]),
46:([(55,179),(315,177),(320,334),(44,334)],9),
50:([(237,188),(432,202),(422,304),(230,295)],0),
}
# Curved/folded boundaries: explicit independent cubic controls, indexed by outgoing segment.
P={
17: ([(54,122),(389,60),(633,404),(242,543),(62,184),(51,138)],{0:[(143,132),(292,83)],4:[(55,164),(51,146)]}),
18: ([(107,75),(553,90),(494,611),(56,583)],{1:[(540,246),(508,441)],3:[(76,429),(91,209)]}),
22: ([(146,178),(484,184),(516,618),(116,628),(131,403)],{1:[(490,318),(508,493)],3:[(120,554),(126,475)],4:[(129,320),(137,243)]}),
25: ([(54,36),(573,34),(549,273),(558,332),(574,562),(66,570),(78,332)],{1:[(575,139),(568,212)],2:[(554,298),(557,321)],3:[(564,412),(568,491)],5:[(66,480),(70,394)],6:[(51,249),(50,141)]}),
26: ([(234,38),(539,294),(411,589),(158,378),(107,337)],{1:[(509,390),(447,511)],2:[(330,525),(239,443)],4:[(153,236),(197,125)]}),
27: ([(103,53),(564,221),(589,588),(54,498)],{1:[(591,312),(602,465)],3:[(67,349),(86,196)]}),
28: ([(75,259),(407,49),(509,347),(441,419),(212,583)],{1:[(445,135),(482,258)],2:[(490,372),(465,399)]}),
35: ([(178,121),(545,115),(533,328),(534,519),(455,522),(110,500),(127,244)],{1:[(540,178),(538,248)],2:[(531,383),(534,458)],3:[(511,522),(479,523)],4:[(350,516),(205,506)],5:[(126,407),(126,315)]}),
36: ([(43,65),(282,73),(321,83),(597,84),(597,499),(322,502),(311,508),(47,505),(31,497)],{0:[(130,67),(236,69)],1:[(303,74),(311,78)],4:[(487,497),(391,505)],6:[(218,506),(106,507)],7:[(36,505),(31,503)],8:[(31,349),(32,160)]}),
37: ([(48,118),(632,149),(639,223),(626,506),(606,533),(21,477),(10,353)],{0:[(231,139),(447,148)],2:[(637,329),(633,427)],4:[(440,516),(172,486)],6:[(26,268),(36,180)]}),
39: ([(126,52),(586,78),(505,349),(493,574),(319,573),(172,588),(52,584),(88,408)],{1:[(565,164),(513,278)],2:[(496,425),(503,502)],3:[(437,572),(377,569)],4:[(263,578),(218,588)],5:[(129,588),(93,586)],6:[(62,524),(82,451)],7:[(98,275),(113,133)]}),
47: ([(145,49),(237,113),(362,166),(442,165),(510,144),(612,181),(621,193),(600,498),(601,589),(582,582),(330,560),(33,522),(30,505),(72,371),(98,246),(145,93)],{0:[(173,75),(207,98)],1:[(280,134),(319,154)],2:[(390,165),(418,155)],3:[(466,162),(492,148)],4:[(550,148),(586,172)],5:[(618,184),(621,186)],6:[(617,295),(615,414)],7:[(596,530),(598,565)],9:[(499,572),(413,566)],10:[(222,547),(119,536)],12:[(41,461),(61,413)]}),
48: ([(93,0),(242,0),(247,171),(251,298),(255,465),(259,575),(264,639),(190,636),(105,613),(94,558),(94,342),(92,170)],{1:[(244,55),(246,112)],4:[(257,505),(259,541)],6:[(240,639),(210,639)],7:[(158,632),(133,620)],8:[(101,595),(96,580)],9:[(92,493),(94,409)]}),
49: ([(62,79),(267,74),(284,481),(61,486)],{1:[(273,211),(280,349)],3:[(59,350),(62,212)]}),
}
occluded={7,10,15,18,22,26,28,29,37,41,43,44,45};cropped={29,37,41,48};folds={17,25,26,28,35,36,37,39,47,48,49}
def unit(v):return v/np.linalg.norm(v,axis=-1,keepdims=True)
def normal(v):v=unit(v);return np.stack([-v[...,1],v[...,0]],axis=-1)
def rounded(q,rad):
 q=np.array(q,float); rad=[rad]*len(q) if isinstance(rad,(int,float)) else rad;a=[];segments=[]
 for i,v in enumerate(q):
  rr=rad[i]
  if not rr:a.append(v);segments.append(None);continue
  ui=unit(q[i-1]-v);uo=unit(q[(i+1)%len(q)]-v);A=v+rr*ui;B=v+rr*uo
  theta=np.arccos(np.clip(-np.dot(ui,uo),-1,1));k=4/3*np.tan(theta/4)/np.tan(theta/2)
  a.extend([A,B]);segments.extend([(A+k*(v-A),B+k*(v-B)),None])
 a=np.array(a);ctrl=[]
 for i,A in enumerate(a):
  B=a[(i+1)%len(a)];ctrl.append(segments[i] if segments[i] is not None else (A+(B-A)/3,A+2*(B-A)/3))
 return a,np.array(ctrl)
def offset(a,c,d=4):
 ni=normal(a-np.roll(c[:,1],1,axis=0));no=normal(c[:,0]-a)
 new=a+d*(ni+no)/np.maximum(.15,1+(ni*no).sum(1))[:,None]
 t=np.linspace(0,1,101)[:,None];b0=(1-t)**3;b1=3*(1-t)**2*t;b2=3*(1-t)*t*t;b3=t**3;out=[]
 for i,A in enumerate(a):
  j=(i+1)%len(a);B,C=c[i];D=a[j]
  target=b0*A+b1*B+b2*C+b3*D+d*normal(3*(1-t)**2*(B-A)+6*(1-t)*t*(C-B)+3*t*t*(D-C))
  out.append(np.linalg.lstsq(np.hstack([b1[1:-1],b2[1:-1]]),(target-b0*new[i]-b3*new[j])[1:-1],rcond=None)[0])
 return new,np.array(out)
assert set(Q)|set(P)==set(range(1,51))-{1,2,6}
protected={i:hashlib.sha256((data/m[i-1]['label']).read_bytes()).hexdigest() for i in [1,2,6]};report=[]
for n in sorted(set(Q)|set(P)):
 entry=m[n-1];path=data/entry['label'];dest=backup/path.name
 if not dest.exists():shutil.copy2(path,dest)
 d=json.loads(path.read_text());w,h=Image.open(data/entry['image']).size;pw,ph=Image.open(review/f'Previews/{n:02}.png').size;scale=np.array([w/pw,h/ph]);wh=np.array([w,h])
 if n in Q:
  q=np.array(Q[n][0],float);qa,qc=rounded(q*scale,0);qa,qc=offset(qa,qc)
  rad=Q[n][1];rad=[rad]*4 if isinstance(rad,(int,float)) else rad
  a,c=rounded(qa,[max(.5,v*scale.mean()-4) if v else 0 for v in rad])
 else:
  raw,curves=P[n];a=np.array(raw,float);c=np.array([[A+(a[(i+1)%len(a)]-A)/3,A+2*(a[(i+1)%len(a)]-A)/3] for i,A in enumerate(a)])
  for i,ctrl in curves.items():c[i]=ctrl
  # Auxiliary four corners are metadata; fullPolygon is authoritative.
  q=a[[np.argmin(a.sum(1)),np.argmax(a[:,0]-a[:,1]),np.argmax(a.sum(1)),np.argmin(a[:,0]-a[:,1])]]
 if n not in Q:a,c=offset(a*scale,c*scale)
 a=np.clip(a,[0,0],[w,h]);c=np.clip(c,[0,0],[w,h])
 d['fullPolygon']=[dict(x=round(A[0]/w,8),y=round(A[1]/h,8),corner=True,smooth=True,broken=True,hIn=(c[i-1,1]/wh).round(8).tolist(),hOut=(c[i,0]/wh).round(8).tolist()) for i,A in enumerate(a)]
 d['corners']=(q*scale/wh).round(8).tolist();d['inset_px']=4.0
 d['edge_rule']='Main physical document boundary; preserve real folds/outdents, infer behind occluders, simplify tiny tears; cubic rounded corners; approximately 4 source pixels inward.'
 d['edge_refined']='Visually placed sparse anchors with independent cubic handles; source-resolution geometric inset.'
 d['uncertain']=n in occluded|cropped;d['outline_review']={'status':'reviewed draft','hidden_edge_inferred':n in occluded,'image_clipped':n in cropped,'folds_preserved':n in folds,'rounded_corner_curves':n in Q and bool(np.any(Q[n][1]))}
 tmp=path.with_suffix('.json.tmp');tmp.write_text(json.dumps(d,indent=2)+'\n');tmp.replace(path)
 report.append(dict(index=n,name=entry['name'],points=len(a),**d['outline_review']))
assert all(hashlib.sha256((data/m[i-1]['label']).read_bytes()).hexdigest()==v for i,v in protected.items())
(review/'Batch Review.json').write_text(json.dumps(report,indent=2)+'\n')
print(f'Saved {len(report)} outlines; protected 1, 2, 6 unchanged; backups at {backup}.')
