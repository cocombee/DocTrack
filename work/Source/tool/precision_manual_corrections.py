from pathlib import Path
import json,numpy as np,ast
from PIL import Image,ImageDraw
r=Path(__file__).resolve().parents[2];out=r/'Outline Review/Precision Pass';m=json.loads((r/'Data/manifest.json').read_text())
s=ast.parse((r/'Source/tool/annotate_remaining.py').read_text());exec(compile(ast.Module(body=[v for v in s.body if isinstance(v,ast.FunctionDef) and v.name in ['unit','normal','rounded','offset']],type_ignores=[]),'<geometry>','exec'))
for n in [4,44]:
 e=m[n-1];name=Path(e['label']).name;p=out/'Candidates'/name;d=json.loads(p.read_text());im=Image.open(r/'Data'/e['image']).convert('RGB');wh=np.array(im.size)
 if n==4:
  a=np.array([(250,155),(823,152),(825,519),(247,528)],float);c=np.array([[(438,151),(638,155)],[(824,274),(827,399)],[(638,524),(438,527)],[(245,401),(249,279)]],float);a,c=offset(a,c)
 else:
  q=np.array([(391,205),(722,333),(587,508),(262,408)],float);a,c=rounded(q,0);a,c=offset(a,c);a,c=rounded(a,5)
 d['fullPolygon']=[dict(x=float(q[0]/wh[0]),y=float(q[1]/wh[1]),hIn=(c[i-1,1]/wh).tolist(),hOut=(c[i,0]/wh).tolist(),smooth=True,broken=True,corner=True) for i,q in enumerate(a)]
 d['edge_refined']='Physical paper edge manually distinguished from internal printing and surrounding transparent laminate; source-pixel inset applied.' if n==4 else 'Corrected visible lower paper boundary; occluded side corners inferred, with rounded corner curves.'
 p.write_text(json.dumps(d,indent=2)+'\n')
 dr=ImageDraw.Draw(im);pts=[]
 for i,A in enumerate(a):
  j=(i+1)%len(a);t=np.linspace(0,1,100,endpoint=False)[:,None];pts.extend((1-t)**3*A+3*(1-t)**2*t*c[i,0]+3*(1-t)*t*t*c[i,1]+t**3*a[j])
 dr.line([tuple(v) for v in pts]+[tuple(pts[0])],fill=(0,210,90),width=2);im.save(out/f'{n:02}.png')
rules=r.parent/'Document Outline Rules.md';s=rules.read_text();s+='\n- Image 4 correction: the black rectangular frame is printed inside the paper. Inset from the outer physical paper edge, never from the printed frame. Distinguish printing, paper, transparent laminate, and cast shadows before edge fitting; high contrast alone cannot identify a boundary.\n';rules.write_text(s)
