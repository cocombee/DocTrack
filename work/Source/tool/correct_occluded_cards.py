from pathlib import Path
import numpy as np,json,ast,shutil
from PIL import Image,ImageDraw
r=Path(__file__).resolve().parents[2];out=r/'Outline Review/Precision Pass';m=json.loads((r/'Data/manifest.json').read_text());s=ast.parse((r/'Source/tool/annotate_remaining.py').read_text());exec(compile(ast.Module(body=[v for v in s.body if isinstance(v,ast.FunctionDef) and v.name in ['unit','normal','rounded','offset']],type_ignores=[]),'<geometry>','exec'))
for n,q in {43:[(342.2,354.6),(533.2,432),(485.6,574.7),(277.4,516.6)],44:[(388,205),(724,333),(584,515),(238,396)]}.items():
 e=m[n-1];p=r/'Data'/e['label'];backup=out/f'Image {n} Rejected Inference.json'
 if not backup.exists():shutil.copy2(p,backup)
 d=json.loads(p.read_text());im=Image.open(r/'Data'/e['image']).convert('RGB');wh=np.array(im.size);a,c=rounded(q,0);a,c=offset(a,c);a,c=rounded(a,2)
 d['fullPolygon']=[dict(x=float(v[0]/wh[0]),y=float(v[1]/wh[1]),hIn=(c[i-1,1]/wh).tolist(),hOut=(c[i,0]/wh).tolist(),smooth=True,broken=True,corner=True) for i,v in enumerate(a)]
 d['corners']=(np.array(q)/wh).tolist();d['uncertain']=True;d['edge_refined']='Reconstructed concealed left corners by intersecting visible side-edge continuations with the bottom edge. Thumb boundaries do not constrain paper shape. Approximately 4px source inset with small true corner arcs.';d['outline_review']['status']='corrected occlusion inference; pending user review';d['outline_review']['hidden_edge_inferred']=True
 text=json.dumps(d,indent=2)+'\n';p.write_text(text);(out/'Candidates'/p.name).write_text(text)
 pts=[]
 for i,A in enumerate(a):
  j=(i+1)%len(a);t=np.linspace(0,1,100,endpoint=False)[:,None];pts.extend((1-t)**3*A+3*(1-t)**2*t*c[i,0]+3*(1-t)*t*t*c[i,1]+t**3*a[j])
 dr=ImageDraw.Draw(im);dr.line([tuple(v) for v in pts]+[tuple(pts[0])],fill=(0,210,90),width=2);im.save(out/f'{n:02}.png')
print('Saved corrected 43 and 44; image 45 unchanged.')
