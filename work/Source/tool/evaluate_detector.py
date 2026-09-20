from pathlib import Path
from PIL import Image,ImageDraw
import numpy as np,json,sys
r=Path(__file__).resolve().parents[2];version=sys.argv[1] if len(sys.argv)>1 else 'V1';pred=r/f'Scanner Integration/Predictions {version}';m=json.loads((r/'Data/manifest.json').read_text());reports=[];sheets=[]
for n,e in enumerate(m,1):
 f=pred/(e['name']+'.json')
 if not f.exists():continue
 d=json.loads(f.read_text());ref=json.loads((r/'Scanner Integration/Reference Baseline'/e['label']).read_text());p=ref['fullPolygon'];curve=[]
 for i,a in enumerate(p):
  b=p[(i+1)%len(p)];A=np.array([a['x'],a['y']]);B=np.array(a['hOut']) if a.get('smooth') else A;D=np.array([b['x'],b['y']]);C=np.array(b['hIn']) if b.get('smooth') else D;t=np.linspace(0,1,40,endpoint=False)[:,None];curve.extend((1-t)**3*A+3*(1-t)**2*t*B+3*(1-t)*t*t*C+t**3*D)
 q=np.array([[v['x'],v['y']] for v in d['outerBoundary']]);curve=np.array(curve);a=Image.new('1',(512,512));b=Image.new('1',(512,512));ImageDraw.Draw(a).polygon([tuple(v*511) for v in curve],fill=1)
 if len(q)>2:ImageDraw.Draw(b).polygon([tuple(v*511) for v in q],fill=1)
 aa=np.array(a);bb=np.array(b);iou=float((aa&bb).sum()/max(1,(aa|bb).sum()));report={'index':n,'name':e['name'],'split':e['split'],'iou':iou,'status':d['status'],'confidence':d['confidence'],'ms':d['elapsedMs'],'pass_iou_90':iou>=.90};reports.append(report)
 im=Image.open(r/'Data'/e['image']).convert('RGB');im.thumbnail((310,320));wh=np.array(im.size);draw=ImageDraw.Draw(im);draw.line([tuple(v*wh) for v in curve]+[tuple(curve[0]*wh)],fill=(40,255,70),width=2)
 if len(q)>2:draw.line([tuple(v*wh) for v in q]+[tuple(q[0]*wh)],fill=(0,150,255),width=2)
 sheets.append((n,im,iou))
for start in range(0,len(sheets),10):
 c=Image.new('RGB',(1600,720),(20,20,25));draw=ImageDraw.Draw(c)
 for k,(n,im,iou) in enumerate(sheets[start:start+10]):x=k%5*320;y=k//5*360;c.paste(im,(x+(320-im.width)//2,y+30));draw.text((x+7,y+5),f'{n} IoU {iou:.2f}',fill='white')
 c.save(pred/f'Review {start+1}.jpg')
summary={'count':len(reports),'mean_iou':float(np.mean([v['iou'] for v in reports])),'passes_90':sum(v['pass_iou_90'] for v in reports),'median_ms':float(np.median([v['ms'] for v in reports])),'test':{}}
for split in ['train','validation','test']:
 rows=[v for v in reports if v['split']==split]
 if rows:summary['test'][split]={'count':len(rows),'mean_iou':float(np.mean([v['iou'] for v in rows])),'passes_90':sum(v['pass_iou_90'] for v in rows)}
(pred/'Evaluation.json').write_text(json.dumps({'summary':summary,'images':reports},indent=2)+'\n');print(json.dumps(summary));print('Worst:',[(v['index'],round(v['iou'],2)) for v in sorted(reports,key=lambda v:v['iou'])[:12]])
