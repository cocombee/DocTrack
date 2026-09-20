from pathlib import Path
from PIL import Image, ImageDraw
base=Path(__file__).resolve().parents[1]
N=2048
im=Image.new('RGBA',(N,N))
d=ImageDraw.Draw(im)
def xy(points): return [(round(x*N/64),round(y*N/64)) for x,y in points]
def box(x,y,w,h):return (round(x*N/64),round(y*N/64),round((x+w)*N/64),round((y+h)*N/64))
d.rounded_rectangle(box(3,3,58,58),radius=round(13*N/64),fill='#17382d')
d.polygon(xy([(21,15),(35,15),(43,23),(43,49),(21,49)]),fill='#f4f8f5')
d.line(xy([(35,15),(35,23),(43,23)]),fill='#82b99d',width=round(1.7*N/64),joint='curve')
for a,b in [((26,31),(38,31)),((26,37),(35,37))]:d.line(xy([a,b]),fill='#688779',width=round(1.8*N/64))
d.rounded_rectangle(box(16,11,32,42),radius=round(4*N/64),outline='#62e6a6',width=round(1.7*N/64))
for x,y in [(16,15),(48,15),(16,49),(48,49)]:d.ellipse(box(x-2.5,y-2.5,5,5),fill='#62e6a6')
for n in [16,32,64,128,256,512,1024]:
 im.resize((n,n),Image.Resampling.LANCZOS).save(base/f'macos/Runner/Assets.xcassets/AppIcon.appiconset/app_icon_{n}.png')
im.resize((1024,1024),Image.Resampling.LANCZOS).save(base.parent.parent/'outputs/DocTrack Icon.png')
