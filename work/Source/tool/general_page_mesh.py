"""Diagnostic middle stage: all supported page guides fit one shared mesh.

Runs after the caller's angle projection and before any local repair. Native
words only associate real glyph measurements; no word is corrected separately.
Original material registrations are read-only. No app routing is changed.
"""
import numpy as np
import importlib.util
from pathlib import Path

# Earlier experimental receipt modules deliberately prepend a frozen ruler
# directory. Bind this stage to its sibling solver without replacing theirs.
_spec=importlib.util.spec_from_file_location('_general_page_ruler',
    Path(__file__).with_name('mesh_ruler.py'))
_ruler=importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ruler)
ruler_mesh=_ruler.ruler_mesh
from physical_text_guides import associated_rows,measured_text_rows


def measured_guides(gray, safe, ledger, words):
    h,w=gray.shape;basis=np.array([w-1,h-1])
    horizontal=[];vertical=[];evidence=[]
    for index,stroke in enumerate(ledger['complete_printed_strokes']):
        points=np.asarray(stroke['source_points'],float)
        axis=int(stroke['axis'])
        guide=(points,4.)
        (horizontal if axis==0 else vertical).append(guide)
        evidence.append(dict(kind='actual_printed_stroke',axis=1-axis,
            owner=index,points=points.tolist(),confidence=4.))
    for measured in measured_text_rows(gray,safe,associated_rows(words,w,h)):
        points=measured['points']
        confidence=.4 if measured['compact_word_guide'] else 1.5
        horizontal.append((points,confidence))
        evidence.append(dict(kind='supported_glyph_baseline_model',axis=1,
            points=points.tolist(),confidence=confidence,
            measurement={k:v for k,v in measured.items() if k!='points'},
            model_is_not_registered_ink=True))
    return horizontal,vertical,evidence


def solve_general(horizontal,vertical,*,n=33):
    grid=ruler_mesh(horizontal,vertical,n=n,fixed_boundary=True)
    du,dv=np.diff(grid,axis=1),np.diff(grid,axis=0)
    determinants=[(a[...,0]*b[...,1]-a[...,1]*b[...,0])*(n-1)**2
        for a in (du[:-1],du[1:]) for b in (dv[:,:-1],dv[:,1:])]
    minimum=float(min(q.min() for q in determinants))
    u,v=np.meshgrid(np.linspace(0,1,n),np.linspace(0,1,n))
    identity=np.stack([u,v],-1)
    edges=np.r_[grid[0]-identity[0],grid[-1]-identity[-1],
                grid[:,0]-identity[:,0],grid[:,-1]-identity[:,-1]]
    certificate=dict(minimum_continuous_bilinear_jacobian=minimum,
        boundary_maximum_displacement=float(abs(edges).max()),
        range=[float(grid.min()),float(grid.max())],
        accepted=bool(np.isfinite(grid).all() and minimum>.2 and
                      abs(edges).max()==0 and grid.min()>=0 and grid.max()<=1))
    return grid,certificate
