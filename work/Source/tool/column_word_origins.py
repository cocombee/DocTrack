"""Follow virtual font origins through their actual immutable word ink.

A native leading-edge midpoint can lie in blank stock before the first glyph.
Its document pixel may stretch without moving the text. Recover its original
word association from the column's registration frame, never from a candidate.
Ambiguous or incomplete associations retain the legacy material-point guard.
"""
import json
import numpy as np
from composite_mesh_sampling import MeshSampler
from composite_mesh_inverse import inverse_retained


def _key(record):
    return json.dumps({k:record[k] for k in ('normal_axis','points','reference')},
                      sort_keys=True,allow_nan=False)


def column_word_bindings(base, records, pixel_size, *, word_sources=None):
    scale=np.asarray(pixel_size,dtype=float)
    if scale.shape!=(2,) or not np.isfinite(scale).all() or (scale<=0).any():
        raise ValueError('Invalid column pixel basis')
    frames=[];node=base
    while node is not None:
        frames.append((node,{_key(r) for r in node.get('established_text_columns',[])}))
        node=node.get('preserved_mesh')
    cache={};output=[]
    for record in records:
        key=_key(record);registered=[frame for frame,keys in frames if key in keys]
        child=registered[-1] if registered else None
        capture=child.get('preserved_mesh') if child is not None else base
        registry=(child.get('retained_word_clouds',[]) if child is not None else
                  base.get('retained_word_clouds',[]) if word_sources is None else word_sources)
        reference=np.asarray(record['reference'],dtype=float)
        source=np.asarray(record['points'],dtype=float);axis=record['normal_axis']
        if (capture is None or axis not in (0,1) or source.shape!=reference.shape
                or source.ndim!=2 or source.shape[1]!=2 or len(source)<3
                or not np.isfinite(source).all() or not np.isfinite(reference).all()):
            output.append(None);continue
        identity=(id(capture),id(registry))
        if identity not in cache:
            sampler=MeshSampler(capture)
            clouds=[np.asarray(g,dtype=float) for g in registry]
            if any(g.ndim!=2 or g.shape[1]!=2 or not len(g) or not np.isfinite(g).all()
                   for g in clouds):
                output.append(None);continue
            current=(np.split(inverse_retained(sampler,np.vstack(clouds)),
                              np.cumsum([len(g) for g in clouds])[:-1]) if clouds else [])
            cache[identity]=(sampler,clouds,current)
        sampler,clouds,current=cache[identity]
        if abs((sampler.sample(reference)-source)*scale).max()>1e-6:
            output.append(None);continue
        selected=[]
        for origin in reference:
            candidates=[]
            for index,group in enumerate(current):
                low,high=group.min(0),group.max(0)
                font=(high[1-axis]-low[1-axis])*scale[1-axis]
                gap=(low[axis]-origin[axis])*scale[axis]
                tangent=(origin[1-axis]-np.array([low[1-axis],high[1-axis]]))*scale[1-axis]
                if (font>5 and -.25*font<=gap<=.5*font
                        and tangent[0]>=-.25*font and tangent[1]<=.25*font):
                    candidates.append(index)
            if len(candidates)!=1:break
            selected.append(candidates[0])
        if (len(selected)!=len(reference) or len(set(selected))!=len(selected)
                or any(len(clouds[i])<12 for i in selected)):
            output.append(None);continue
        output.append({'normal_axis':axis,'column_key':key,
                       'input_basis':'Original column registration frame and immutable word clouds',
                       'words':[{'source_points':clouds[i].tolist(),
                                 'reference_points':current[i].tolist(),
                                 'reference_origin':origin.tolist()}
                                for i,origin in zip(selected,reference)]})
    return output


def bound_column_audit(sampler, record, binding, pixel_size):
    """Keep the same .05px column limit and .1px rigid-word shape limits."""
    if binding['column_key']!=_key(record) or binding['normal_axis']!=record['normal_axis']:
        raise ValueError('Column binding belongs to different immutable evidence')
    scale=np.asarray(pixel_size,dtype=float);origins=[];fits=[];pairs=[];proper=True
    reference_origins=np.asarray(record['reference'],dtype=float)
    if (scale.shape!=(2,) or not np.isfinite(scale).all() or (scale<=0).any()
            or len(binding['words'])!=len(reference_origins)):
        raise ValueError('Invalid complete column binding or pixel basis')
    for word,expected_origin in zip(binding['words'],reference_origins):
        reference=np.asarray(word['reference_points'],dtype=float)*scale
        source=np.asarray(word['source_points'],dtype=float)
        origin=np.asarray(word['reference_origin'],dtype=float)
        if (source.ndim!=2 or source.shape[1]!=2 or reference.shape!=source.shape
                or not np.isfinite(source).all() or not np.isfinite(reference).all()
                or origin.shape!=(2,) or not np.array_equal(origin,expected_origin)):
            raise ValueError('Column binding changed its original paired evidence')
        actual=inverse_retained(sampler,source)*scale
        if reference.shape!=actual.shape or len(reference)<12 or not np.isfinite(actual).all():
            raise ValueError('Column word needs complete paired material points')
        a,z=reference-reference.mean(0),actual-actual.mean(0)
        u,_,v=np.linalg.svd(a.T@z);rotation=u@v
        proper &= bool(np.linalg.det(rotation)>0)
        fits.append(float(np.linalg.norm(a@rotation-z,axis=1).max()))
        ids=np.linspace(0,len(a)-1,min(96,len(a)),dtype=int)
        pairs.append(float(abs(np.linalg.norm(a[ids,None]-a[None,ids],axis=2)-
                               np.linalg.norm(z[ids,None]-z[None,ids],axis=2)).max()))
        origin=origin*scale
        origins.append((origin-reference.mean(0))@rotation+actual.mean(0))
    reference=np.asarray(record['reference'])*scale;origins=np.asarray(origins)
    if origins.shape!=reference.shape:raise ValueError('Column needs every original font origin')
    error=float(np.ptp((origins-reference)[:,record['normal_axis']]))
    return {'relative_alignment_change_px':error,
            'accepted':bool(proper and error<.05 and max(fits)<.1 and max(pairs)<.1),
            'basis':'Immutable word ink and virtual native font origins',
            'maximum_word_shape_error_px':max(fits),
            'maximum_word_pair_change_px':max(pairs)}
