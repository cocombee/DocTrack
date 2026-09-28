"""Exact-raster word evidence from DocTrack's local Apple Vision bridge."""
import json
import sys
import time
import uuid
from pathlib import Path

import cv2
import numpy as np


def validated_words(value):
    if not isinstance(value, list):
        return []
    words = []
    for item in value:
        if not isinstance(item, dict):
            continue
        confidence = item.get('confidence')
        text = item.get('text')
        if (not isinstance(text, str) or not text.strip() or
                not isinstance(confidence, (int, float)) or
                not np.isfinite(confidence) or not .25 < confidence <= 1):
            continue
        try:
            points = np.asarray(item.get('points'), dtype=float)
        except (ValueError, TypeError):
            continue
        if (points.shape != (4, 2) or not np.isfinite(points).all() or
                (points < 0).any() or (points > 1).any()):
            continue
        edges = np.roll(points, -1, axis=0) - points
        following = np.roll(edges, -1, axis=0)
        turns = edges[:, 0] * following[:, 1] - edges[:, 1] * following[:, 0]
        if not ((turns > 1e-12).all() or (turns < -1e-12).all()):
            continue
        word={'text':text,'confidence':float(confidence),'points':points.tolist()}
        extras=item.get('ink_envelopes',[])
        if isinstance(extras,list):
            envelopes=[];lo,hi=points.min(0),points.max(0);span=hi-lo
            for extra in extras[:4]:
                validated=validated_words([{'text':text,'confidence':confidence,'points':extra}])
                if not validated:continue
                envelope=np.asarray(validated[0]['points']);a,z=envelope.min(0),envelope.max(0)
                if np.all(a>=lo-span*.25) and np.all(z<=hi+span*.25):envelopes.append(envelope.tolist())
            if envelopes:word['ink_envelopes']=envelopes
        words.append(word)
    return words


def request_words(gray, folder, *, timeout=120, emit=None):
    """Ask the owning Dart process to recognize these exact PNG pixels."""
    folder = Path(folder)
    name = 'word-evidence-' + uuid.uuid4().hex
    raster, response = folder / (name + '.png'), folder / (name + '.json')
    try:
        if not cv2.imwrite(str(raster), gray):
            return []
        message = {'event': 'request_document_words', 'raster': raster.name,
                   'response': response.name}
        if emit is None:
            print(json.dumps(message), flush=True)
        else:
            emit(message)
        deadline = time.monotonic() + timeout
        while True:
            if response.is_file() and not response.is_symlink():
                return validated_words(json.loads(response.read_text()))
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return []
            time.sleep(min(.05, remaining))
    except (OSError, ValueError, TypeError, cv2.error):
        return []
    finally:
        for path in (raster, response):
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass


def ink_groups(gray, words, safe=None):
    return [record['ink'] for record in ink_observations(gray,words,safe)]


def ink_observations(gray, words, safe=None,*,localization_halo=1):
    """Word ink in the precise CV2 measurement-grid coordinate convention.

    Vision quads refer to pixel edges normalized by width/height. The mesh
    measurement uses pixel samples at index/(dimension-1), so captured ink
    must use that basis; a half-pixel convention from another render is invalid.
    """
    h, w = gray.shape
    if not isinstance(localization_halo,int) or not 1<=localization_halo<=5:
        raise ValueError('Native word localization halo must be one to five pixels')
    if min(h, w) < 2:
        return []
    contrast = cv2.GaussianBlur(gray.astype(np.float32), (0, 0),
                               max(2, 7 * max(h, w) / 1280)) - gray
    supported = contrast > 6
    if safe is not None:
        supported &= np.asarray(safe) > 250
    groups = []
    for word in validated_words(words):
        mask = np.zeros(gray.shape, np.uint8)
        for envelope in [word['points'],*word.get('ink_envelopes',[])]:
            quad=np.asarray(envelope)*[w,h]-.5
            cv2.fillConvexPoly(mask,np.rint(quad).astype(np.int32),1)
        # Include faint measured letter edges next to the native envelope.
        mask = cv2.dilate(mask, np.ones((2*localization_halo+1,)*2, np.uint8))
        y, x = np.where((mask > 0) & supported)
        if len(x) >= 12:
            groups.append({'ink':np.c_[x / (w - 1), y / (h - 1)],
                           'quad':(np.asarray(word['points'])*[w,h]-.5)/[w-1,h-1]})
    return groups


def measurement_words(words, width, height):
    """Convert Vision pixel edges to the CV2 sample-index mesh basis."""
    if min(width,height)<2:
        return []
    return [{**word,'points':((np.asarray(word['points'])*[width,height]-.5)/
                             [width-1,height-1]).tolist()}
            for word in validated_words(words)]


def recover_faint_components(gray,observations,words,safe=None,*,halo=3):
    """Add compact unowned glyph ink without expanding ownership over rules.

    Existing native points and quads remain unchanged. Only a uniquely nearby
    native observation can receive a measured component with real strong ink.
    Long strokes, ambiguous neighbors and already captured bodies are skipped.
    """
    h,w=gray.shape;scale=max(h,w)/1280
    expanded=ink_observations(gray,words,safe,localization_halo=halo)
    contrast=cv2.GaussianBlur(gray.astype(np.float32),(0,0),max(2,7*scale))-gray
    supported=contrast>6
    if safe is not None:supported &= np.asarray(safe)>250
    _,labels,stats,_=cv2.connectedComponentsWithStats(supported.astype(np.uint8))
    known=np.zeros_like(supported)
    result=[{**q,'ink':np.array(q['ink'],copy=True)} for q in observations]
    for q in result:
        p=np.rint(q['ink']*[w-1,h-1]).astype(int);known[p[:,1],p[:,0]]=True
    covered=np.bincount(labels[known],minlength=len(stats))
    candidates={}
    for q in expanded:
        matches=[i for i,b in enumerate(result) if np.array_equal(b['quad'],q['quad'])]
        if len(matches)!=1:continue
        p=np.rint(q['ink']*[w-1,h-1]).astype(int)
        ids,counts=np.unique(labels[p[:,1],p[:,0]],return_counts=True)
        for comp,count in zip(ids,counts):
            if comp and count>=.2*stats[comp,cv2.CC_STAT_AREA]:
                candidates.setdefault(int(comp),set()).add(matches[0])
    audit=[]
    for comp,owners in candidates.items():
        x,y,cw,ch,area=stats[comp]
        if (len(owners)!=1 or covered[comp]>=.8*area or
                not max(3,3*scale)<=ch<=38*scale or not 2<=cw<=min(2.4*ch,w*.04)
                or area<max(6,6*scale*scale)):
            continue
        yy,xx=np.where(labels[y:y+ch,x:x+cw]==comp);yy+=y;xx+=x
        if np.count_nonzero(contrast[yy,xx]>10)<3:continue
        owner=next(iter(owners));points=np.c_[xx/(w-1),yy/(h-1)]
        before=len(result[owner]['ink'])
        result[owner]['ink']=np.unique(np.vstack([result[owner]['ink'],points]),axis=0)
        audit.append({'word_index':owner,'component':comp,'points':points.tolist(),
                      'new_points':len(result[owner]['ink'])-before,
                      'previously_observed_points':int(covered[comp]),'component_points':int(area)})
    return result,{'components':audit,'existing_points_preserved':True}


def merge_crop_words(existing,crop_words,bounds,width,height):
    """Map exact crop pixel edges back and keep only independent full words."""
    x0,y0,x1,y1=bounds
    if not (0<=x0<x1<=width and 0<=y0<y1<=height):
        raise ValueError('Word crop must be inside the measurement raster')
    output=validated_words(existing)
    for word in validated_words(crop_words):
        pixels=np.asarray(word['points'])*[x1-x0,y1-y0]
        # Boundary tokens can be only part of a word. They cannot justify
        # rotating a glyph or redefining an existing word's ownership.
        if (pixels.min(0)<2).any() or (pixels.max(0)>[x1-x0-2,y1-y0-2]).any():continue
        points=(pixels+[x0,y0])/[width,height]
        lo,hi=points.min(0),points.max(0)
        overlaps=[]
        for index,before in enumerate(output):
            other=np.asarray(before['points']);a,z=other.min(0),other.max(0)
            overlap=np.maximum(0,np.minimum(hi,z)-np.maximum(lo,a)).prod()
            minimum_area=min((hi-lo).prod(),(z-a).prod())
            if minimum_area>0 and overlap/minimum_area>.5:
                overlaps.append(index)
        if not overlaps:
            output.append({**word,'points':points.tolist()})
        elif len(overlaps)==1:
            # Keep the established quad and owner. A larger agreeing crop
            # may supply faint ink edges missing from that same word, only
            # when it does not cross another independent word's envelope.
            index=overlaps[0];before=output[index];quad=np.asarray(before['points']);a,z=quad.min(0),quad.max(0);span=z-a
            extension=np.maximum(a-lo,0)+np.maximum(hi-z,0)
            crosses_other=False
            for other_index,other in enumerate(output):
                if other_index==index:continue
                other_points=np.asarray(other['points']);other_lo,other_hi=other_points.min(0),other_points.max(0)
                overlap=np.maximum(0,np.minimum(hi,other_hi)-np.maximum(lo,other_lo))*[width,height]
                if overlap.prod()>1:crosses_other=True;break
            if not crosses_other and np.all(lo>=a-span*.25) and np.all(hi<=z+span*.25) and np.max(extension*[width,height])>1:
                extras=before.get('ink_envelopes',[])
                if not any(np.array_equal(points,np.asarray(extra)) for extra in extras):
                    output[index]={**before,'ink_envelopes':[*extras,points.tolist()][:4]}
    return output


def request_words_with_crops(gray,folder,*,timeout=120,emit=None,provider=None):
    """Optional grid-document evidence from unmodified pixel crops.

    Full-page observations retain ownership. Crops add only missed words;
    they never replace working quads or add partially clipped tokens.
    """
    provider=request_words if provider is None else provider
    h,w=gray.shape
    words=provider(gray,folder,timeout=timeout,emit=emit)
    padding=max(8,round(48*max(h,w)/1280))
    for row in range(2):
        for col in range(2):
            x0=max(0,col*w//2-padding);x1=min(w,(col+1)*w//2+padding)
            y0=max(0,row*h//2-padding);y1=min(h,(row+1)*h//2+padding)
            crop=provider(gray[y0:y1,x0:x1],folder,timeout=timeout,emit=emit)
            words=merge_crop_words(words,crop,[x0,y0,x1,y1],w,h)
    return words
