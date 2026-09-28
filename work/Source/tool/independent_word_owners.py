"""Give each independently observed word its own rigid translation."""
import numpy as np


def word_groups(boxes, native_start, *, reading_axis=0):
    groups=[[i] for i in range(native_start,len(boxes))]
    unmatched=[]
    for i in range(native_start):
        lo,hi=boxes[i].min(0),boxes[i].max(0)
        area=np.prod(np.maximum(hi-lo,1e-12))
        scores=[]
        for j in range(native_start,len(boxes)):
            other_lo,other_hi=boxes[j].min(0),boxes[j].max(0)
            overlap=np.prod(np.maximum(np.minimum(hi,other_hi)-np.maximum(lo,other_lo),0))
            scores.append(overlap/area)
        if scores and max(scores)>=.5:
            groups[int(np.argmax(scores))].append(i)
        else:
            unmatched.append(i)
    # Preserve conservative component associations for ink without native
    # word evidence. Never merge two independently observed word owners.
    fallback=[]
    reading_boxes=boxes if reading_axis==0 else boxes[:,:,::-1]
    centers=reading_boxes.mean(1)
    for i in sorted(unmatched,key=lambda j:centers[j,1]):
        for group in fallback:
            heights=np.ptp(reading_boxes[group,:,1],axis=1)
            height=np.ptp(reading_boxes[i,:,1])
            if (reading_boxes[group,:,0].min()-.06<centers[i,0]<reading_boxes[group,:,0].max()+.06 and
                    abs(centers[group,1].mean()-centers[i,1])<.004 and
                    max(height,np.median(heights))<min(height,np.median(heights))*1.7):
                group.append(i)
                break
        else:
            fallback.append([i])
    return groups+fallback
