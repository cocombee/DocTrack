"""Measure large closed printed frames; never infer a frame from a crop edge."""
import cv2
import numpy as np

def printed_frame(gray):
    h,w=gray.shape
    binary=cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,31,12)
    contours,_=cv2.findContours(binary,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
    for c in sorted(contours,key=cv2.contourArea,reverse=True):
        area=cv2.contourArea(c);x,y,bw,bh=cv2.boundingRect(c)
        if not .55*w*h<area<.96*w*h:continue
        if min(bw/w,bh/h)<.7 or area/(bw*bh)<.9:continue
        if x<max(3,w*.005) or y<max(3,h*.005) or x+bw>w*.995 or y+bh>h*.995:continue
        filled=np.zeros_like(gray);cv2.drawContours(filled,[c],-1,255,-1)
        yy=np.linspace(y+bh*.035,y+bh*.965,65)
        xx=np.linspace(x+bw*.035,x+bw*.965,65)
        left=[];right=[];top=[];bottom=[]
        for v in yy:
            hits=np.flatnonzero(filled[round(v)]);left.append(hits[0]);right.append(hits[-1])
        for u in xx:
            hits=np.flatnonzero(filled[:,round(u)]);top.append(hits[0]);bottom.append(hits[-1])
        if np.ptp(np.array(right)-left)>bw*.12 or np.ptp(np.array(bottom)-top)>bh*.12:continue
        norm=np.array([w-1,h-1])
        return ([np.column_stack([xx,top])/norm,np.column_stack([xx,bottom])/norm],
                [np.column_stack([left,yy])/norm,np.column_stack([right,yy])/norm])
    return [],[]
