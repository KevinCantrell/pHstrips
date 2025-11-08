# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 16:12:20 2020

@author: David Campbell 
"""
import cv2
import numpy as np

font = cv2.FONT_HERSHEY_SIMPLEX
lineType = 1
antiAliasing = cv2.LINE_AA
# call cv2.putText with the correct arguments
#  cv2.putText(displayFrame,setting,(DisplayWidth-(parmWidth*5),parmHeight*(setRow+1)), font, fontScale,setColor,1,cv2.LINE_AA)
#   BlCoord = bottom left corner of origin
# 1700x900 0.3 fontScale
def OpenCVPutText(img, text, bLCoord, color, fontScale = 0.3):
    #fontScale = img.shape[0]/(1700/0.7)
    cv2.putText(img,text,bLCoord, font, fontScale,color,lineType,antiAliasing)

def OpenCVRebalanceImage(frame,rfactor,gfactor,bfactor):
    offset=np.zeros(frame[:,:,0].shape,dtype="uint8")
    frame[:,:,0]=cv2.scaleAdd(frame[:,:,0], bfactor, offset)
    frame[:,:,1]=cv2.scaleAdd(frame[:,:,1], gfactor, offset)
    frame[:,:,2]=cv2.scaleAdd(frame[:,:,2], rfactor, offset)
    return frame

def MidPoint(pt1,pt2):
    return ((pt1[0]+pt2[0])/2.0, (pt1[1]+pt2[1])/2.0)

def OpenCVDisplayedHistogram(image,channel,mask,NumBins,DataMin,DataMax,x,y,w,h,DisplayImage,color,integrationWindow,labelFlag,labelText=""):
    x=np.round(x,decimals=0).astype(int)
    y=np.round(y,decimals=0).astype(int)
    w=np.round(w,decimals=0).astype(int)
    h=np.round(h,decimals=0).astype(int)
    avgVal=cv2.meanStdDev(image,mask=mask)
    histdata = cv2.calcHist([image],[channel],mask,[NumBins],[DataMin,DataMax])
    #domValue=np.argmax(histdata)
    #domCount=np.max(histdata)/np.sum(histdata) 
    sortArg=np.argsort(histdata,axis=0)
    domValue=np.sum(histdata[sortArg[-5:][:,0]][:,0]*sortArg[-5:][:,0])/np.sum(histdata[sortArg[-5:][:,0]][:,0])
    domCount=np.sum(histdata[sortArg[-5:][:,0]][:,0])/np.sum(histdata)
    #numpixels=sum(np.array(histdata[domValue-integrationWindow:domValue+integrationWindow+1]))
    cv2.normalize(histdata, histdata, 0, h, cv2.NORM_MINMAX)
    if w>NumBins:
        binWidth = w/NumBins
    else:
        binWidth=1
    #img = np.zeros((h, NumBins*binWidth, 3), np.uint8)
    for i in range(NumBins):
        freq = int(histdata[i])
        cv2.rectangle(DisplayImage, ((i*binWidth)+x, y+h), (((i+1)*binWidth)+x, y+h-freq), color)
    if labelFlag:
        cv2.putText(DisplayImage,labelText+" m="+'{0:.2f}'.format(domValue/float(NumBins-1)*(DataMax-DataMin))+" p="+'{0:.2f}'.format(domCount)+" a="+'{0:.2f}'.format(avgVal[0][channel][0])+" s="+'{0:.2f}'.format(avgVal[1][channel][0]),(x,y+h+12), font, 0.4,color,1,cv2.LINE_AA)
    return (avgVal[0][channel][0],avgVal[1][channel][0],domValue/float(NumBins-1)*(DataMax-DataMin))
        
def OpenCVDisplayedScatter(img, xdata,ydata,x,y,w,h,color, circleThickness,ydataRangemin=None, ydataRangemax=None,xdataRangemin=None, xdataRangemax=None, alpha=1,labelFlag=True):      
    if xdataRangemin==None: 
         xdataRangemin=np.min(xdata)       
    if xdataRangemax==None: 
         xdataRangemax=np.max(xdata) 
    if ydataRangemin==None: 
         ydataRangemin=np.min(ydata) 
    if ydataRangemax==None: 
         ydataRangemax=np.max(ydata)
    xdataRange=xdataRangemax-xdataRangemin
    ydataRange=ydataRangemax-ydataRangemin
    if xdataRange!=0:
        xscale=float(w)/xdataRange
    else:
        xscale=1
    if ydataRange!=0:
        yscale=float(h)/ydataRange
    else:
        yscale=1
    
    #changed the code below to loop through the data and us opencv functions to draw the data points
    xdata=((xdata-xdataRangemin)*xscale).astype(np.int)
    xdata[xdata>w]=w
    xdata[xdata<0]=0
    ydata=((ydataRangemax-ydata)*yscale).astype(np.int)
    ydata[ydata>h]=h
    ydata[ydata<0]=0
    cv2.rectangle(img,(x,y),(x+w+1,y+h+1),color,1)
    for ptx, pty in zip(xdata, ydata):
        if xdata.any() > 0 and ydata.any() > 0:
            cv2.circle(img, (x + ptx,y + pty), circleThickness, color, -1)
    OpenCVPutText(img,str(round(xdataRangemax,0)),(x+w-15,y+h+15),color, fontScale = w / 700)
    OpenCVPutText(img,str(round(xdataRangemin,0)),(x-5,y+h+15),color, fontScale = w / 700)
    OpenCVPutText(img,str(round(ydataRangemax,0)),(x-40,y+10),color, fontScale = w / 700)
    OpenCVPutText(img,str(round(ydataRangemin,0)),(x-40,y+h-5),color, fontScale = w / 700)
        
def ShiftHOriginToValue(hue,maxHue,newOrigin,direction='cw'):
    shifthsv=np.copy(hue).astype('float')
    shiftAmount=maxHue-newOrigin
    shifthsv[hue<newOrigin]=shifthsv[hue<newOrigin]+shiftAmount
    shifthsv[hue>=newOrigin]=shifthsv[hue>=newOrigin]-newOrigin
    hue=shifthsv
    if direction=='ccw':
        hue=maxHue-hue
    return hue

def OpenCVRotateBound(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))