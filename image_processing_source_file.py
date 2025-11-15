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

def OpenCVDisplayedHistogram(image,channel,mask,NumBins,DataMin,DataMax,x,y,w,h,DisplayImage,color,integrationWindow,labelFlag,labelText="",fontScale = 0.4):
    x=np.round(x,decimals=0).astype(int)
    y=np.round(y,decimals=0).astype(int)
    w=np.round(w,decimals=0).astype(int)
    h=np.round(h,decimals=0).astype(int)
    avgVal=cv2.meanStdDev(image,mask=mask)
    histdata = cv2.calcHist([image],[channel],mask,[NumBins],[DataMin,DataMax])
    domValue=np.argmax(histdata)
    pixelCount=np.sum(histdata) 
    # if pixelCount>0:
    #     domCount=np.max(histdata)/pixelCount
    # else:
    #     domCount=0
    #sortArg=np.argsort(histdata,axis=0)
    #domValue=np.sum(histdata[sortArg[-5:][:,0]][:,0]*sortArg[-5:][:,0])/np.sum(histdata[sortArg[-5:][:,0]][:,0])
    #domCount=np.sum(histdata[sortArg[-5:][:,0]][:,0])/np.sum(histdata)
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
        cv2.putText(DisplayImage,labelText+" m="+'{0:.2f}'.format(domValue/float(NumBins-1)*(DataMax-DataMin))+" n="+'{:4d}'.format(int(pixelCount))+" a="+'{0:.2f}'.format(avgVal[0][channel][0])+" s="+'{0:.2f}'.format(avgVal[1][channel][0]),(x,y+h+12), font, fontScale,color,1,cv2.LINE_AA)
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

def FindLargestContour(mask):
    if float(float(cv2.__version__[0])+float(cv2.__version__[2])/10)>=4:
        contours,hierarchy = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    else:
        image,contours,hierarchy = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)   
    if len(contours)>=1:
        maxArea=0
        contourIndex=0
        largestContourIndex=0
        for contour in contours:
            area=cv2.contourArea(contour)
            if area>maxArea:
                maxArea=area
                largestContourIndex=contourIndex
            contourIndex=contourIndex+1
        largestContour=contours[largestContourIndex]
        boundingRectangle=cv2.minAreaRect(largestContour)
        return(largestContour,maxArea,boundingRectangle)
    else:
        return(np.array([]),0,False)  

def FindContoursInside(mask,boundingContour,areaMin,areaMax,drawColor,frameForDrawing):
    ptsFound=np.zeros((40,4),dtype='float32')
    if float(float(cv2.__version__[0])+float(cv2.__version__[2])/10)>=4:
        contours,hierarchy = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    else:
        image,contours,hierarchy = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    circleIndex=0
    for contour in contours:
        area=cv2.contourArea(contour)
        if (area>=areaMin) & (area<=areaMax):
            M = cv2.moments(contour)
            if M['m00']>0:
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])
                dist = cv2.pointPolygonTest(boundingContour,(cx,cy),False)
                if dist!=-1:
                    ptsFound[circleIndex,0]=cx
                    ptsFound[circleIndex,1]=cy
                    ptsFound[circleIndex,2]=area
                    ptsFound[circleIndex,3]=1
                    circleIndex=circleIndex+1
                    cv2.drawContours(frameForDrawing,[contour],0,drawColor,5)
                    cv2.circle(frameForDrawing,(int(cx),int(cy)), 10, drawColor, -1)
    return(ptsFound[0:circleIndex,:])

def RegisterImageColorRectangleFlex(frame,frameForDrawing,boxLL,boxUL,boxC1,boxC2,boxC3,boxC4,boxOR,boxWH,epsilonWeight=0.1):
    if frame.size<=1:
        return(np.array([0]),frameForDrawing)
    hsvFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    boxMask = cv2.inRange(hsvFrame, np.array(boxLL), np.array(boxUL)) 
    outerBoxContour,boxArea,boxBoundingRectangle=FindLargestContour(boxMask)
    if outerBoxContour.size==0:
        return(np.array([0]),frameForDrawing)
    if outerBoxContour.size!=0:
        cv2.drawContours(frameForDrawing,[outerBoxContour],0,(255,0,255),10)
        epsilon = epsilonWeight*cv2.arcLength(outerBoxContour,True)
        approx = cv2.approxPolyDP(outerBoxContour,epsilon,True)
        approx=approx[:,0,:]
        if approx.shape[0]!=4:
            approx = cv2.boxPoints(boxBoundingRectangle)
        position=np.sum(approx,axis=1)
        order=np.argsort(position)
        approxSort=np.copy(approx)
        approx[0,:] = approxSort[order[0],:]
        approx[1,:] = approxSort[order[1],:]
        approx[2,:] = approxSort[order[2],:]
        approx[3,:] = approxSort[order[3],:]
        distances=np.zeros(4)
        distances[1]= np.linalg.norm(approx[0,:]-approx[1,:])
        distances[2]= np.linalg.norm(approx[0,:]-approx[2,:])
        distances[3]= np.linalg.norm(approx[0,:]-approx[3,:])
        order=np.argsort(distances)
        ptsFound=np.copy(approx)
        ptsFound[0,:] = approx[order[0],:]
        ptsFound[1,:] = approx[order[1],:]
        ptsFound[2,:] = approx[order[2],:]
        ptsFound[3,:] = approx[order[3],:]
        orientation=boxOR[0]
        #these can likely be switched to the settings cl1, etc
        ptsCard = np.float32([[boxC1[0],boxC1[1]],[boxC2[0],boxC2[1]],[boxC3[0],boxC3[1]],[boxC4[0],boxC4[1]]])
        ptsImage = np.float32([[135,220],[765,220],[135,1095],[765,1095]]) 
        if ptsFound.shape[0]==4:
            if orientation==1:
                ptsImage[0,0]=ptsFound[0,0]
                ptsImage[0,1]=ptsFound[0,1]
                ptsImage[1,0]=ptsFound[1,0]
                ptsImage[1,1]=ptsFound[1,1]
                ptsImage[2,0]=ptsFound[2,0]
                ptsImage[2,1]=ptsFound[2,1]
                ptsImage[3,0]=ptsFound[3,0]
                ptsImage[3,1]=ptsFound[3,1]
            else:
                ptsImage[0,0]=ptsFound[3,0]
                ptsImage[0,1]=ptsFound[3,1]
                ptsImage[1,0]=ptsFound[2,0]
                ptsImage[1,1]=ptsFound[2,1]
                ptsImage[2,0]=ptsFound[1,0]
                ptsImage[2,1]=ptsFound[1,1]
                ptsImage[3,0]=ptsFound[0,0]
                ptsImage[3,1]=ptsFound[0,1]
            Mrot = cv2.getPerspectiveTransform(ptsImage,ptsCard)
            rotImage = cv2.warpPerspective(frame,Mrot,(boxWH[0],boxWH[1]))
            return(rotImage,frameForDrawing)
        else:
            return(np.array([0]),frameForDrawing)
    else:
        return(np.array([0]),frameForDrawing)

def RegisterImageColorRectangle(frame,frameForDrawing,dictSet):
    hsvFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    boxMask = cv2.inRange(hsvFrame, np.array(dictSet['bcr ll']), np.array(dictSet['bcr ul'])) 
    outerBoxContour,boxArea,boxBoundingRectangle=FindLargestContour(boxMask)
    if outerBoxContour.size!=0:
        cv2.drawContours(frameForDrawing,[outerBoxContour],0,(255,0,255),40)
        epsilon = 0.01*cv2.arcLength(outerBoxContour,True)
        approx = cv2.approxPolyDP(outerBoxContour,epsilon,True)
        approx=approx[:,0,:]
        if approx.shape[0]!=4:
            approx = cv2.boxPoints(boxBoundingRectangle)
        position=np.sum(approx,axis=1)
        order=np.argsort(position)
        approxSort=np.copy(approx)
        approx[0,:] = approxSort[order[0],:]
        approx[1,:] = approxSort[order[1],:]
        approx[2,:] = approxSort[order[2],:]
        approx[3,:] = approxSort[order[3],:]
        distances=np.zeros(4)
        distances[1]= np.linalg.norm(approx[0,:]-approx[1,:])
        distances[2]= np.linalg.norm(approx[0,:]-approx[2,:])
        distances[3]= np.linalg.norm(approx[0,:]-approx[3,:])
        order=np.argsort(distances)
        ptsFound=np.copy(approx)
        ptsFound[0,:] = approx[order[0],:]
        ptsFound[1,:] = approx[order[1],:]
        ptsFound[2,:] = approx[order[2],:]
        ptsFound[3,:] = approx[order[3],:]
        orientation=dictSet['brt or'][0]
        #these can likely be switched to the settings cl1, etc
        ptsCard = np.float32([[dictSet['bl1 xy'][0],dictSet['bl1 xy'][1]],[dictSet['bl2 xy'][0],dictSet['bl2 xy'][1]],[dictSet['bl3 xy'][0],dictSet['bl3 xy'][1]],[dictSet['bl4 xy'][0],dictSet['bl4 xy'][1]]])
        ptsImage = np.float32([[135,220],[765,220],[135,1095],[765,1095]]) 
        if ptsFound.shape[0]==4:
            if orientation==1:
                ptsImage[0,0]=ptsFound[0,0]
                ptsImage[0,1]=ptsFound[0,1]
                ptsImage[1,0]=ptsFound[1,0]
                ptsImage[1,1]=ptsFound[1,1]
                ptsImage[2,0]=ptsFound[2,0]
                ptsImage[2,1]=ptsFound[2,1]
                ptsImage[3,0]=ptsFound[3,0]
                ptsImage[3,1]=ptsFound[3,1]
            elif orientation==2:
                ptsImage[0,0]=ptsFound[3,0]
                ptsImage[0,1]=ptsFound[3,1]
                ptsImage[1,0]=ptsFound[2,0]
                ptsImage[1,1]=ptsFound[2,1]
                ptsImage[2,0]=ptsFound[1,0]
                ptsImage[2,1]=ptsFound[1,1]
                ptsImage[3,0]=ptsFound[0,0]
                ptsImage[3,1]=ptsFound[0,1]
            elif orientation==3:
                ptsImage[0,0]=ptsFound[2,0]
                ptsImage[0,1]=ptsFound[2,1]
                ptsImage[1,0]=ptsFound[3,0]
                ptsImage[1,1]=ptsFound[3,1]
                ptsImage[2,0]=ptsFound[0,0]
                ptsImage[2,1]=ptsFound[0,1]
                ptsImage[3,0]=ptsFound[1,0]
                ptsImage[3,1]=ptsFound[1,1]
            elif orientation==4:
                ptsImage[0,0]=ptsFound[1,0]
                ptsImage[0,1]=ptsFound[1,1]
                ptsImage[1,0]=ptsFound[0,0]
                ptsImage[1,1]=ptsFound[0,1]
                ptsImage[2,0]=ptsFound[3,0]
                ptsImage[2,1]=ptsFound[3,1]
                ptsImage[3,0]=ptsFound[2,0]
                ptsImage[3,1]=ptsFound[2,1]
            Mrot = cv2.getPerspectiveTransform(ptsImage,ptsCard)
            rotImage = cv2.warpPerspective(frame,Mrot,(dictSet['box wh'][0],dictSet['box wh'][1]))
            #rotImage = cv2.warpPerspective(frame,Mrot,(dictSet['box wh'][1],dictSet['box wh'][0]))
            return(rotImage,frameForDrawing)
        else:
            return(np.array([0]),frameForDrawing)
    else:
        return(np.array([0]),frameForDrawing)
    
def RegisterImageColorCard(frame,frameForDrawing,dictSet):
    hsvFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    boxMask = cv2.inRange(hsvFrame, np.array(dictSet['box ll']), np.array(dictSet['box ul'])) 
    c12CircleMask = cv2.inRange(hsvFrame, np.array(dictSet['c12 ll']), np.array(dictSet['c12 ul'])) 
    c34CircleMask = cv2.inRange(hsvFrame, np.array(dictSet['c34 ll']), np.array(dictSet['c34 ul'])) 
    outerBoxContour,boxArea,boxBoundingRectangle=FindLargestContour(boxMask)
    if outerBoxContour.size!=0:
        cv2.drawContours(frameForDrawing,[outerBoxContour],0,(0,255,0),2)
        ptsC12 = FindContoursInside(c12CircleMask,outerBoxContour,boxArea*0.001,boxArea*0.25,(255,0,0),frameForDrawing)    
        ptsC34 = FindContoursInside(c34CircleMask,outerBoxContour,boxArea*0.001,boxArea*0.25,(0,0,255),frameForDrawing)    
        ptsFound = np.concatenate((ptsC12, ptsC34), axis=0) 
        ptsCard = np.float32([[dictSet['cl1 xy'][0],dictSet['cl1 xy'][1]],[dictSet['cl2 xy'][0],dictSet['cl2 xy'][1]],[dictSet['cl3 xy'][0],dictSet['cl3 xy'][1]],[dictSet['cl4 xy'][0],dictSet['cl4 xy'][1]]])
        ptsImage = np.float32([[135,220],[765,220],[135,1095],[765,1095]]) 
        if ptsFound.shape[0]==4:
            if (cv2.pointPolygonTest(outerBoxContour,ip.MidPoint(ptsFound[0,0:2],ptsFound[2,0:2]),False)==-1) & (cv2.pointPolygonTest(outerBoxContour,ip.MidPoint(ptsFound[0,0:2],ptsFound[3,0:2]),False)==-1):
                ptsImage[0,0]=ptsFound[0,0]
                ptsImage[0,1]=ptsFound[0,1]
                ptsImage[1,0]=ptsFound[1,0]
                ptsImage[1,1]=ptsFound[1,1]
            else:
                ptsImage[1,0]=ptsFound[0,0]
                ptsImage[1,1]=ptsFound[0,1]
                ptsImage[0,0]=ptsFound[1,0]
                ptsImage[0,1]=ptsFound[1,1]
            if (cv2.pointPolygonTest(outerBoxContour,ip.MidPoint(ptsImage[1,0:2],ptsFound[2,0:2]),False)==-1):
                ptsImage[2,0]=ptsFound[2,0]
                ptsImage[2,1]=ptsFound[2,1]
                ptsImage[3,0]=ptsFound[3,0]
                ptsImage[3,1]=ptsFound[3,1]
            else:
                ptsImage[3,0]=ptsFound[2,0]
                ptsImage[3,1]=ptsFound[2,1]
                ptsImage[2,0]=ptsFound[3,0]
                ptsImage[2,1]=ptsFound[3,1]
            Mrot = cv2.getPerspectiveTransform(ptsImage,ptsCard)
            #the last tulpe below needs to be in settings
            rotImage = cv2.warpPerspective(frame,Mrot,(dictSet['box wh'][0],dictSet['box wh'][1]))
            return(rotImage,frameForDrawing)
        else:
            return(np.array([0]),frameForDrawing)
    else:
        return(np.array([0]),frameForDrawing)

def WhiteBalanceFrame(displayFrame,rotImage,frame,frameForDrawing,dictSet,wbList=["WB1"]):
    rgbWBR=np.zeros((rotImage.shape),dtype='uint8')
    for wbRegion in wbList:
        rgbWBR[dictSet[wbRegion+' xy'][1]:dictSet[wbRegion+' xy'][1]+dictSet[wbRegion+' wh'][1], dictSet[wbRegion+' xy'][0]:dictSet[wbRegion+' xy'][0]+dictSet[wbRegion+' wh'][0]] = rotImage[dictSet[wbRegion+' xy'][1]:dictSet[wbRegion+' xy'][1]+dictSet[wbRegion+' wh'][1], dictSet[wbRegion+' xy'][0]:dictSet[wbRegion+' xy'][0]+dictSet[wbRegion+' wh'][0]]
        cv2.rectangle(frameForDrawing,(dictSet[wbRegion+' xy'][0],dictSet[wbRegion+' xy'][1]),(dictSet[wbRegion+' xy'][0]+dictSet[wbRegion+' wh'][0],dictSet[wbRegion+' xy'][1]+dictSet[wbRegion+' wh'][1]),(0,0,255),10 )
        if dictSet[wbRegion+' hs'][2]!=0:
            valSummary,stdSummary,resMask,resRGB,contourArea,boundingRectangle,histogramImage=SummarizeROI(rotImage,wbRegion,dictSet,connectedOnly=True,histogramHeight=dictSet['dsp wh'][1])
            displayFrame=OpenCVComposite(histogramImage, displayFrame, dictSet[wbRegion+' hs'])
        else:
            valSummary,stdSummary,resMask,resRGB,contourArea,boundingRectangle,histogramImage=SummarizeROI(rotImage,wbRegion,dictSet,connectedOnly=True)                
        if dictSet[wbRegion+' ds'][2]!=0:
            displayFrame=OpenCVComposite(resRGB, displayFrame, dictSet[wbRegion+' ds'])
    hsvWBR = cv2.cvtColor(rgbWBR, cv2.COLOR_BGR2HSV)
    maskWBR = cv2.inRange(hsvWBR, np.array(dictSet['WBR ll']), np.array(dictSet['WBR ul']))
    if np.sum(maskWBR)>0:
        RGBGreyWBR=cv2.mean(rgbWBR, mask=maskWBR)
        bscale=RGBGreyWBR[0]
        gscale=RGBGreyWBR[1]
        rscale=RGBGreyWBR[2]
        if dictSet['WBR sl'][0]!=0:
            scalemin=dictSet['WBR sl'][0]
        else:
            scalemin=min(rscale,gscale,bscale)
        if (scalemin!=0) & (min(rscale,gscale,bscale)!=0):
            rfactor=float(scalemin)/float(rscale)
            gfactor=float(scalemin)/float(gscale)
            bfactor=float(scalemin)/float(bscale)
        rgbWBR=ip.OpenCVRebalanceImage(rgbWBR,rfactor,gfactor,bfactor)
        rgbWBR = cv2.bitwise_and(rgbWBR,rgbWBR, mask= maskWBR)
        rotImage=ip.OpenCVRebalanceImage(rotImage,rfactor,gfactor,bfactor)
        frame=ip.OpenCVRebalanceImage(frame,rfactor,gfactor,bfactor)
    return(rgbWBR,rotImage,frame,frameForDrawing)

def ColorBalanceFrame(displayFrame,rotImage,frame,frameForDrawing,dictSet,refList=['RF1']):
    rgbCLR=np.zeros((rotImage.shape),dtype='uint8')
    referenceStats=np.zeros((16,len(refList)))    
    for refRegion,refNumber in zip(refList,range(len(refList))):
        rgbCLR[dictSet[refRegion+' xy'][1]:dictSet[refRegion+' xy'][1]+dictSet[refRegion+' wh'][1], dictSet[refRegion+' xy'][0]:dictSet[refRegion+' xy'][0]+dictSet[refRegion+' wh'][0]] = rotImage[dictSet[refRegion+' xy'][1]:dictSet[refRegion+' xy'][1]+dictSet[refRegion+' wh'][1], dictSet[refRegion+' xy'][0]:dictSet[refRegion+' xy'][0]+dictSet[refRegion+' wh'][0]]
        cv2.rectangle(frameForDrawing,(dictSet[refRegion+' xy'][0],dictSet[refRegion+' xy'][1]),(dictSet[refRegion+' xy'][0]+dictSet[refRegion+' wh'][0],dictSet[refRegion+' xy'][1]+dictSet[refRegion+' wh'][1]),(255,0,0),10 )
        if dictSet[refRegion+' hs'][2]!=0:
            valSummary,stdSummary,resMask,resRGB,contourArea,boundingRectangle,histogramImage=SummarizeROI(rotImage,refRegion,dictSet,connectedOnly=False,histogramHeight=dictSet['dsp wh'][1])
            displayFrame=OpenCVComposite(histogramImage, displayFrame, dictSet[refRegion+' hs'])
        else:
            valSummary,stdSummary,resMask,resRGB,contourArea,boundingRectangle,histogramImage=SummarizeROI(rotImage,refRegion,dictSet,connectedOnly=False)                
        if dictSet[refRegion+' ds'][2]!=0:
            displayFrame=OpenCVComposite(resRGB, displayFrame, dictSet[refRegion+' ds'])
            
        referenceStats[0:12,refNumber]=valSummary
        area=cv2.countNonZero(resMask)
        referenceStats[12,refNumber]=area
        if area>0:
            referenceStats[13,refNumber]=boundingRectangle[1][0]
            referenceStats[14,refNumber]=boundingRectangle[1][1]
            referenceStats[15,refNumber]=contourArea
    if (dictSet['flg hb'][0]==1):
        tableB=HistogramMatchTable(rgbCLR[:,:,0], templateHistogram[0])
        tableG=HistogramMatchTable(rgbCLR[:,:,1], templateHistogram[1])
        tableR=HistogramMatchTable(rgbCLR[:,:,2], templateHistogram[2])
        rotImage=ip.OpenCVHistogramBalanceImage(rotImage,tableR,tableG,tableB)
        frame=ip.OpenCVHistogramBalanceImage(frame,tableR,tableG,tableB)
    elif (dictSet['flg hb'][0]==3):
        refCLR=np.zeros((referenceFrame.shape),dtype='uint8')
        for refRegion,refNumber in zip(refList,range(len(refList))):
            refCLR[dictSet[refRegion+' xy'][1]:dictSet[refRegion+' xy'][1]+dictSet[refRegion+' wh'][1], dictSet[refRegion+' xy'][0]:dictSet[refRegion+' xy'][0]+dictSet[refRegion+' wh'][0]] = referenceFrame [dictSet[refRegion+' xy'][1]:dictSet[refRegion+' xy'][1]+dictSet[refRegion+' wh'][1], dictSet[refRegion+' xy'][0]:dictSet[refRegion+' xy'][0]+dictSet[refRegion+' wh'][0]]
        tgt_histB, _ = np.histogram(refCLR[:,:,0].ravel(), 256, [0,256])
        nonBlack=float(refCLR[:,:,0].size)-tgt_histB[0]
        tgt_histB[0] = 0
        templateHistogram[0] = np.cumsum(tgt_histB) / float(nonBlack)
        tgt_histG, _ = np.histogram(refCLR[:,:,1].ravel(), 256, [0,256])
        nonBlack=float(refCLR[:,:,1].size)-tgt_histG[0]
        tgt_histG[0] = 0
        templateHistogram[1] = np.cumsum(tgt_histG) / float(nonBlack)
        tgt_histR, _ = np.histogram(refCLR[:,:,2].ravel(), 256, [0,256])
        nonBlack=float(refCLR[:,:,2].size)-tgt_histR[0]
        tgt_histR[0] = 0
        templateHistogram[2] = np.cumsum(tgt_histR) / float(nonBlack)
        tableB=HistogramMatchTable(rgbCLR[:,:,0], templateHistogram[0])
        tableG=HistogramMatchTable(rgbCLR[:,:,1], templateHistogram[1])
        tableR=HistogramMatchTable(rgbCLR[:,:,2], templateHistogram[2])
        rotImage=ip.OpenCVHistogramBalanceImage(rotImage,tableR,tableG,tableB)
        frame=ip.OpenCVHistogramBalanceImage(frame,tableR,tableG,tableB)
    else:
        tableB=[]
        tableG=[]
        tableR=[]
    return(referenceStats,rgbCLR,tableB,tableG,tableR,rotImage,frame,frameForDrawing)

def OpenCVComposite(sourceImage, targetImage,settingsWHS):
    if (sourceImage.size==0) or (sourceImage.shape[1]==0) or (sourceImage.shape[0]==0):
        return targetImage
    if settingsWHS[2]!=100:
        scaleFactor=settingsWHS[2]/100
        if (int(sourceImage.shape[1]*scaleFactor)>0) and (int(sourceImage.shape[0]*scaleFactor)>0):
            imageScaled = cv2.resize(sourceImage, (int(sourceImage.shape[1]*scaleFactor),int(sourceImage.shape[0]*scaleFactor)), interpolation = cv2.INTER_AREA)
        else:
            imageScaled=sourceImage
    else:
        imageScaled=sourceImage
    xTargetStart=int(targetImage.shape[0]*settingsWHS[1]/100)
    xTargetEnd=int((targetImage.shape[0]*settingsWHS[1]/100)+imageScaled.shape[0])
    yTargetStart=int(targetImage.shape[1]*settingsWHS[0]/100)
    yTargetEnd=int((targetImage.shape[1]*settingsWHS[0]/100)+imageScaled.shape[1])
    if xTargetEnd>targetImage.shape[0]:
        xTargetEnd=targetImage.shape[0]
        xSourceEnd=targetImage.shape[0]-int(targetImage.shape[0]*settingsWHS[1]/100)
    else:
        xSourceEnd=imageScaled.shape[0]
    
    if yTargetEnd>targetImage.shape[1]:
        yTargetEnd=targetImage.shape[1]
        ySourceEnd=targetImage.shape[1]-int(targetImage.shape[1]*settingsWHS[0]/100)
    else:
        ySourceEnd=imageScaled.shape[1]
    if len(imageScaled.shape)==3:
        targetImage[xTargetStart:xTargetEnd,yTargetStart:yTargetEnd,:]=imageScaled[0:xSourceEnd,0:ySourceEnd,:]
    else:
        targetImage[xTargetStart:xTargetEnd,yTargetStart:yTargetEnd,0]=imageScaled[0:xSourceEnd,0:ySourceEnd]
        targetImage[xTargetStart:xTargetEnd,yTargetStart:yTargetEnd,1]=imageScaled[0:xSourceEnd,0:ySourceEnd]
        targetImage[xTargetStart:xTargetEnd,yTargetStart:yTargetEnd,2]=imageScaled[0:xSourceEnd,0:ySourceEnd]
    return targetImage

def DisplayAllSettings(dictSet,parmWidth,parmHeight,displayFrame,fontScale):
    setRow=0
    activeSettingsRow=dictSet['set rc'][0]
    activeSettingsColumn=dictSet['set rc'][1]
    if activeSettingsColumn>len(dictSet[sorted(dictSet)[activeSettingsRow]])-1:
        activeSettingsColumn=len(dictSet[sorted(dictSet)[activeSettingsRow]])-1
        dictSet['set rc'][1]=activeSettingsColumn
    for setRow,setting in zip(range(len(dictSet)),sorted(dictSet)): 
        if (activeSettingsRow==setRow):
            setColor=(0,0,255)
        else:
            setColor=(255,255,255)
        ip.OpenCVPutText(displayFrame, setting, (int(parmWidth*0.2),parmHeight*(setRow+1)), setColor, fontScale = fontScale)
        if activeSettingsColumn>len(dictSet[sorted(dictSet)[activeSettingsRow]])-1:
            activeSettingsColumn=len(dictSet[sorted(dictSet)[activeSettingsRow]])-1
        for setCol in range(len(dictSet[setting])):
            if (activeSettingsColumn==setCol) & (activeSettingsRow==setRow):
                setColor=(0,0,255)
            else:
                setColor=(255,255,255)
            ip.OpenCVPutText(displayFrame,str(dictSet[setting][setCol]),(parmWidth*(setCol+2),parmHeight*(setRow+1)),setColor, fontScale = fontScale)
    return displayFrame

def DisplaySomeSettings(dictSet,parmWidth,parmHeight,displayFrame,numRowsPad,fontScale):
    settings=sorted(dictSet)
    setRow=0
    activeSettingsRow=dictSet['set rc'][0]
    activeSettingsColumn=dictSet['set rc'][1]
    if activeSettingsRow-numRowsPad>=0:
        startRow=activeSettingsRow-numRowsPad
    else:
        startRow=0
    if activeSettingsRow+numRowsPad<=len(settings):
        endRow=activeSettingsRow+numRowsPad
    else:
        endRow=len(settings)    
    numRows=endRow-startRow
    if activeSettingsColumn>len(dictSet[settings[activeSettingsRow]])-1:
        activeSettingsColumn=len(dictSet[settings[activeSettingsRow]])-1
        dictSet['set rc'][1]=activeSettingsColumn
    for numRow,setRow,setting in zip(range(numRows),range(startRow,endRow),settings[startRow:endRow]): 
        if (activeSettingsRow==setRow):
            setColor=(0,0,255)
        else:
            setColor=(255,255,255)
        ip.OpenCVPutText(displayFrame, setting, (int(parmWidth*0.2),parmHeight*(numRow+1)), setColor, fontScale = fontScale)
        if activeSettingsColumn>len(dictSet[settings[activeSettingsRow]])-1:
            activeSettingsColumn=len(dictSet[settings[activeSettingsRow]])-1
        for setCol in range(len(dictSet[setting])):
            if (activeSettingsColumn==setCol) & (activeSettingsRow==setRow):
                setColor=(0,0,255)
            else:
                setColor=(255,255,255)
            ip.OpenCVPutText(displayFrame,str(dictSet[setting][setCol]),(parmWidth*(setCol+2),parmHeight*(numRow+1)),setColor, fontScale = fontScale)
    return displayFrame

def SummarizeROI(rotImage,roiSetName,dictSet,connectedOnly=True,histogramHeight=0):
    rgbROI = rotImage[dictSet[roiSetName+' xy'][1]:dictSet[roiSetName+' xy'][1]+dictSet[roiSetName+' wh'][1], dictSet[roiSetName+' xy'][0]:dictSet[roiSetName+' xy'][0]+dictSet[roiSetName+' wh'][0]]
    if rgbROI.size==0:
        #return(allROIsummary[0,:,0],allROIsummary[1,:,0],resMask,resFrameROI,contourArea,boundingRectangle,False)
        return(np.array([0,0,0,0,0,0,0,0,0,0,0,0]),np.array([0,0,0,0,0,0,0,0,0,0,0,0]),np.array([]),np.array([]),np.array([]),np.array([]),np.array([]))
    hsvROI = cv2.cvtColor(rgbROI, cv2.COLOR_BGR2HSV)
    hsvROI[:,:,0]=ip.ShiftHOriginToValue(hsvROI[:,:,0],dictSet['hue lo'][0],dictSet['hue lo'][1])
    labROI = cv2.cvtColor(rgbROI, cv2.COLOR_BGR2LAB)
    absROI=cv2.LUT(rgbROI, linLUTabs)*64
    if roiSetName[0:2]=="WB":
        maskROI = cv2.inRange(hsvROI, np.array(dictSet['WBR'+' ll']), np.array(dictSet['WBR'+' ul']))
    else:        
        maskROI = cv2.inRange(hsvROI, np.array(dictSet[roiSetName+' ll']), np.array(dictSet[roiSetName+' ul']))
    #following is only necessary if finding largest connected contour
    if connectedOnly:
        contourROI,contourArea,boundingRectangle=FindLargestContour(maskROI)
        contourMask=np.zeros((maskROI.shape),dtype='uint8')
        if contourArea>0:
            cv2.drawContours(contourMask,[contourROI],0,(255),-1)
        resMask = cv2.bitwise_and(maskROI,maskROI, mask= contourMask)
    else:
        resMask = maskROI
        boundingRectangle=((0, 0),(0,0),0)
        contourArea=0
    resFrameROI = cv2.bitwise_and(rgbROI,rgbROI, mask= resMask)
    rgbROIsummary=cv2.meanStdDev(rgbROI,mask=resMask)
    hsvROIsummary=cv2.meanStdDev(hsvROI,mask=resMask)
    labROIsummary=cv2.meanStdDev(labROI,mask=resMask)
    absROIsummary=cv2.meanStdDev(absROI,mask=resMask)
    allROIsummary=np.concatenate((rgbROIsummary,hsvROIsummary,labROIsummary,absROIsummary),axis=1)
    tempROIsummary=np.copy(allROIsummary)
    allROIsummary[:,0,:]=tempROIsummary[:,2,:]
    allROIsummary[:,2,:]=tempROIsummary[:,0,:]
    allROIsummary[:,9,:]=tempROIsummary[:,11,:]
    allROIsummary[:,11,:]=tempROIsummary[:,9,:]
    if histogramHeight!=0:
        inputImages= [rgbROI,rgbROI,rgbROI,hsvROI,hsvROI,hsvROI,labROI,labROI,labROI,absROI,absROI,absROI]
        rows=range(len(inputImages))
        displayColors=[(0,0,255),(0,255,0),(255,50,50),(255,255,0),(200,200,200),(128,128,128),(255,255,255),(255,0,255),(0,255,255),(0,0,255),(0,255,0),(255,50,50)]
        channels=[2,1,0,0,1,2,0,1,2,2,1,0]
        labels=["R","G","B","H","S","V","L","a","b","Ra","Ga","Ba"]
        singleHeight=int((histogramHeight-10)/len(inputImages))
        histogramFrame = np.zeros((histogramHeight, 255+20, 3), np.uint8)
        for row, displayColor, inputImage, channel, label in zip(rows, displayColors, inputImages, channels,labels):              
            mean,std,most=ip.OpenCVDisplayedHistogram(inputImage,channel,resMask,256,0,255,5,row*singleHeight+5,256,singleHeight-15,histogramFrame,displayColor,5,True,label,fontScale=0.38)
        return(allROIsummary[0,:,0],allROIsummary[1,:,0],resMask,resFrameROI,contourArea,boundingRectangle,histogramFrame)
    else:
        return(allROIsummary[0,:,0],allROIsummary[1,:,0],resMask,resFrameROI,contourArea,boundingRectangle,False)

def SummarizeFrame(frame,dictSet,histogramHeight=0):
    hsvROI = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    boxMask = cv2.inRange(hsvROI, np.array(dictSet['box ll']), np.array(dictSet['box ul']))
    boxFrameROI = cv2.bitwise_and(hsvROI,hsvROI, mask= boxMask)
    c12Mask = cv2.inRange(hsvROI, np.array(dictSet['c12 ll']), np.array(dictSet['c12 ul']))
    c12FrameROI = cv2.bitwise_and(hsvROI,hsvROI, mask= c12Mask)
    c34Mask = cv2.inRange(hsvROI, np.array(dictSet['c34 ll']), np.array(dictSet['c34 ul']))
    c34FrameROI = cv2.bitwise_and(hsvROI,hsvROI, mask= c34Mask)
    
    inputImages= [boxFrameROI,boxFrameROI,boxFrameROI,c12FrameROI,c12FrameROI,c12FrameROI,c34FrameROI,c34FrameROI,c34FrameROI]
    inputMasks= [boxMask,boxMask,boxMask,c12Mask,c12Mask,c12Mask,c34Mask,c34Mask,c34Mask]
    rows=range(len(inputImages))
    displayColors=[(255,255,0),(200,200,200),(128,128,128),(0,255,255),(200,200,200),(128,128,128),(255,0,255),(200,200,200),(128,128,128)]
    channels=[0,1,2,0,1,2,0,1,2]
    labels=["Hb","Sb","Vb","Hca","Sca","Vca","Hcb","Scb","Vcb"]
    singleHeight=int((histogramHeight-10)/len(inputImages))
    histogramFrame = np.zeros((histogramHeight, 255+20, 3), np.uint8)
    for row, displayColor, inputImage, resMask, channel, label in zip(rows, displayColors, inputImages, inputMasks, channels,labels):              
        mean,std,most=ip.OpenCVDisplayedHistogram(inputImage,channel,resMask,256,0,255,5,row*singleHeight+5,256,singleHeight-15,histogramFrame,displayColor,5,True,label,fontScale=0.38)
    return(histogramFrame,boxMask,c12Mask,c34Mask)
       
def ProcessOneFrame(frame,dictSet,displayFrame,wbList=["WB1"],roiList=["RO1"],refList=["RF1"],swatchList=[]):
    frameForDrawing=np.copy(frame)
    frameStats=np.zeros((16,2,len(roiList)))    
    swatchStats=np.zeros((16,2,len(swatchList)))    
    referenceColorStats=np.zeros((16,len(refList))) 
    if dictSet['flg rf'][0]==1:
        rotImage,frameForDrawing = RegisterImageColorCard(frame,frameForDrawing,dictSet)
        skipFrame=False
        if rotImage.size==1:
            skipFrame=True
            rotImage = np.copy(frame)
    elif dictSet['flg rf'][0]==2:
        rotImage,frameForDrawing = RegisterImageColorRectangle(frame,frameForDrawing,dictSet)
        #rotImage,frameForDrawing = RegisterImageColorRectangleFlex(frame,frameForDrawing,?????)
        #RegisterImageColorRectangleFlex(frame,frameForDrawing,boxLL,boxUL,boxC1,boxC2,boxC3,boxC4,boxOR,boxWH)
        
        skipFrame=False
        if rotImage.size==1:
            skipFrame=True
            rotImage = np.copy(frame)
    else:
        rotImage = np.copy(frame)
        skipFrame=False
    rotForDrawing=np.copy(rotImage)
    if skipFrame==False:
        if dictSet['flg wb'][1]==1:
            referenceColorStats,rgbCLR,tableB,tableG,tableR,rotImage,frame,rotForDrawing=ColorBalanceFrame(displayFrame,rotImage,frame,rotForDrawing,dictSet,refList=refList)
            if dictSet['flg di'][0]==1:
                cv2.imshow("CLR",rgbCLR)
        else:
            rgbCLR=[[0]]
        if dictSet['flg wb'][0]==1:
            rgbWBR,rotImage,frame,rotForDrawing=WhiteBalanceFrame(displayFrame,rotImage,frame,rotForDrawing,dictSet,wbList=wbList)
            if dictSet['flg di'][0]==1:
                cv2.imshow("WBR",rgbWBR)
            #if dictSet['WBR ds'][2]!=0:
            #    displayFrame=OpenCVComposite(rgbWBR, displayFrame, dictSet['WBR ds'])
            #hsvWBR = cv2.cvtColor(rgbWBR, cv2.COLOR_BGR2HSV)
            #maskWBR = cv2.inRange(hsvWBR, np.array(dictSet['WBR ll']), np.array(dictSet['WBR ul']))
            #rgbWBRsummary=cv2.meanStdDev(rgbWBR,mask=maskWBR)
            #resFrameWBR = cv2.bitwise_and(rgbWBR,rgbWBR, mask= maskWBR)
        if dictSet['flg di'][0]==1:
            cv2.imshow("RotatedImage",rotImage)
        for roiSetName,roiNumber in zip(roiList,range(len(roiList))):
            if dictSet[roiSetName+' hs'][2]!=0:
                valSummary,stdSummary,resMask,resRGB,contourArea,boundingRectangle,histogramImage=SummarizeROI(rotImage,roiSetName,dictSet,connectedOnly=dictSet[roiSetName+' ct'][0],histogramHeight=dictSet['dsp wh'][1])
                displayFrame=OpenCVComposite(histogramImage, displayFrame, dictSet[roiSetName+' hs'])
            else:
                valSummary,stdSummary,resMask,resRGB,contourArea,boundingRectangle,histogramImage=SummarizeROI(rotImage,roiSetName,dictSet,connectedOnly=dictSet[roiSetName+' ct'][0])
            cv2.rectangle(rotForDrawing,(dictSet[roiSetName+' xy'][0],dictSet[roiSetName+' xy'][1]),(dictSet[roiSetName+' xy'][0]+dictSet[roiSetName+' wh'][0],dictSet[roiSetName+' xy'][1]+dictSet[roiSetName+' wh'][1]),(0,255,0),10 )
            frameStats[0:12,0,roiNumber]=valSummary
            frameStats[0:12,1,roiNumber]=stdSummary
            area=cv2.countNonZero(resMask)
            frameStats[12,0,roiNumber]=area
            if area>0:
                frameStats[13,0,roiNumber]=boundingRectangle[1][0]
                frameStats[14,0,roiNumber]=boundingRectangle[1][1]
                frameStats[15,0,roiNumber]=contourArea
            if dictSet['flg di'][0]==1:
                cv2.imshow(roiSetName,resRGB)
            if dictSet[roiSetName+' ds'][2]!=0:
                displayFrame=OpenCVComposite(resRGB, displayFrame, dictSet[roiSetName+' ds'])
            if dictSet[roiSetName+' cs'][2]!=0:
                #box = cv2.boxPoints(boundingRectangle)
                if resMask.size>1:
                    x,y,w,h = cv2.boundingRect(resMask)
                #displayFrame=OpenCVComposite(resRGB[x:x+w,y:y+h,:], displayFrame, dictSet[roiSetName+' cs'])
                    displayFrame=OpenCVComposite(resRGB[y:y+h,x:x+w,:], displayFrame, dictSet[roiSetName+' cs'])
        for swatchName,swatchNumber in zip(swatchList,range(len(swatchList))):
            valSummary,stdSummary,resMask,resRGB,contourArea,boundingRectangle,histogramImage=SummarizeROI(rotImage,swatchName,dictSet,connectedOnly=1)
            cv2.rectangle(rotForDrawing,(dictSet[swatchName+' xy'][0],dictSet[swatchName+' xy'][1]),(dictSet[swatchName+' xy'][0]+dictSet[swatchName+' wh'][0],dictSet[swatchName+' xy'][1]+dictSet[swatchName+' wh'][1]),(0,0,255),10 )
            swatchStats[0:12,0,swatchNumber]=valSummary
            swatchStats[0:12,1,swatchNumber]=stdSummary
            area=cv2.countNonZero(resMask)
            swatchStats[12,0,swatchNumber]=area               
            swatchStats[13,0,swatchNumber]=dictSet[swatchName+' vl'][0]               
            swatchStats[14,0,swatchNumber]=dictSet[swatchName+' vl'][1]                
    else:
        rgbCLR=[[0]]
    return frameStats,referenceColorStats,swatchStats,displayFrame,frame,frameForDrawing,rotImag
