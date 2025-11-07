# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 16:10:14 2020

@author: Dabid
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#font = cv2.FONT_HERSHEY_SIMPLEX

def hyst(x, th_lo, th_hi, initial = False):
    # http://stackoverflow.com/questions/23289976/how-to-find-zero-crossings-with-hysteresis
    hi = x >= th_hi
    lo_or_hi = (x <= th_lo) | hi
    ind = np.nonzero(lo_or_hi)[0]
    if not ind.size: # prevent index error if ind is empty
        return np.zeros_like(x, dtype=bool) | initial
    cnt = np.cumsum(lo_or_hi) # from 0 to len(x)
    return np.where(cnt, hi[ind[cnt-1]], initial)
 
def crossBoolean(x, y, crossPoint=0, direction='cross'):
    """
    Given a Series returns all the index values where the data values equal 
    the 'cross' value. 
 
    Direction can be 'rising' (for rising edge), 'falling' (for only falling 
    edge), or 'cross' for both edges
    """
    # Find if values are above or bellow yvalue crossing:
    above=y > crossPoint
    below=np.logical_not(above)
    left_shifted_above = above[1:]
    left_shifted_below = below[1:]
    x_crossings = []
    # Find indexes on left side of crossing point
    if direction == 'rising':
        idxs = (left_shifted_above & below[0:-1]).nonzero()[0]
    elif direction == 'falling':
        idxs = (left_shifted_below & above[0:-1]).nonzero()[0]
    else:
        rising = left_shifted_above & below[0:-1]
        falling = left_shifted_below & above[0:-1]
        idxs = (rising | falling).nonzero()[0]
 
    # Calculate x crossings with interpolation using formula for a line:
    x1 = x[idxs]
    x2 = x[idxs+1]
    y1 = y[idxs]
    y2 = y[idxs+1]
    x_crossings = (crossPoint-y1)*(x2-x1)/(y2^y1) + x1
 
    return x_crossings,idxs

def PolyReg(X,Y,order):
    """
    Perform a least squares polynomial fit
    
    Parameters
    ----------
        X: a numpy array with shape M
            the independent variable 
        Y: a numpy array with shape M
            the dependent variable
        order: integer
            the degree of the fitting polynomial
    
    Returns
    -------
    a dict with the following keys:
        'coefs': a numpy array with length order+1 
            the coefficients of the fitting polynomial, higest order term first
        'errors': a numpy array with length order+1
            the standard errors of the calculated coefficients, 
            only returned if (M-order)>2
        'sy': float
            the standard error of the fit
        'n': integer
            number of data points (M)
        'poly':  class in numpy.lib.polynomial module
            a polynomial with coefficients (coefs) and degreee (order),
            see example below
        'res': a numpy array with length M
            the residuals of the fit
    
    Examples
    --------
    >>> x = np.array([0.0, 1.0, 2.0, 3.0,  4.0,  5.0])
    >>> y = np.array([0.0, 0.8, 0.9, 0.1, -0.8, -1.0])
    >>> fit = PolyReg(x, y, 2)
    >>> fit
    {'coefs': array([-0.16071429,  0.50071429,  0.22142857]),
     'errors': array([0.06882765, 0.35852091, 0.38115025]),
     'n': 6,
     'poly': poly1d([-0.16071429,  0.50071429,  0.22142857]),
     'res': array([-0.22142857,  0.23857143,  0.32      , -0.17714286, -0.45285714,
         0.29285714]),
     'sy': 0.4205438655564278}
    
    It is convenient to use the "poly" key for dealing with fit polynomials:
    
    >>> fit['poly'](0.5)
    0.43160714285714374
    >>> fit['poly'](10)
    -10.842857142857126
    >>> fit['poly'](np.linspace(0,10,11))
    array([  0.22142857,   0.56142857,   0.58      ,   0.27714286,
        -0.34714286,  -1.29285714,  -2.56      ,  -4.14857143,
        -6.05857143,  -8.29      , -10.84285714])
    """
    n=len(X)
    if X.shape!=Y.shape:
        raise Exception('The shape of X and Y should be the same')
    df=n-(order+1)
    if df<0:
        raise Exception('The number of data points is too small for that many coefficients')
    #if df = 0, 1, or 2 we call numpy's polyfit function without calculating the covariance matrix
    elif df<(3):
        coefs=np.polyfit(X,Y,order)
        p=np.poly1d(coefs)
        yFit=p(X)
        res=Y-yFit
        sy=np.sqrt( np.sum(res**2) / df )
        if order==1:
            #if the fit is linear we can explicitly calculate the standard errors of the slope and intercept
            #http://www.chem.utoronto.ca/coursenotes/analsci/stats/ErrRegr.html
            stdErrors=np.zeros((2))
            xVar=np.sum((X-np.mean(X))**2)
            sm=sy/np.sqrt(xVar)
            sb=np.sqrt(np.sum(X**2)/(n*xVar))*sy
            stdErrors[0]=sm
            stdErrors[1]=sb            
        else:
            stdErrors=np.full((order+1),np.inf)
    else:
        #The diagonal of the covariance matrix is the square of the standard error for each coefficent
        #NOTE 1: The polyfit function conservatively scales the covariance matrix. Dividing by (n-# coefs-2) rather than (n-# coefs)
        #NOTE 2: Because of this scaling factor, you can get division by zero in the covariance matrix when (# coefs-n)<2
        coefs,cov=np.polyfit(X,Y,order,cov=True)
        p=np.poly1d(coefs)
        yFit=p(X)
        res=Y-yFit
        sy=np.sqrt( np.sum(res**2) / df )
        stdErrors=np.sqrt(np.diagonal(cov)*(df-2)/df)
    return {'coefs':coefs,'errors':stdErrors,'sy':sy,'n':n,'poly':p,'res':res}

def FormatSciUsingError(x,e,withError=False,extraDigit=0):
    """
    Format the value, x, as a string using scientific notation and rounding appropriately based on the absolute error, e
    
    Parameters
    ----------
        x: number
            the value to be formatted 
        e: number
            the absolute error of the value
        withError: bool, optional
            When False (the default) returns a string with only the value. When True returns a string containing the value and the error
        extraDigit: int, optional
            number of extra digits to return in both value and error
    
    Returns
    -------
    a string
    
    Examples
    --------
    >>> FormatSciUsingError(3.141592653589793,0.02718281828459045)
    '3.14E+00'
    >>> FormatSciUsingError(3.141592653589793,0.002718281828459045)
    '3.142E+00'
    >>> FormatSciUsingError(3.141592653589793,0.002718281828459045,withError=True)
    '3.142E+00 (+/- 3E-03)'
    >>> FormatSciUsingError(3.141592653589793,0.002718281828459045,withError=True,extraDigit=1)
    '3.1416E+00 (+/- 2.7E-03)'
    >>> FormatSciUsingError(123456,123,withError=True)
    '1.235E+05 (+/- 1E+02)'
    """
    if abs(x)>=e:
        NonZeroErrorX=np.floor(np.log10(abs(e)))
        NonZeroX=np.floor(np.log10(abs(x)))
        formatCodeX="{0:."+str(int(NonZeroX-NonZeroErrorX+extraDigit))+"E}"
        formatCodeE="{0:."+str(extraDigit)+"E}"
    else:
        formatCodeX="{0:."+str(extraDigit)+"E}"
        formatCodeE="{0:."+str(extraDigit)+"E}"
    if withError==True:
        return formatCodeX.format(x)+" (+/- "+formatCodeE.format(e)+")"
    else:
        return formatCodeX.format(x)

def AnnotateFit(fit,axisHandle,annotationText='Eq',color='black',arrow=False,xArrow=0,yArrow=0,xText=0.5,yText=0.2,boxColor='0.9'):
    """
    Annotate a figure with information about a PolyReg() fit
    
    see https://matplotlib.org/api/_as_gen/matplotlition='cw'):
    shifthsv=np.copy(hue).astype('float')b.pyplot.annotate.html
    https://matplotlib.org/examples/pylab_examples/annotation_demo3.html
    
    Parameters
    ----------
        fit: dict, returned by the function PolyReg(X,Y,order)
            the fit to be summarized in the figure annotation 
        axisHandle: a matplotlib axes class
            the axis handle to the figure to be annotated
        annotationText: string, optional
            When "Eq" (the default) displays a formatted polynomial with the coefficients (rounded according to their error) in the fit. When "Box" displays a formatted box with the coefficients and their error terms.  When any other string displays a text box with that string.
        color: a valid color specification in matplotlib, optional
            The color of the box outline and connecting arrow.  Default is black. See https://matplotlib.org/users/colors.html
        arrow: bool, optional
            If True (default=False) draws a connecting arrow from the annotation to a point on the graph.
        xArrow: float, optional 
            The X coordinate of the arrow head using units of the figure's X-axis data. If unspecified or 0 (and arrow=True), defaults to the center of the X-axis.
        yArrow: float, optional 
            The Y coordinate of the arrow head using units of the figure's Y-axis data. If unspecified or 0 (and arrow=True), defaults to the calculated Y-value at the center of the X-axis.
        xText: float, optional 
            The X coordinate of the annotation text using the fraction of the X-axis (0=left,1=right). If unspecified, defults to the center of the X-axis.
        yText: float, optional 
            The Y coordinate of the annotation text using the fraction of the Y-axis (0=bottom,1=top). If unspecified, defults to 20% above the bottom.
    
    Returns
    -------
    a dragable matplotlib Annotation class
    
    Examples
    --------
    >>> annLinear=AnnotateFit(fitLinear,ax)
    >>> annLinear.remove()
    """
    c=fit['coefs']
    e=fit['errors']
    t=len(c)
    if annotationText=='Eq':
        annotationText="y = "
        for order in range(t):
            exponent=t-order-1
            if exponent>=2:
                annotationText=annotationText+FormatSciUsingError(c[order],e[order])+"x$^{}$".format(exponent)+" + "
            elif exponent==1:
                annotationText=annotationText+FormatSciUsingError(c[order],e[order])+"x + "
            else:
                annotationText=annotationText+FormatSciUsingError(c[order],e[order])
        annotationText=annotationText+", sy={0:.1E}".format(fit['sy'])
    elif annotationText=='Box':
        annotationText="Fit Details:\n"
        for order in range(t):
            exponent=t-order-1
#            annotationText=annotationText+"C$_{x^{"+str(exponent)+"}}$ = "+FormatSciUsingError(c[order],e[order],extraDigit=1)+' $ \pm $ '+"{0:.1E}".format(e[order])+'\n'
        annotationText=annotationText+'n = {0:d}'.format(fit['n'])+', DoF = {0:d}'.format(fit['n']-t)+", s$_y$ = {0:.1E}".format(fit['sy'])
    if (arrow==True):
        if (xArrow==0):
            xSpan=axisHandle.get_xlim()
            xArrow=np.mean(xSpan)
        if (yArrow==0):    
            yArrow=fit['poly'](xArrow)
        annotationObject=axisHandle.annotate(annotationText, 
                xy=(xArrow, yArrow), xycoords='data',
                xytext=(xText, yText),  textcoords='axes fraction',
                arrowprops={'color': color, 'width':1, 'headwidth':5},
                bbox={'boxstyle':'round', 'edgecolor':color,'facecolor':boxColor}
                )
    else:
        xSpan=axisHandle.get_xlim()
        xArrow=np.mean(xSpan)
        ySpan=axisHandle.get_ylim()
        yArrow=np.mean(ySpan)
        annotationObject=axisHandle.annotate(annotationText, 
                xy=(xArrow, yArrow), xycoords='data',
                xytext=(xText, yText),  textcoords='axes fraction',
                ha="left", va="center",
                bbox={'boxstyle':'round', 'edgecolor':color,'facecolor':boxColor}
                )
    annotationObject.draggable()
    return annotationObject

def ProcessI2(sgList,parameterStats,dictSet,frameNumber,file_path):
    labels=["R","G","B","H","S","V","L","a","b","Ra","Ga","Ba","Ga-Ra","Ba-Ra","Ga-Ba"]
    #dfMean=pd.DataFrame(data=parameterStats[0:12,0,0:frameNumber,1].transpose(),columns=["R","G","B","H","S","V","L*","a*","b*","Ra","Ga","Ba"],index=parameterStats[31,0,0:frameNumber,1])
    #dfStdev=pd.DataFrame(data=parameterStats[0:12,1,0:frameNumber,1].transpose(),columns=["R","G","B","H","S","V","L*","a*","b*","Ra","Ga","Ba"],index=parameterStats[31,0,0:frameNumber,1])
    #dfMost=pd.DataFrame(data=parameterStats[0:12,2,0:frameNumber,1].transpose(),columns=["R","G","B","H","S","V","L*","a*","b*","Ra","Ga","Ba"],index=parameterStats[31,0,0:frameNumber,1])

    writer = pd.ExcelWriter(file_path, engine='xlsxwriter')
    workbook  = writer.book
    
    for signal,index in zip(sgList,range(len(sgList))):
        minSignal=dictSet[signal+' lm'][0]
        maxSignal=dictSet[signal+' lm'][1]
        minArea=dictSet[signal+' lm'][2]
        dfMinArea=parameterStats[15,0,0:frameNumber,0]>minArea
        dfBool=(dfMinArea) & (parameterStats[20+index,0,0:frameNumber,0]<=maxSignal) & (parameterStats[20+index,0,0:frameNumber,0]>=minSignal)
        worksheetFit = workbook.add_worksheet(signal+"_Fit")
        worksheetFit.write('A1', 'Time')
        worksheetFit.write('B1', labels[dictSet[signal+" c1"][0]]+","+labels[dictSet[signal+" c2"][0]])
        worksheetFit.write('C1', 'Time (linear range B-R)')
        worksheetFit.write('D1', labels[dictSet[signal+" c1"][0]]+","+labels[dictSet[signal+" c2"][0]] + ' range ' +str(dictSet[signal+" lm"][0])+" to "+str(dictSet[signal+" lm"][1]))
        worksheetFit.write_column('A2',parameterStats[30,0,0:frameNumber,0])
        worksheetFit.write_column('B2',parameterStats[20+index,0,0:frameNumber,0])
        worksheetFit.write_column('C2',parameterStats[30,0,0:frameNumber,0][dfBool])
        worksheetFit.write_column('D2',parameterStats[20+index,0,0:frameNumber,0][dfBool])
        numEntries=parameterStats[20+index,0,0:frameNumber,0][dfBool].size
        numIndex=str(numEntries+1)
        worksheetFit.write_array_formula('J3:K5', '{=LINEST(D2:D'+numIndex+',C2:C'+numIndex+',TRUE,TRUE)}')
        worksheetFit.write('J2', 'Slope')
        worksheetFit.write('K2', 'Intercept')
        worksheetFit.write('I3', 'coefs')
        worksheetFit.write('I4', 'errors')
        worksheetFit.write('I5', 'r2, sy')
        
        chart1 = workbook.add_chart({'type': 'scatter'})
        numAllEntries=parameterStats[20+index,0,0:frameNumber,0].size
        chart1.add_series({
            'name': labels[dictSet[signal+" c1"][0]]+","+labels[dictSet[signal+" c2"][0]]+' linear',
            'categories': [signal+"_Fit", 1, 2, 1+numEntries-1, 2],
            'values': [signal+"_Fit", 1, 3, 1+numEntries-1, 3],
            'trendline': {
                'type': 'linear',
                'display_equation': True,
                'line': {
                'color': 'black',
                'width': 2,
                },
                'forward': parameterStats[20+index,0,frameNumber-1,0],
                'backward': parameterStats[20+index,0,0,0],
            },
            'marker': {
                'type': 'circle',
                'size': 8,
                'fill':   {'color': '#a66fb5'},
            },
        })
        chart1.add_series({
            'name': labels[dictSet[signal+" c1"][0]]+","+labels[dictSet[signal+" c2"][0]]+' all',
            'categories': [signal+"_Fit", 1, 0, 1+numAllEntries-1, 0],
            'values': [signal+"_Fit", 1, 1, 1+numAllEntries-1, 1],
            'marker': {
                    'type': 'circle',
                    'size': 4,
                    'fill':   {'color': '#490648'},
            },
        })
    
        #chart1.set_title ({'name': labels[dictSet['ya1 ch'][0]]+' Change'})
        if (parameterStats[30,0,0:frameNumber,0].size!=0) and (parameterStats[20+index,0,0:frameNumber,0].size!=0):
            chart1.set_x_axis({
                    'name': 'Time (seconds)',
                    'min': np.min(np.floor(parameterStats[30,0,0:frameNumber,0])),
                    'max': np.max(np.ceil(parameterStats[30,0,0:frameNumber,0]))
                    })
            chart1.set_y_axis({
                    'name': 'Signal',
                    'min': np.min(np.floor(parameterStats[20+index,0,0:frameNumber,0])),
                    'max': np.max(np.ceil(parameterStats[20+index,0,0:frameNumber,0])),
                    'major_gridlines': {
                            'visible': False,
                            },
                    })
            #chart1.set_style(6)
            chart1.set_legend({'position': 'none'})
            worksheetFit.insert_chart('I8', chart1, {'x_offset': 25, 'y_offset': 10})  
    workbook.close()
