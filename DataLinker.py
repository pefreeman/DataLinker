#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 14:53:25 2018

@author: peterfreeman (based on R code by Jaehyeok Shin)

Note that all comments are cut-and-pasted directly from analogous
portions of Jaehyeok's R code base.
"""

import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import eigsh
from scipy.spatial.distance import pdist, squareform, directed_hausdorff
from sklearn.neighbors import NearestNeighbors

class DataLinker(object):
    
    def __init__(self,
                 probSeq1=np.arange(0.05,0.975,0.05),
                 probSeq2=np.arange(0.05,0.975,0.05),
                 dimEmbed=2,numEigen=10,k=10,alpha=0.5,
                 numLandmark=10,alg='PseudoRef'):
        self.probSeq1    = probSeq1
        self.probSeq2    = probSeq2
        self.dimEmbed    = 2 # other values not allowed currently
        self.numEigen    = numEigen
        self.k           = k
        self.numLandmark = numLandmark
        self.alg         = alg
        self.epsSeq1     = np.zeros(len(probSeq1))
        self.epsSeq2     = np.zeros(len(probSeq2))
        self.alpha       = alpha
            
    def link(self,dataSet1,dataSet2):
        dist1 = pdist(dataSet1)
        dist2 = pdist(dataSet2)
        
        for ii in range(len(self.probSeq1)):
            self.epsSeq1[ii] = np.square(np.percentile(dist1,100*self.probSeq1[ii]))
        for ii in range(len(self.probSeq2)):
            self.epsSeq2[ii] = np.square(np.percentile(dist2,100*self.probSeq2[ii]))
            
        gridSearchResult = self.gridSearchAlign(D1=squareform(dist1),
                                                D2=squareform(dist2),
                                                dataSet1=dataSet1,
                                                dataSet2=dataSet2)
            
        if self.alg == 'PseudoRef':
            lossMat = self.alpha*gridSearchResult['lossL2LandmarkMat'] + \
                      (1-self.alpha)*gridSearchResult['lossHausdorffMat']
        else:
            lossMat = gridSearchResult['lossHausdorffMat']
            
        eps1,eps2 = self.gridSearchInd(lossMat)
            
        result = self.dMapAlign(D1=squareform(dist1),D2=squareform(dist2),
                                dataSet1=dataSet1,dataSet2=dataSet2,
                                eps1=eps1,eps2=eps2)
        return result['X'],result['Y']
 
    def predict(self,linkDict,distTestTrain):
        Q = linkDict['Q'] 
        if self.alg == 'PseudoRef':    
            XOrg = np.matmul(linkDict['diffcoord'],Q.T)/linkDict['k'] + linkDict['landmarkMean']
            XOrg = self.inverseCenteringScaling(XOrg,center=linkDict['center'],scale=linkDict['scale'])    
            XNew = self.nystromExtension(XOrg,distTestTrain,eps=linkDict['eps'],eigenvals=linkDict['eigenvals'])    
            XNew = self.centeringScaling(XNew,center=linkDict['center'],scale=linkDict['scale'])
            XNew = XNew - linkDict['landmarkMean']
            XNew = np.matmul(XNew,Q)*linkDict['k']
        else:
            XOrg = np.matmul(linkDict['diffcoord'],Q.T)
            XOrg = self.inverseCenteringScaling(XOrg,center=linkDict['center'],scale=linkDict['scale'])
            XNew = self.nystromExtension(XOrg,distTestTrain,eps=linkDict['eps'],eigenvals=linkDict['eigenvals'])
            XNew = self.centeringScaling(XNew,center=linkDict['center'],scale=linkDict['scale'])
            XNew = np.matmul(XNew,linkDict['Q'])
        return XNew

    def loss(self,linkDict,xOrig,xNew,yNew,kMatch=3): 
        nTest,tmp  = np.shape(yNew)
        losses     = np.zeros(nTest)
        matchMat,_ = self.match(linkDict=linkDict,dataSet=yNew,kMatch=kMatch)
        for ii in range(nTest):
            ind        = matchMat[ii,:]
            xNew_i     = xNew[ii,:]
            estimates  = np.mean(xOrig[ind,:],axis=0)
            losses[ii] = np.sqrt(np.mean(np.square(xNew_i-estimates)))
        return losses
    
    def match(self,linkDict,dataSet,kMatch=5,pseudoCall=False):
        if pseudoCall == False:
            referenceDat = linkDict['diffcoord']
        else:
            referenceDat = linkDict
        nRowQuery = np.shape(dataSet)[0]
        indMat = np.zeros(shape=(nRowQuery,kMatch))
        weiMat = np.zeros(shape=(nRowQuery,kMatch))
        for ii in range(nRowQuery):
            tmp = referenceDat - dataSet[ii,:]
            tmp = np.sqrt(np.sum(tmp*tmp,axis=1))
            indMat[ii,:] = np.argsort(tmp)[0:kMatch]
            weiMat[ii,:] = np.sort(tmp)[0:kMatch]
        if kMatch == 1:
            indMat = np.ndarray.flatten(indMat)
            weiMat = np.ndarray.flatten(weiMat)
        weiMat = np.exp(-weiMat/np.std(weiMat))
        return indMat.astype(int),weiMat

###############################################################################

    def diffuseInvariant(self,D,eps,delta=10^-10):
        n = D.shape[0]
        K = np.exp(-1 * (D**2) / eps)
        ## next two lines added to match Jaehyeok's code
        v = K.sum(axis=1)
        K = K/np.outer(v,v.T)
        ##
        v = K.sum(axis=1)**0.5
        A = K / np.outer(v, v.T)
        ind = np.array([[row,col] 
            for col in range(len(A))
                for row in range(len(A[0]))
                    if A[row][col] > delta
        ])
        row = ind[..., 0]
        col = ind[..., 1]
        data = A[row, col]
        Asp = csc_matrix((data, (row, col)), shape=(n, n)).toarray()
        neff = min(self.numEigen+1,n)
        eigenvals, eigenvecs = eigsh(Asp, k=neff, which="LA", ncv=n)
        (eigenvals, eigenvecs) = (eigenvals[::-1], eigenvecs[..., ::-1])
        psi = eigenvecs / (eigenvecs[..., 0:1].dot(np.ones((1, neff))))
        lam = eigenvals[1:] / (1 - eigenvals[1:])
        lam = np.outer(np.array([1]*n), lam.T)
        X = psi[..., 1:self.numEigen+1] * lam[..., 0:self.numEigen]  
        return X,eigenvals[1:]

    def weightFn(self,X):
        neigh = NearestNeighbors(n_neighbors=self.k+1)
        neigh.fit(X)
        distances, indices = neigh.kneighbors(X)
        w = distances[:,self.k]
        return w/sum(w)
    
    def geometricCenter(self,X,w):
        prod = (X.T * w).T          # Multiply each row by a constant
        return np.sum(prod,axis=0)  # colSums

    def geometricCenteringScaling(self,X):
        w               = self.weightFn(X)
        geomCenter      = self.geometricCenter(X,w)
        XCentered       = X-geomCenter
        scaleConstant   = np.sqrt(np.max(np.sum(np.square(XCentered),axis=1)))
        XCenteredScaled = XCentered/scaleConstant
        return { 'XCenteredScaled':XCenteredScaled,
                 'geomCenter':geomCenter,
                 'scaleConstant':scaleConstant }

    def dMapGeometricCenteringScaling(self,D,eps):
        X,eigenvals = self.diffuseInvariant(D,eps=eps)
        X = X[:,:self.dimEmbed]
        XGeomTrans = self.geometricCenteringScaling(X)
        return XGeomTrans,eigenvals
    
    def f(self,eps,D):
        GeomTrans,Eigenvals = self.dMapGeometricCenteringScaling(D=D,eps=eps)
        return GeomTrans['XCenteredScaled'],GeomTrans['geomCenter'],GeomTrans['scaleConstant'],Eigenvals
    
    def align(self,X,Y,XLandmark,YLandmark):
        XLandmarkMean = np.mean(X[XLandmark,:],axis=0)
        YLandmarkMean = np.mean(Y[YLandmark,:],axis=0)
        X = X-XLandmarkMean
        Y = Y-YLandmarkMean
        k,Q = self.kQFn(X[XLandmark,:],Y[YLandmark,:])
        YDiffCoord = k * np.matmul(Y,Q)
        return { 'XDiffCoord':X,'YDiffCoord':YDiffCoord,'XLandmark':XLandmark,
                 'YLandmark':YLandmark,'k':k,'Q':Q,
                 'XLandmarkMean':XLandmarkMean,'YLandmarkMean':YLandmarkMean }
        
    def geometricPCA(self,XCenteredScaled):
        wCenteredScaled = self.weightFn(XCenteredScaled)
        W = np.diag(wCenteredScaled)
        XMatrix = np.asarray(XCenteredScaled)
        XtWX = np.matmul(np.matmul(XMatrix.T,W),XMatrix)
        w,v = np.linalg.eig(XtWX)
        return v

    def gridSearchAlign(self,D1,D2,dataSet1=None,dataSet2=None):
        listX = []
        for eps in self.epsSeq1:
            X,_,_,_ = self.f(eps,D1)
            listX.append(X)
        listY = []
        for eps in self.epsSeq2:
            Y,_,_,_ = self.f(eps,D2)
            listY.append(Y)
        lossL2LandmarkMat = np.zeros((len(self.epsSeq1),len(self.epsSeq2)))
        lossHausdorffMat  = np.zeros((len(self.epsSeq1),len(self.epsSeq2)))
        if self.alg == 'PseudoRef':
            # The issue: f() provides diffusion maps that may be randomly flipped across x and/or y.
            # makePseudoLandmarks creates landmarks for Y given X, and now flips the first diffusion coordinate if it
            # is "reversed" relative to the first diffusion coordinates of X.
            # To deal with the second diffusion coordinate, we now try both Y[:,1] and -Y[:,1] and see which gives
            # the minimum l2 landmark loss.
            for ii in range(len(self.epsSeq1)):
                for jj in range(len(self.epsSeq2)):
                    landmark1,landmark2,listY[jj] = self.makePseudoLandmarks(dataSet1,dataSet2,listX[ii],listY[jj])
                    result1 = self.align(X=listX[ii],Y=listY[jj],XLandmark=landmark1,YLandmark=landmark2)
                    lossL2Landmark1 = np.mean(np.sqrt(np.sum(np.square(result1['XDiffCoord'][landmark1,:]-result1['YDiffCoord'][landmark2,:]),axis=1)))
                    listY[jj][:,1] = np.negative(listY[jj][:,1])
                    result2 = self.align(X=listX[ii],Y=listY[jj],XLandmark=landmark1,YLandmark=landmark2)
                    lossL2Landmark2 = np.mean(np.sqrt(np.sum(np.square(result2['XDiffCoord'][landmark1,:]-result2['YDiffCoord'][landmark2,:]),axis=1)))
                    if lossL2Landmark1 < lossL2Landmark2:
                        result = result1
                        lossL2LandmarkMat[ii,jj] = lossL2Landmark1
                    else:
                        result = result2
                        lossL2LandmarkMat[ii,jj] = lossL2Landmark2
                    d1, i1, i2 = directed_hausdorff(result['XDiffCoord'],result['YDiffCoord'])
                    d2, j1, j2 = directed_hausdorff(result['YDiffCoord'],result['XDiffCoord'])
                    lossHausdorffMat[ii,jj] = np.max([d1,d2])
            return { 'lossL2LandmarkMat':lossL2LandmarkMat,'lossHausdorffMat':lossHausdorffMat }
        else:
            for ii in range(len(self.epsSeq1)):
                for jj in range(len(self.epsSeq2)):
                    QX = self.geometricPCA(listX[ii])
                    QY = self.geometricPCA(listY[jj])
                    XDiffCoord = np.matmul(listX[ii],QX)
                    _,lossHausdorffMat[ii,jj] = self.checkAllQY(XDiffCoord=XDiffCoord,YCenteredScaled=listY[jj],QY=QY)       
            return { 'lossHausdorffMat':lossHausdorffMat }
    
    def dMapAlign(self,D1,D2,dataSet1,dataSet2,eps1,eps2):
        X,XCenter,XScale,XEigenvals = self.f(eps1,D1)
        Y,YCenter,YScale,YEigenvals = self.f(eps2,D2)
        if self.alg == 'PseudoRef':
            landmark1,landmark2,Y = self.makePseudoLandmarks(dataSet1,dataSet2,X,Y)
            result1 = self.align(X=X,Y=Y,XLandmark=landmark1,YLandmark=landmark2)
            lossL2Landmark1 = np.mean(np.sqrt(np.sum(np.square(result1['XDiffCoord'][landmark1,:]-result1['YDiffCoord'][landmark2,:]),axis=1)))
            Y[:,1] = np.negative(Y[:,1])
            result2 = self.align(X=X,Y=Y,XLandmark=landmark1,YLandmark=landmark2)
            lossL2Landmark2 = np.mean(np.sqrt(np.sum(np.square(result2['XDiffCoord'][landmark1,:]-result2['YDiffCoord'][landmark2,:]),axis=1)))
            if lossL2Landmark1 < lossL2Landmark2:
                result = result1
            else:
                result = result2
            d1, i1, i2 = directed_hausdorff(result['XDiffCoord'],result['YDiffCoord'])
            d2, j1, j2 = directed_hausdorff(result['YDiffCoord'],result['XDiffCoord'])
            lossHausdorff = np.max([d1,d2])
            return { 'lossHausdorff':lossHausdorff,'lossL2Landmark':np.min([lossL2Landmark1,lossL2Landmark2]),
                    'X':{'diffcoord':result['XDiffCoord'],'center':XCenter,'scale':XScale,'eigenvals':XEigenvals,
                         'eps':eps1,'landmarkMean':result['XLandmarkMean'],'landmark':result['XLandmark'] },
                    'Y':{'diffcoord':result['YDiffCoord'],'center':YCenter,'scale':YScale,'eigenvals':YEigenvals,
                         'eps':eps2,'landmarkMean':result['YLandmarkMean'],'landmark':result['YLandmark'],
                         'k':result['k'],'Q':result['Q'] }    
                   }
        else:
            QX = self.geometricPCA(X)
            QY = self.geometricPCA(Y)    
            XDiffCoord = np.matmul(X,QX)
            indMinimizeHaus,lossHausdorff = self.checkAllQY(XDiffCoord=XDiffCoord,YCenteredScaled=Y,QY=QY)
            YDiffCoord = np.matmul(Y,QY*indMinimizeHaus)
            return { 'lossHausdorff':lossHausdorff,
                    'X':{'diffcoord':XDiffCoord,'center':XCenter,'scale':XScale,'Q':QX,
                         'eigenvals':XEigenvals,'eps':eps1},
                    'Y':{'diffcoord':YDiffCoord,'center':YCenter,'scale':YScale,'Q':QY*indMinimizeHaus,
                         'eigenvals':YEigenvals,'eps':eps2} }    
                                 
    def centeringScaling(self,X,center,scale):
        X = X - center
        return (X.T/scale).T 
    
    def inverseCenteringScaling(self,XCenteredScaled,center,scale):
        X = (XCenteredScaled.T * scale).T
        return X+center
    
    def nystromExtension(self,X,distBetweenNewAndOld,eps,eigenvals):
        _,nOld = np.shape(distBetweenNewAndOld)
        eigenvals = eigenvals[:X.shape[1]]
        XNew = np.exp(-np.square(distBetweenNewAndOld)/eps)
        v1 = np.sum(XNew,axis=1)
        v2 = np.sum(XNew,axis=0)
        XNew = XNew/np.outer(v1,v2.T)
        v = np.sum(XNew,axis=1)
        denom = np.tile(v,(nOld,1)).T
        XNew = XNew/denom
        XNew = np.matmul(np.matmul(XNew,X),np.diag(1/eigenvals))
        return XNew

    def kQFn(self,X,Y):
        U, s, V = np.linalg.svd(np.matmul(Y.T,X))
        Q = np.matmul(U,V.T)
        k = np.sum(s)/np.sum(np.diag(np.matmul(Y.T,Y)))
        return k,Q

    def makePseudoLandmarks(self,dataSet1,dataSet2,X,Y):
         gridX = np.linspace(np.min(X[:,0]),np.max(X[:,0]),self.numLandmark)
         landmark1 = np.zeros(len(gridX),dtype=np.int)
         ii = 0
         for x in gridX:
             landmark1[ii] = np.argmin(np.abs(X[:,0]-x))
             ii = ii+1
         matchMat,_ = self.match(linkDict=dataSet2,dataSet=dataSet1[landmark1,:],
                                 kMatch=1,pseudoCall=True)
         landmark2 = np.asarray(matchMat.T)
         if np.corrcoef(X[landmark1,0],Y[landmark2,0])[0,1] < 0:
             Y[:,0] = np.negative(Y[:,0])
             
         return landmark1,landmark2,Y

    def gridSearchInd(self,lossMat):
        s = np.sort(lossMat.flatten())[0]
        r,c = np.where(lossMat==s)
        return self.epsSeq1[r],self.epsSeq2[c]     

    def calHausdorffUseInd(self,ind,QY,YCenteredScaled,XDiffCoord):
        YDiffCoordInd = np.matmul(YCenteredScaled,QY*ind)
        d1, i1, i2 = directed_hausdorff(XDiffCoord,YDiffCoordInd)
        d2, j1, j2 = directed_hausdorff(YDiffCoordInd,XDiffCoord)
        return np.max([d1,d2])

    def checkAllQY(self,XDiffCoord,YCenteredScaled,QY):
        allPossibleInd = np.array([(x,y) for x in (1,-1) for y in (1,-1)]) # assumes np.shape(QY) = 2x2
        r,c = np.shape(allPossibleInd)
        haus = np.zeros(r)
        for ii in range(r):
            haus[ii] = self.calHausdorffUseInd(allPossibleInd[ii,:],QY,YCenteredScaled,XDiffCoord)
            indMinimizeHaus = allPossibleInd[np.argmin(haus),:]    # may need astype(int) in here somewhere
        return indMinimizeHaus,np.min(haus)
