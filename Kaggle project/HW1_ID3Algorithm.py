import pandas as pd
import numpy as np
import random


class treeNode:
    def __init__(self):
        self.subNodes = None
        self.nextNode = None
        self.leafNode = None
        self.depth = 0
        self.feature = None
        self.value = None

class DT:
    def __init__(self,data,m='entropy', depth=None, pickTie=True,
                 num=False, multiples=None):
        self.treeNode = None
        self.features = np.array(data.iloc[:,:-1])
        self.feature_names = np.array(data.columns[:-1])
        self.labels = np.array(data.iloc[:,-1])
        self.uniqueLabels = list(set(self.labels))
        self.num=num
        self.multiples = np.ones((len(self.features),))
        self.med=None
        self.num_Indx = []
        self.pickAny=pickTie

        if depth is None:
            self.deplim = len(self.feature_names)

        elif depth is not None:
            self.deplim = depth
        if self.deplim > len(self.feature_names) or self.deplim <1:
            raise ValueError('Invalid depth')

        if multiples is None:
            self.multiples=np.ones((len(self.features),))
        else:
            self.multiples = multiples


        if m == 'entropy':
            self.IGmethod = self.calculateEntropy
        elif m == 'majorityError':
            self.IGmethod = self.calculateME
        else:
            self.IGmethod = self.calculateGI

    def calculateEntropy(self,indx):
       labels = self.labels[indx]
       multiples = self.multiples[indx]
       labelNos = np.zeros((len(self.uniqueLabels),))
       for i, con in enumerate(self.uniqueLabels):
           indxset = np.array(labels)==con
           labelNos[i]=sum(multiples[indxset])

       if len(labelNos[labelNos!=0])==1:
           entropy=0
       else:
           prob=labelNos/sum(labelNos)
           prob=prob[prob!=0]
           entropy=-(np.sum(prob*np.log2(prob)))
       return entropy

    def calculateME(self,indx):
        labels = self.labels[indx]
        multiples = self.multiples[indx]
        labelNos = np.zeros((len(self.uniqueLabels),))
        for i, con in enumerate(self.uniqueLabels):
            indxset = np.array(labels) == con
            labelNos[i] = sum(multiples[indxset])

        Max = max(labelNos)
        ME = 1- Max/sum(labelNos)

        return ME

    def calculateGI(self,indx):
        labels = self.labels[indx]
        multiples = self.multiples[indx]
        labelNos = np.zeros((len(self.uniqueLabels),))
        for i, con in enumerate(self.uniqueLabels):
            indxset = np.array(labels) == con
            labelNos[i] = sum(multiples[indxset])

        prob=labelNos/sum(labelNos)
        GI = 1 - (np.sum(prob**2))
        return GI

    def calculateIG(self,indx,FID):
        IG=self.IGmethod(indx)
        featureVals = self.features[indx,FID]
        s = sum(self.multiples[indx])
        feature_uni = list(set(featureVals))
        IG_f = 0
        for v in range(len(feature_uni)):
            indxl=np.where(featureVals==feature_uni[v])[0]
            feature_id=list(indx[list(indxl)])
            feature_s=sum(self.multiples[feature_id])
            IG_sub=self.IGmethod(feature_id)
            IG_f += feature_s/ s*IG_sub
        IG-=IG_f
        return IG

    def nextFeature(self,indx,feature_names):
        fIDs = [i for i in range(len(self.feature_names)) if self.feature_names[i] in feature_names]
        fIG = [self.calculateIG(indx, fID) for fID in fIDs]

        if self.pickAny:
            bestgain = np.array(fIG) == max(fIG)
            if sum(bestgain) != 1:
                sameindx=[i for i in range(len(bestgain)) if bestgain[i]==True]
                anyindx = random.choice(sameindx)
                bestFeature=feature_names[fIG.index(fIG[anyindx])]
                bestFeatureIDX=fIDs[fIG.index(fIG[anyindx])]
            else:
                bestFeature=feature_names[fIG.index(max(fIG))]
                bestFeatureIDX=fIDs[fIG.index(max(fIG))]

        else:
            bestFeature=feature_names[fIG.index(max(fIG))]
            bestFeatureIDX=fIDs[fIG.index(max(fIG))]

        return bestFeature, bestFeatureIDX

    def convertToBinary(self,indx,feature_names):
        fIDs = list(range(len(self.feature_names)))
        fID_type=self.features[0,fIDs]
        nFID = [Iden for i, Iden in enumerate(fIDs) if isinstance(fID_type[i], (float,int))]
        self.med=np.zeros((len(feature_names),))
        for ftr in nFID:
            med=np.median(self.features[indx,ftr])
            self.med[ftr] = med
            self.num_Indx.append(ftr)
            self.features[:,ftr] = self.features[:,ftr] > med

    def recursiveFunc(self,indx,feature_names,treenode,lastMax=None):
        if not treenode:
            treenode=treeNode()
        featureLabels=self.labels[indx]
        if len(set(featureLabels)) ==True:
            treenode.feature=self.labels[indx[0]]
            treenode.leafNode=1
            return treenode

        elif len(indx)==False:
            treenode.feature=lastMax
            treenode.leafNode=1
            return treenode

        uno,loc = np.unique(featureLabels,return_inverse=True)
        same=uno[np.argmax(np.bincount(loc))]

        if treenode.depth == self.deplim:
            treenode.feature = same
            treenode.leafNode=True
            return treenode

        bestFeature, bestFeatureIDX = self.nextFeature(indx,feature_names)
        treenode.feature=bestFeature
        treenode.subNodes=[]
        selectedFeatVals=list(set(self.features[:,bestFeatureIDX]))

        for v in selectedFeatVals:
            subnode=treeNode()
            subnode.depth=treenode.depth+1
            subnode.value=v
            treenode.subNodes.append(subnode)
            indpos = np.where(self.features[indx,bestFeatureIDX]==v)[0]
            subnodeINDX=indx[list(indpos)]
            if len(subnodeINDX)==False:
                subnode.nextNode=self.recursiveFunc(subnodeINDX,[],subnode,lastMax=same)
            else:
                if len(feature_names)!=False and bestFeature in feature_names:
                    nextfeat=feature_names.copy()
                    indxelim=np.where(bestFeature==nextfeat)[0][0]
                    nextfeat=np.delete(nextfeat,indxelim)
                subnode.nextNode=self.recursiveFunc(subnodeINDX,nextfeat,subnode)

        return treenode

class testtree:
    def __init__(self,learntTree,check,ti,num=False,multiples=None):
        self.wholeTree=learntTree
        self.feature_names=np.array(check.columns)
        self.beginCheck=np.array(check.iloc[:,:-1])
        self.beginlabelling = np.array(check.iloc[:,-1])
        self.num=num
        if self.num:
            self.med=ti.med
            self.num_Indx=ti.num_Indx
        if multiples is None:
            self.multiples=np.ones((len(self.beginCheck),))
        else:
            self.multiples=multiples

    def loopings(self, ptreeNode,sst,ssl):
        errrors=np.zeros((len(ssl),))
        for r in range(len(sst)):
            leafnode=False
            treenode=ptreeNode
            while not leafnode:
                splitt=treenode.feature
                splittIndx = np.where(np.array(splitt==self.feature_names))[0][0]
                valueNxt=sst[r,splittIndx]
                for nxtNode in treenode.subNodes:
                    if nxtNode.value==valueNxt:
                        treenode=nxtNode
                        break
                if treenode.leafNode==1:
                    leafnode=1
            errrors[r]=ssl[r]==treenode.feature
        return errrors

def ID3_algo(self):

    indx=np.arange(len(self.features))
    feature_names = self.feature_names.copy()
    if self.num:
        self.convertToBinary(indx, feature_names)
    self.treeNode=self.recursiveFunc(indx,feature_names,self.treeNode)
    return self.treeNode

def testID3(self):
    presentnode=self.wholeTree
    fulltest=self.beginCheck
    fulllabelling=self.beginlabelling
    if self.num:
        for indx in self.num_Indx:
            fulltest[:,indx] = fulltest[:,indx].copy()>self.med[indx]

    errrors=self.loopings(presentnode,fulltest,fulllabelling)


    errrors2=errrors*self.multiples
    errortotal = np.sum(errrors2) / sum(self.multiples)

    return errrors, errortotal


"""
TESTING
"""

class testing:
    def __init__(self,ms,data,depth,num=False,same=True):
        self.ms=ms
        self.feature_names=np.array(data[0].columns[:-1])
        self.datatrain=data[0]
        self.datatest=data[1]
        self.depth=np.linspace(1,depth,depth)
        self.num=num
        self.tr_error=np.zeros((len(ms),len(self.depth)))
        self.te_error=np.zeros((len(ms),len(self.depth)))
        self.same=same

    def testerror(self,DT,check,ti,num=False):
        errror=0
        ei=testtree(DT,check,ti,num=num)
        _,errror=testID3(ei)
        return errror

    def check(self):
        for i, m in enumerate(self.ms):
            for j, dep in enumerate(self.depth):
                ti=None
                DTT=None
                ti=DT(self.datatrain,depth=dep, m=m,num=self.num,pickTie=self.same)

                print('DECISION TREE building with depth size:{} using {}'.format(dep,m))
                DD=ID3_algo(ti)
                print('Tree = DONE')
                print('Now, training and testing')
                self.tr_error[i,j]=self.testerror(DD,self.datatrain,ti,num=self.num)
                self.te_error[i, j] = self.testerror(DD, self.datatest, ti, num=self.num)
                print('Check complete\n')
        print('Finish\n')

        return self.tr_error,self.te_error





