import matplotlib.pyplot as plt
import numpy as np

from sklearn import decomposition
from sklearn import datasets
from sklearn.svm import SVC

from parser0 import *

K = 3
PERCENT = 10
PASSRATIO = 0.7
SUCCESSRATIO = 0.9
MODE = 4

#####################################################################################
def Draw(fig, ax, X, label0):
    x = np.arange(0, len(X))
    y = X
    ax.plot(x, y, label=label0)


def Fill(fig, ax, minX, maxX, label0):
    x = np.arange(0, max(len(minX), len(maxX)))
    y1 = minX
    y2 = maxX
    ax.fill_between(x, y1, y2, label=label0, facecolor='gray')
#####################################################################################

def DrawHitLineChart(X, yActive, yPassive, activeLabel):
    fig, ax = plt.subplots()
    plt.xlabel('kMultiplier')
    plt.ylabel('meanHitRatio')
    ax.plot(X, yActive, label = activeLabel)
    ax.plot(X, yPassive, label = 'others')
##    Draw(fig, ax, yActive, activeLabel)
##s    Draw(fig, ax, yPassive, 'others')
    ax.legend()
    ax.set_title('Hit Ratios of %s and others (in model %s)' % (activeLabel, activeLabel))



def DrawHitLineChart2(X, ys, labels, activeLabel):
    fig, ax = plt.subplots()
    plt.xlabel('kMultiplier')
    plt.ylabel('meanHitRatio')
    for j in range(MODE):
        ax.plot(X, ys[j], label = labels[j])
    ax.legend()
    ax.set_title('Hit Ratios (in model %s)' % (activeLabel))
    
    
def DrawEnvelope(meanCurves, stdCurves, labels, independent = True):
    XLABEL = 'timestamp'
    YLABEL = 'pca_value'
    TITLE = 'Envelope (mean +- ' + str(K) + ' std)'
    
    if(independent):
        fig, ax = plt.subplots(len(meanCurves), sharex = True, sharey = True)
    
        plt.xlabel(XLABEL)
        plt.ylabel(YLABEL)
        ax[0].set_title(TITLE)
        
        for i in range(len(meanCurves)):
            y1 = meanCurves[i] - K * stdCurves[i]
            y2 = meanCurves[i] + K * stdCurves[i]
            Draw(fig, ax[i], meanCurves[i], labels[i] + '_mean')    
            Fill(fig, ax[i], y1, y2, labels[i] + '_envelope')
            ax[i].legend()
    else:
        fig, ax = plt.subplots()
    
        plt.xlabel(XLABEL)
        plt.ylabel(YLABEL)
        ax.set_title(TITLE)
        
        for i in range(len(meanCurves)):
            y1 = meanCurves[i] - K * stdCurves[i]
            y2 = meanCurves[i] + K * stdCurves[i]
            Draw(fig, ax, meanCurves[i], labels[i] + '_mean')    
            Fill(fig, ax, y1, y2, labels[i] + '_envelope')
        ax.legend()

def DrawEnvelope3(peakMeans, peakStds, peakKX, valeyMeans, valeyStds, valeyKX, labels):
    XLABEL = 'timeStamp'
    YLABEL = 'PCA_value'
    TITLE = 'Envelope'


    fig, ax = plt.subplots()
    plt.xlabel(XLABEL)
    plt.ylabel(YLABEL)
    ax.set_title(TITLE)    
    
    colors = ['blue', 'orange', 'green', 'red']
    for i in range(MODE):
        x = np.arange(0, PAGESIZE)
        y = [peakMeans[i] for _ in range(PAGESIZE)]
        ax.plot(x, y, label = labels[i] + '_peak', color = 'gray')    
        
        x = np.arange(0, PAGESIZE)
        y = [valeyMeans[i] for _ in range(PAGESIZE)]
        ax.plot(x, y, label = labels[i] + '_valey', color = 'gray') 
        
        y1 = [peakMeans[i] - peakKX[i] * peakStds[i] for _ in range(PAGESIZE)]
        y2 = [peakMeans[i] + peakKX[i] * peakStds[i] for _ in range(PAGESIZE)]
        ax.fill_between(x, y1, y2, label = labels[i] + '_envelope', facecolor = colors[i])
        
        y1 = [valeyMeans[i] - valeyKX[i] * valeyStds[i] for _ in range(PAGESIZE)]
        y2 = [valeyMeans[i] + valeyKX[i] * valeyStds[i] for _ in range(PAGESIZE)]
        ax.fill_between(x, y1, y2, label = labels[i] + '_envelope', facecolor = colors[i])
        
    ax.legend()    
    
    
def DrawEnvelope2(trainingDataList, labels):
    XLABEL = 'timestamp'
    YLABEL = 'pca_value'
    TITLE = 'Envelope (mean +- ' + str(K) + ' std)'
    
    meanCurves = []
    stdCurves = []
    for i in range(MODE):
        trainData = np.array(trainingDataList[i])
        mean = np.mean(trainData)
        std = np.std(trainData)
        meanCurves.append([mean for _ in range(PAGESIZE)])
        stdCurves.append([std for _ in range(PAGESIZE)])
        
    fig, ax = plt.subplots()

    plt.xlabel(XLABEL)
    plt.ylabel(YLABEL)
    ax.set_title(TITLE)
    
    for i in range(len(meanCurves)):
        meanCurves[i] = np.array(meanCurves[i])
        stdCurves[i] = np.array(stdCurves[i])
        
        y1 = meanCurves[i] - K * stdCurves[i]
        y2 = meanCurves[i] + K * stdCurves[i]
        Draw(fig, ax, meanCurves[i], labels[i] + '_mean')    
        Fill(fig, ax, y1, y2, labels[i] + '_envelope')
        
    ax.legend()

def FindValeysSorted(X, RATIO = 10):
    valeys = []
    pagesize = len(X)
    for j in range(1, pagesize - 1):
        now = X[j]
        prevv = X[j - 1]
        nextt = X[j + 1]
        #valey detect
        if now < prevv and now < nextt:
            valeys.extend(now)
            
    valeys.sort()
    valeys = valeys[:int(pagesize * RATIO / 100)]
    
    return valeys

def FindPeaksSorted(X, RATIO = 10):
    peaks = []
    pagesize = len(X)
    for j in range(1, pagesize - 1):
        now = X[j]
        prevv = X[j - 1]
        nextt = X[j + 1]
        # peak detect
        if now > prevv and now > nextt:
            # stored absolute value
            peaks.extend(now)
    
    peaks.sort()
    peaks.reverse()
    peaks = peaks[:int(pagesize * RATIO / 100)]
    
    return peaks


def CalculateHitRatio(mean, std, spotCurve, k = K):
    hitCount = 0
    for i in range(len(spotCurve)):
        if abs(spotCurve[i] - mean) <= k * std:
            hitCount += 1
    return float(hitCount) / len(spotCurve)


def FindKX(means, stds, spotList):
    kX = []
    for i in range(MODE):
        tmps = []

        for k0 in range(5, 100+1, 1):
            kMulti = k0 * 0.1
            h = [] # list of tuple(h0, h1, h2, h3)
            for j in range(MODE):
                hitRatios = []
                for k in range(len(spotList[j])):
                    hitRatio = CalculateHitRatio(means[i], stds[i], spotList[j][k], kMulti)
                    hitRatios.append(hitRatio)
                hitRatios = np.array(hitRatios)
                h.append(np.mean(hitRatios))

            gaps = []
            for j in range(MODE):
                if i != j:
                    gaps.append(h[i] - h[j])

            if len(gaps) > 0:
                tmps.append([min(gaps), kMulti])
        
  ##      print(tmps)
        nowMax = 0
        nowK = 0
        for k1 in range(len(tmps)):
            if tmps[k1][0] > nowMax:
                nowMax, nowK = tmps[k1][0], tmps[k1][1]
                
        kX.append(nowK)    
    return kX
        
        
def Train(trainData):
    """
    we consider larger peaks which occupy top (RATIO)%
    """

    # preprocess
    trainDataList = []
    for i in range(MODE):
        trainDataList.append(Paging(trainData[i]))
    
    # preprocess peak and valey
    peakMeans = []
    peakStds = []
    
    valeyMeans = []
    valeyStds = []
    for i in range(MODE):
        # find peak
        peaks = FindPeaksSorted(trainData[i])
        peakMeans.append(np.mean(peaks))
        peakStds.append(np.std(peaks))
        
        valeys = FindValeysSorted(trainData[i])
        valeyMeans.append(np.mean(valeys))
        valeyStds.append(np.std(valeys))
    

    peaksList = []  
    valeysList = []
    for i in range(MODE):
        peakList = []
        valeyList = []
        for k in range(len(trainDataList[i])):
            peaks = FindPeaksSorted(trainDataList[i][k])
            valeys = FindValeysSorted(trainDataList[i][k])
            
            peakList.append(peaks)
            valeyList.append(valeys)
            
        peaksList.append(peakList)
        valeysList.append(valeyList)
    
    # find the best K for every mode, and put into kX
    peakKX = FindKX(peakMeans, peakStds, peaksList)
    valeyKX = FindKX(valeyMeans, valeyStds, valeysList)
    
    return (peakMeans, peakStds, peakKX, valeyMeans, valeyStds, valeyKX)


def Predict(data, peakMeans, peakStds, peakKX, valeyMeans, valeyStds, valeyKX):
    # preprocess peak
    peaks = FindPeaksSorted(data, 10)
    valeys = FindValeysSorted(data, 10)
    
    tmps = []
    for i in range(MODE):
        hitPeakRatio = CalculateHitRatio(peakMeans[i], peakStds[i], peaks, peakKX[i])
        hitValeyRatio = CalculateHitRatio(valeyMeans[i], valeyStds[i], valeys, valeyKX[i])
        tmps.append([(hitPeakRatio + hitValeyRatio) / 2.0, i])
    
    return max(tmps)[1]


def WriteByLine(fpp, X):
    for i in range(MODE):
        fpp.write(str(X[i]))
        if i < MODE - 1:
            fpp.write(',')
        else:
            fpp.write('\n')
    
def WriteToFile(peakMeans, peakStds, peakKX, valeyMeans, valeyStds, valeyKX):
    fpp = open('motorcycle.txt', 'w')
    WriteByLine(fpp, peakMeans)
    WriteByLine(fpp, peakStds)
    WriteByLine(fpp, peakKX)
    WriteByLine(fpp, valeyMeans)
    WriteByLine(fpp, valeyStds)
    WriteByLine(fpp, valeyKX)
    fpp.close()    
    
def Run(trainPrefix, testPrefix):
    labels = ['fan0',
              'fan1',
              'fan2',
              'fan3']
    
    trainFileList = []
    testFileList = []
    for i in range(MODE):
        trainFileList.append(trainPrefix + '_' + labels[i])
        testFileList.append(testPrefix + '_' + labels[i])
    
    # preprocess
    trainDataList = []
    testDataList = []
    allTrainData = []
    allTestData = []
    
    
    # read file   
    for i in range(MODE):
        trainData = Parse(Read(trainFileList[i]))
        testData = Parse(Read(testFileList[i]))
        ##
        allTrainData.append(trainData)
        allTestData.append(testData)
        ##
        trainDataList.append(Paging(trainData))
        testDataList.append(Paging(testData))
    
    peakMeans, peakStds, peakKX, valeyMeans, valeyStds, valeyKX = Train(allTrainData)
    WriteToFile(peakMeans, peakStds, peakKX, valeyMeans, valeyStds, valeyKX)
    
    
    for i in range(MODE):
        # now at mode i
        print('now at mode %d' % i)
       ## print(Predict(allTestData[i],  peakMeans, peakStds, kX))
        result = []
        ##print(testDataList[i])
        for j in range(len(testDataList[i])):
            result.append(Predict(testDataList[i][j], peakMeans, peakStds, peakKX, valeyMeans, valeyStds, valeyKX))
        print(result)
        
    DrawEnvelope3(peakMeans, peakStds, peakKX, valeyMeans, valeyStds, valeyKX, labels)
    # draw
   ## DrawEnvelope(meanCurves, stdCurves, labels) 
   ## DrawEnvelope(meanCurves, stdCurves, labels, False) 
    
    ## plt.savefig(figurePrefix + ('@pagesize=%d' % pagesize))
        

   ## DrawXYZ(trainingFileList)    
   ## DrawIndepenet(files)
   ## DrawMixed(data, files[0])               
    plt.show()       
    
    
if __name__ == '__main__':
##    Run('0328_2_9600_d100', '0328_2_9600_d100')
##    Run('0328_2_9600_d100', '0328_3_9600_d100')
##    Run('0328_2_9600_d100', '0328_4_9600_d100')
##    Run('0328_2_9600_d100', '0328_5_9600_d100')
   ## Run('0328_3_9600_d100', '0328_3_9600_d100')
   ## Run('0328_4_9600_d100', '0328_4_9600_d100')
   ## Run('0328_5_9600_d100', '0328_5_9600_d100')
    
 ##   Run('0329_1', '0329_1')
 ##   Run('0329_2', '0329_2')
    
    Run('0330_2', '0330_2')
    
    
    