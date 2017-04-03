import matplotlib.pyplot as plt
import numpy as np
from parser0 import *
from mpl_toolkits.mplot3d import Axes3D

TOP_PEAK_PERCENT = 10

MODE = 4

LABELS = ['fan0',
          'fan1',
          'fan2',
          'fan3']


def FindValleysSorted(X, ratio = TOP_PEAK_PERCENT):
    valleys = []
    pagesize = len(X)
    for j in range(1, pagesize - 1):
        now = X[j]
        prevv = X[j - 1]
        nextt = X[j + 1]
        #valley detect
        if now < prevv and now < nextt:
            valleys.extend(now)
            
    valleys.sort()
    valleys = valleys[:int(pagesize * ratio / 100)]
    
    return valleys


def FindPeaksSorted(X, ratio = TOP_PEAK_PERCENT):
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
    peaks = peaks[:int(pagesize * ratio / 100)]
    
    return peaks


def PlotScatter(data, filenamePrefix = ''):
    fig = plt.figure()
    ax = Axes3D(fig)
    # preprocess
    dataList = []
    for i in range(MODE):
        dataList.append(Paging(data[i]))
        
    ax.set_xlabel('meanGap1')
    ax.set_ylabel('meanGap2')
    ax.set_zlabel('meanGap3')
    ax.set_title('Scatters of Mean Gaps in 3D (' + filenamePrefix + ')')    

    for i in range(MODE):
        gapList = []
        for k in range(len(dataList[i])):
            gap = []
            for j in range(3):
                fragment = dataList[i][k][int(PAGESIZE * j / 3): int(PAGESIZE * (j + 1) / 3)]
                peaks = FindPeaksSorted(fragment)
                valleys = FindValleysSorted(fragment)
                if len(peaks) == 0:
                    peaks.append(0)
                if len(valleys) == 0:
                    valleys.append(0)
                gap.append(np.mean(peaks) - np.mean(valleys))
            gapList.append(gap)
            
        nowList = [[], [], []]
        for j in range(len(gapList)):
            for k in range(3):
                nowList[k].append(gapList[j][k])

        ax.scatter(nowList[0], nowList[1], nowList[2], label = LABELS[i])
    
    ax.legend()
        
    plt.savefig(filenamePrefix +'.png') 
    
    
def Train(trainData):
    """
    we consider larger peaks which occupy top (RATIO)%
    """

    # preprocess
    trainDataList = []
    for i in range(MODE):
        trainDataList.append(Paging(trainData[i]))
    
    # split every file
    gapsXList = []
    gapsYList = []

    
    for i in range(MODE):
        gapList = []
        for k in range(len(trainDataList[i])):
            gap = []
            for j in range(3):
                fragment = trainDataList[i][k][int(PAGESIZE * j / 3): int(PAGESIZE * (j + 1) / 3)]
                peaks = FindPeaksSorted(fragment)
                valleys = FindValleysSorted(fragment)
                if len(peaks) == 0:
                    peaks.append(0)
                if len(valleys) == 0:
                    valleys.append(0)
                gap.append(np.mean(peaks) - np.mean(valleys))
            gapList.append(gap)
                    
            
    return gapsXList, gapsYList


def Predict(data, peakMeans, peakStds, peakKX, valleyMeans, valleyStds, valleyKX):
    # preprocess peak
    peaks = FindPeaksSorted(data)
    valleys = FindValleysSorted(data)

    tmps = []
    for i in range(MODE):
        hitPeakRatio = CalculateHitRatio(peakMeans[i], peakStds[i], peaks, peakKX[i])
        hitValleyRatio = CalculateHitRatio(valleyMeans[i], valleyStds[i], valleys, valleyKX[i])
        tmps.append([(hitPeakRatio + hitValleyRatio) / 2.0, i])

    return max(tmps)[1]


def WriteByLine(fpp, X):
    for i in range(MODE):
        fpp.write(str(X[i]))
        if i < MODE - 1:
            fpp.write(',')
        else:
            fpp.write('\n')

def WriteToFile(peakMeans, peakStds, peakKX, valleyMeans, valleyStds, valleyKX):
    fpp = open('motorcycle.txt', 'w')
    WriteByLine(fpp, peakMeans)
    WriteByLine(fpp, peakStds)
    WriteByLine(fpp, peakKX)
    WriteByLine(fpp, valleyMeans)
    WriteByLine(fpp, valleyStds)
    WriteByLine(fpp, valleyKX)
    fpp.close()    
    