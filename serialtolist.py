import sys, serial
import numpy as np
from collections import deque

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
    
def DrawIndepenet(files):
    fig, ax = plt.subplots(len(files), sharex = True, sharey = True)
    plt.xlim(0, 3000)
    #plt.ylim(-100, 100)
    plt.xlabel('timestamp');
    plt.ylabel('pca_value');
    for i in range(len(files)):
        file = files[i]
        records = LoadCSV(file)
        records = GetPCA(records, 1)
        Draw(fig, ax[i], records, file)
        ax[i].legend()    
    
    ax[0].set_title('PCA of dataset')

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
    
    

def DrawMixed(data, labels):
    fig, ax = plt.subplots()
    #plt.ylim(-100, 100)
    plt.xlabel('timestamp');
    plt.ylabel('pca_value');
    for i in range(len(data)):
        Draw(fig, ax, data[i], labels)
        
    ax.legend()      
    ax.set_title('PCA of dataset')


def DrawXYZ(files):
    fig, ax = plt.subplots(3, sharex = True)
    plt.xlim(0, 3000)
   # plt.ylim(-100, 100)
    plt.xlabel('timestamp');
    plt.ylabel('acce');
    ax[0].set_title('original dataset')
    for i in range(len(files)):
        file = files[i]
        records = LoadCSV(file)
        
        xs, ys, zs = [], [], []
        for (x,y,z) in records:
            xs.append(x)
            ys.append(y)
            zs.append(z)
        Draw(fig, ax[0], xs, file + '_X')
        Draw(fig, ax[1], ys, file + '_Y')
        Draw(fig, ax[2], zs, file + '_Z')

    ax[0].legend()     
    ax[1].legend()  
    ax[2].legend()  


def CreateCurve(data):
    meanCurve = []
    stdCurve = []
    
    length = range(len(data[0]))

    for j in length:
        group = []
        for i in range(len(data)):
            group.append(data[i][j])
        group = np.array(group)
        meanCurve.append(np.mean(group))
        stdCurve.append(np.std(group))
        
    meanCurve = np.array(meanCurve)
    stdCurve = np.array(stdCurve)
    
    return meanCurve, stdCurve


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

    plt.xlabel(XLABEL);
    plt.ylabel(YLABEL);
    ax.set_title(TITLE)
    
    for i in range(len(meanCurves)):
        meanCurves[i] = np.array(meanCurves[i])
        stdCurves[i] = np.array(stdCurves[i])
        
        y1 = meanCurves[i] - K * stdCurves[i]
        y2 = meanCurves[i] + K * stdCurves[i]
        Draw(fig, ax, meanCurves[i], labels[i] + '_mean')    
        Fill(fig, ax, y1, y2, labels[i] + '_envelope')
        
    ax.legend()


def BuildSparseVector(meanCurve, stdCurve, spotCurve):
    vec = []
    for i in range(len(meanCurve)):
        if spotCurve[i] > meanCurve[i] + K * stdCurve[i]:
            vec.append(1)
        elif spotCurve[i] < meanCurve[i] - K * stdCurve[i]:
            vec.append(-1)
        else:
            vec.append(0)
    return vec


def FindPeaksSorted(X, RATIO = 10):
    peaks = []
    pagesize = len(X)
    for j in range(1, pagesize - 1):
        now = abs(X[j])
        prevv = abs(X[j - 1])
        nextt = abs(X[j + 1])
        # peak detect
        if now > prevv and now > nextt:
            # stored absolute value
            peaks.extend(now)
    
    peaks.sort()
    peaks.reverse()
    peaks = peaks[:int(pagesize * RATIO / 100)]
    
    return peaks


def CalculateHitRatio(meanCurve, stdCurve, spotCurve):
    hitCount = 0
    for i in range(len(meanCurve)):
        if abs(spotCurve[i] - meanCurve[i]) <= K * stdCurve[i]:
            hitCount += 1
    return float(hitCount) / len(meanCurve)


PEAKMEANS = []
PEAKSTDS = []
KX = []

def CalculateHitRatio2(mean, std, spotCurve, k = K):
    hitCount = 0
    for i in range(len(spotCurve)):
        if abs(spotCurve[i] - mean) <= k * std:
            hitCount += 1
    return float(hitCount) / len(spotCurve)

def Predict(data, peakMeans = PEAKMEANS, peakStds = PEAKSTDS, kX = KX, successRatio = SUCCESSRATIO):
    # preprocess peak
    peaks = FindPeaksSorted(data, 10)
   ## print('peak = ' + str(peaks))
    tmps = []
   ## print(peaks)

    for i in range(MODE):
        hitRatio = CalculateHitRatio2(peakMeans[i], peakStds[i], peaks, kX[i])

        tmps.append([hitRatio, i])
    return max(tmps)[1]

def _Predict(buffer):
    data = Parse(buffer)
    return Predict(data)

# class that holds analog data for N samples
class AnalogData:
    # constr
    def __init__(self, maxLen):
        self.ax = deque([0.0]*maxLen)
        self.ay = deque([0.0]*maxLen)
        self.az = deque([0.0]*maxLen)
        self.maxLen = maxLen

    # ring buffer
    def addToBuf(self, buf, val):
        if len(buf) < self.maxLen:
            buf.append(val)
        else:
            buf.pop()
            buf.appendleft(val)

    # add data
    def add(self, data):
        assert(len(data) == 3)
        self.addToBuf(self.ax, data[0])
        self.addToBuf(self.ay, data[1])
        self.addToBuf(self.az, data[2])
    def mergeToList(self):
        tmps=[[],[],[]]
        tmps[0]=list(self.ax)
        tmps[1]=list(self.ay)
        tmps[2]=list(self.az)
    ##    dataList = []
    ##    for i in range(maxLen):
    ##        dataList.append([tmps[0], tmps[1], tmps[2]])
        return tmps

      
      
# main() function
def main():
    # open feature data
    fp = open('motorcycle.txt', 'r')
    peakMeans = fp.readline().split(',')
    peakStds = fp.readline().split(',')
    kX = fp.readline().split(',')
    for i in range(4):
        peakMeans[i] = float(peakMeans[i])
        peakStds[i] = float(peakStds[i])
        kX[i] = float(kX[i])
        
    ##print(PEAKMEANS)
   ## print(PEAKSTDS)
   ## print(KX)
    
    # plot parameters
    analogData = AnalogData(200)
    dataList=[]
    print('start to receive data...')
    
    # open serial port
    ser = serial.Serial("COM4", 9600)
    ser.readline()
    while True:
        try:
            line = ser.readline()
            try:
                data = [float(val) for val in line.decode().split(',')]
                if(len(data) == 3):
                    analogData.add(data)
                    dataList=analogData.mergeToList()
                    print('======')
                    a = []
                    for k in range(len(dataList[0])):
                        a.append([dataList[0][k], dataList[1][k], dataList[2][k]])
                    realData = Parse(a)
                    print(Predict(realData, peakMeans, peakStds, kX))
                    ##print(_Predict(dataList))
                   ## print("-------")
                   ## print(dataList)
                   ## print("-------")
            except:
                pass
        except KeyboardInterrupt:
            print('exiting')
            break
    # close serial
    ser.flush()
    ser.close()

# call main
if __name__ == '__main__':
    
    main()