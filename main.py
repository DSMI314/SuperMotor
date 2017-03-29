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


def CalculateHitRatio2(mean, std, spotCurve, k = K):
    hitCount = 0
    for i in range(len(spotCurve)):
        if abs(spotCurve[i] - mean) <= k * std:
            hitCount += 1
    return float(hitCount) / len(spotCurve)

def Predict(data, peakMeans, peakStds, successRatio = SUCCESSRATIO):
    # preprocess peak
    peaks = FindPeaksSorted(data, 10)
    peakMean = np.mean(peaks)
    peakStd = np.std(peaks)
    
   ## print('peak = ' + str(peaks))
    hitRatios = []
    for i in range(MODE):
        hitRatio = CalculateHitRatio2(peakMeans[i], peakStds[i], peaks)
        hitRatios.append(hitRatio)
   ## print(hitRatios)
        if hitRatio >= successRatio:
            return i
    return None
    
    
    
def MakeFeatureMatrix(dataList, peakMeans, peakStds, percent = PERCENT, passRatio = PASSRATIO, kMultiplier = K):
    # calculate hit ratio for every mode which fit itself
    matrix = []
    for i in range(MODE):
        tmp = []
        for j in range(MODE):
            hitRatios = []
            for k in range(len(dataList[j])):
                peaks = FindPeaksSorted(dataList[j][k], percent)
                hitRatio = CalculateHitRatio2(peakMeans[i], peakStds[i], peaks, kMultiplier)
                hitRatios.append(hitRatio)
                
           ## print('Put mode %d into model %d to get hitRatio' % (j, i))
            ##print(hitRatios)
            
            #score
            score = 0
            for a in range(len(hitRatios)):
                if hitRatios[a] > passRatio:
                    score += 1.0
            score /= len(hitRatios)            
            tmp.append(score)
        matrix.append(tmp)
    return matrix

def Check(matrix):
    ok = True
    for i in range(MODE):
        for j in range(MODE):
            if i < j and matrix[i][j] >= SUCCESSRATIO:
                ok = False
                break
            if i == j and matrix[i][j] <= SUCCESSRATIO:
                ok = False
                break
    return ok
            
            
if __name__ == '__main__':
    
    trainFileList = ['0328_5_9600_d100_fan0',
                        '0328_5_9600_d100_fan1',
                        '0328_5_9600_d100_fan2',
                        '0328_5_9600_d100_fan3']
    
    figurePrefix = '0328_5_9600_d100'
    
    testFileList = ['0328_2_9600_d100_fan0',
                       '0328_2_9600_d100_fan1',
                       '0328_2_9600_d100_fan2',
                       '0328_2_9600_d100_fan3']
    
    labels = ['fan0',
              'fan1',
              'fan2',
              'fan3']
    
    meanCurves = []
    stdCurves = []

    # preprocess
    trainDataList = []
    testDataList = []
    allTrainData = []
    allTestData = []
    ##DrawEnvelope(meanCurves, stdCurves, labels)
    
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

    """
    we consider larger peaks which occupy top (RATIO)%
    """
    # preprocess peak
    peakMeans = []
    peakStds = []
    for i in range(MODE):
        # find peak
        peaks = FindPeaksSorted(allTrainData[i])
        peakMeans.append(np.mean(peaks))
        peakStds.append(np.std(peaks))
    
    for i in range(MODE):
        print(' mode %d => mean = %.2f, std = %.2f' % (i, peakMeans[i], peakStds[i]))
    ##    DrawEnvelope(peakMeanCurves, peakStdCurves, labels, False) 
    ##    plt.savefig(figurePrefix + ('@ratio=%d' % RATIO))
    """
    for kMultiplier in range(5, 20 + 1, 1):
        for percent in range(10, 50 + 1 , 10):
            for passRatio in range(5, 10 + 1, 1):
                matrix = np.array(MakeFeatureMatrix(trainDataList, peakMeans, peakStds, percent, passRatio / 10.0, kMultiplier * 0.1))
                
                if Check(matrix):
                    print('stdMulti = %.1f, pick top %d percent peaks and %d%% hit ratio to succeed' % (kMultiplier * 0.1, percent, passRatio * 10))
                    print(matrix)
    """
    
    for i in range(MODE):
        print(Predict(allTestData[i],  peakMeans, peakStds))
        # now at mode i
        print('now at mode %d' % i)
        result = []
        for j in range(len(testDataList[i])):
            result.append(Predict(testDataList[i][j], peakMeans, peakStds))
        print(result)                
    # create envelope
    for i in range(MODE):
        meanCurve, stdCurve = CreateCurve(trainDataList[i])
        meanCurves.append(meanCurve)
        stdCurves.append(stdCurve)
    
   ## peakMeanCurves = np.array(peakMeanCurves)
   ## peakStdCurves = np.array(peakStdCurves)
    
    # draw
   ## DrawEnvelope(meanCurves, stdCurves, labels) 
   ## DrawEnvelope(meanCurves, stdCurves, labels, False) 
    
    ## plt.savefig(figurePrefix + ('@pagesize=%d' % pagesize))
        

   ## DrawXYZ(trainingFileList)    
   ## DrawIndepenet(files)
   ## DrawMixed(data, files[0])                   
    plt.show()   
    