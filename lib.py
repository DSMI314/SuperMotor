from parser0 import *
from draw import *
from sklearn.svm import SVC

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


def CalculateHitRatio(mean, std, spotCurve, kMultiplier):
    hitCount = 0
    for i in range(len(spotCurve)):
        if abs(spotCurve[i] - mean) <= kMultiplier * std:
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
        
        nowMax = 0
        nowK = 0
        for k1 in range(len(tmps)):
            if tmps[k1][0] > nowMax:
                nowMax, nowK = tmps[k1][0], tmps[k1][1]
                
        kX.append(nowK)    
    return kX


def Train(trainData, filenamePrefix = ''):
    """
    we consider larger peaks which occupy top (RATIO)%
    """

    # preprocess
    trainDataList = []
    for i in range(MODE):
        trainDataList.append(Paging(trainData[i]))
    
    # preprocess peak and valley for getting gap  
    gapMeans = []
    for i in range(MODE):
        # find peaks and valley
        peaks = FindPeaksSorted(trainData[i])
        valleys = FindValleysSorted(trainData[i])
        
        gapMeans.append(np.mean(peaks) - np.mean(valleys))
    
    # split every file
    gapsXList = []
    gapsYList = []

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')   
    ax.set_xlabel('meanGap1')
    ax.set_ylabel('meanGap2')
    ax.set_zlabel('meanGap3')
    ax.set_title('Scatters of Mean Gaps in 3D (' + filenamePrefix + ')')
    
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
        ax.scatter(gapList[0], gapList[1], gapList[2], label = LABELS[i]) 
        
        gapsXList.extend(list(zip(gapList[0], gapList[1], gapList[2])))
        gapsYList.extend([i] * len(gapList[0]))
        
    ax.legend()
    
    plt.savefig(filenamePrefix +'.png')
    
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
    