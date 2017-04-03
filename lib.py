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


def FindGaps(data):
    gap = []
    for j in range(3):
        fragment = data[int(PAGESIZE * j / 3): int(PAGESIZE * (j + 1) / 3)]
        peaks = FindPeaksSorted(fragment)
        valleys = FindValleysSorted(fragment)
        if len(peaks) == 0:
            peaks.append(0)
        if len(valleys) == 0:
            valleys.append(0)
        gap.append(np.mean(peaks) - np.mean(valleys))
    return gap

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
    
    for i in range(MODE):
        print(len(trainDataList[i]))
        for k in range(len(trainDataList[i])):
            gap = FindGaps(trainDataList[i][k])  
        
            gapsXList.append([gap[0], gap[1], gap[2]])
            gapsYList.append(i)
    
    
    return gapsXList, gapsYList


def Predict(data, SVM):
    gap = FindGaps(data)
    result = SVM.predict([gap])
    return int(result[0])


def WriteToFile(X, y):
    fpp = open('motorcycle.txt', 'w')

    n = len(X)
    for i in range(n):
        fpp.write(str(X[i][0]) + ',' + str(X[i][1]) + ',' + str(X[i][2]))
        if i < n - 1:
            fpp.write('^')
        else:
            fpp.write('\n')   
            
    for i in range(n):
        fpp.write(str(y[i]))
        if i < n - 1:
            fpp.write(',')
        else:
            fpp.write('\n')    
    fpp.close()    
    