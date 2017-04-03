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
        
    # split every file
    gapsList = []
    
    for i in range(MODE):
        gapList = []
        for k in range(len(trainDataList[i])):
            gap = FindGaps(trainDataList[i][k])  
        
            gapList.append(np.mean(gap))
        gapsList.append(np.mean(gapList))
    
    seperators = []
    for i in range(1, MODE):
        seperators.append((gapsList[i - 1] + gapsList[i]) / 2.0)
       
    return seperators


def Predict(data, seperators):
    # preprocess peak
    gap = FindGaps(data)
    target = np.mean(gap)
    for i in range(MODE - 1):
        if target < seperators[i]:
            return i
    return MODE - 1


def WriteByLine(fpp, X):
    n = len(X)
    for i in range(n):
        fpp.write(str(X[i]))
        if i < n - 1:
            fpp.write(',')
        else:
            fpp.write('\n')


def WriteToFile(seperators):
    fpp = open('motorcycle.txt', 'w')
    WriteByLine(fpp, seperators)
    fpp.close()    