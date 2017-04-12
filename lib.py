import matplotlib.pyplot as plt
import numpy as np
from parser0 import *
from mpl_toolkits.mplot3d import Axes3D
from sklearn.svm import SVC

TOP_PEAK_PERCENT = 10

MODE = 4

LABELS = ['fan0',
          'fan1',
          'fan2',
          'fan3']


def FindValleysSorted(X, ratio=TOP_PEAK_PERCENT):
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
    plt.show()


def MyPlot(data, filenamePrefix = ''):
    fig = plt.figure()
    ax = Axes3D(fig)
    # preprocess
    dataList = Paging(data)
    
    ax.set_xlabel('meanGap1')
    ax.set_ylabel('meanGap2')
    ax.set_zlabel('meanGap3')
    ax.set_title('Scatters of Mean Gaps in 3D (' + filenamePrefix + ')') 
    
    gapList = []
    
    for k in range(len(dataList)):
        gap = []
        for j in range(3):
            fragment = dataList[k][int(PAGESIZE * j / 3): int(PAGESIZE * (j + 1) / 3)]
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
    
    ax.scatter(nowList[0], nowList[1], nowList[2], label='fan3')
    
    ax.legend()
            
    plt.savefig(filenamePrefix + '.png')


def train(trainDataList):

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


def train2(trainDataList):

    # split every file
    X = []
    Y = []

    for i in range(MODE):
        for k in range(len(trainDataList[i])):
            gap = FindGaps(trainDataList[i][k])
            X.append([np.mean(gap)])
            Y.append(i)
#    print(X)
    return X, Y


def Predict(target, seperators):
    for i in range(MODE - 1):
        if target < seperators[i]:
            return i
    return MODE - 1


def Predict2(target_gap, clf):
    return int(clf.predict([[target_gap]])[0])


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


def WriteToFile2(X, Y):
    fpp = open('motorcycle.txt', 'w')
    WriteByLine(fpp, X)
    WriteByLine(fpp, Y)
    fpp.close()    