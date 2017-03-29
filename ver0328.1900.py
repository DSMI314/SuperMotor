import matplotlib.pyplot as plt
import numpy as np

from sklearn import decomposition
from sklearn import datasets
from sklearn.svm import SVC

from parser0 import *

K = 1
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

def CalculateHitRatio(meanCurve, stdCurve, spotCurve):
    hitCount = 0
    for i in range(len(meanCurve)):
        if abs(spotCurve[i] - meanCurve[i]) <= K * stdCurve[i]:
            hitCount += 1
    return float(hitCount) / len(meanCurve)

if __name__ == '__main__':
    
    trainingFileList = ['0328_5_9600_d100_fan0',
                        '0328_5_9600_d100_fan1',
                        '0328_5_9600_d100_fan2',
                        '0328_5_9600_d100_fan3']
    
    figurePrefix = '0328_1_9600_d100'
    
    testingFileList = ['0328_2_9600_d100_fan0',
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
    trainingDataList = [] 
    testingDataList = []
    
   # DrawEnvelope(meanCurves, stdCurves, labels)
    for i in range(MODE):
        trainingData = Paging(Parse(trainingFileList[i], i))
        testingData = Paging(Parse(testingFileList[i], i))
        
        trainingDataList.append(trainingData)
        testingDataList.append(testingData)
    
    trainingDataList2 = []     
    for i in range(MODE):
        trainingData = Paging(Parse(trainingFileList[i], i))
        trainingDataList2.append(trainingData)
    
    DrawEnvelope2(trainingDataList2, labels)

    # create envelope
    for i in range(MODE):
        meanCurve, stdCurve = CreateCurve(trainingDataList[i])
        meanCurves.append(meanCurve)
        stdCurves.append(stdCurve)

    """
    for i in range(MODE):
        ratios = []
        for j in range(len(testingDataList[i])):
            hitRatio = CalculateHitRatio(meanCurves[i], stdCurves[i], testingDataList[i][j])
            ratios.append(hitRatio)
        print(('%d mode => ratios list =' % i) + str(ratios))
    """
    DrawEnvelope(meanCurves, stdCurves, labels) 
    DrawEnvelope(meanCurves, stdCurves, labels, False) 
    
    #plt.savefig(figurePrefix + ('@pagesize=%d' % pagesize))
        
    """    
    factX = [[], [], [], []]
    factY = [[], [], [], []]
    # transform test data into sparse vector for 4 envelopes
    for j in range(MODE):
        for k in range(len(trainingDataList[j])):
            for i in range(MODE):
                vec = BuildSparseVector(meanCurves[i], stdCurves[i], trainingDataList[j][k])
                factX[i].append(vec)
                factY[i].append(j)
    """
    
    """
    # train !
    svms = []   
    for i in range(MODE):  
        clf = SVC(C = 1000)
        clf.fit(factX[i], factY[i])
        svms.append(clf)
    
    predictX = [[], [], [], []]
    predictY = [[], [], [], []]
    

    for j in range(MODE):
        for k in range(len(testingDataList[j])):
            for i in range(MODE):
                vec = BuildSparseVector(meanCurves[i], stdCurves[i], testingDataList[j][k])
                predictX[i].append(vec)
                predictY[i].append(j)

    for i in range(MODE):
        print('Use %d-th SVM to predict j-th mode test data' % i)
        for j in range(MODE):
            print(svms[i].predict(predictX[j]))
        
    for i in range(MODE):
        print('The fact of using %d-th SVM to predict j-th mode test data' % i)
        for j in range(MODE):
            print(predictY[j])    
    """

    DrawXYZ(trainingFileList)    
   # DrawIndepenet(files)
   # DrawMixed(data, files[0])                   
    plt.show()   
    