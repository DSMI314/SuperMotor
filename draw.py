import matplotlib.pyplot as plt
import numpy as np

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
##    Draw(fig, ax, yPassive, 'others')
    ax.legend()
    ax.set_title('Hit Ratios of %s and others (in model %s)' % (activeLabel, activeLabel))


def DrawHitLineChart2(X, ys, labels, activeLabel):
    fig, ax = plt.subplots()
    plt.xlabel('kMultiplier')
    plt.ylabel('meanHitRatio')
    for j in range(MODE):
        ax.plot(X, ys[j], label = labels[j])
    ax.legend()
    ax.set_title('Hit Ratios (for peaks in model %s)' % (activeLabel))
    
    
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


def DrawScatter3D(X, Y, Z, labels):
    X