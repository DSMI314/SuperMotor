import sys
from lib import *

CELL_COUNT = 5


def MyConcat(list1, list2):
    result = []
    for x in list1:
        result.append(x)
    for y in list2:
        result.append(y)
    return result


def validate(rawData, offset):
    # preprocess
    trainDataList = []
    testDataList = []
    trainData = []
    testData = []

    # read file
    for i in range(MODE):
        cell_size = int(len(rawData[i]) / CELL_COUNT)
        ##
        trainData.append(MyConcat(rawData[i][:cell_size * offset], rawData[i][cell_size * (offset + 1):]))
        testData.append(rawData[i][cell_size * offset: cell_size * (offset + 1)])
        ##
        trainDataList.append(SlidingWindow(trainData[i]))
        testDataList.append(SlidingWindow(testData[i]))

    seperators = train(trainDataList)

    """
    predict
    """

    score = []
    for i in range(MODE):
        # now at mode i
        print('now at mode %d' % i)
        result = []
        res = 0
        for j in range(len(testDataList[i])):
            gap = np.mean(FindGaps(testDataList[i][j]))
            print(gap)
            pd = Predict(gap, seperators)
            result.append(pd)
            if pd == i:
                res += 1
        print(result)
        res /= len(testDataList[i])
        print('success ratio = %.1f%%\n' % (res * 100))
        score.append(res)
    return np.mean(score), seperators


def validate2(rawData, offset):
    # preprocess
    trainDataList = []
    testDataList = []
    trainData = []
    testData = []

    # read file
    for i in range(MODE):
        cell_size = int(len(rawData[i]) / CELL_COUNT)
        ##
        trainData.append(MyConcat(rawData[i][:cell_size * offset], rawData[i][cell_size * (offset + 1):]))
        testData.append(rawData[i][cell_size * offset: cell_size * (offset + 1)])
        ##
        trainDataList.append(SlidingWindow(trainData[i]))
        testDataList.append(SlidingWindow(testData[i]))

    X, Y = train2(trainDataList)

    """
    predict
    """
    clf = SVC(kernel='linear')
    clf.fit(X, Y)
    score = []
    for i in range(MODE):
        # now at mode i
        print('now at mode %d' % i)
        result = []
        res = 0
        for j in range(len(testDataList[i])):
            gap = np.mean(FindGaps(testDataList[i][j]))
            print(gap)
            pd = Predict2(gap, clf)
            result.append(pd)
            if pd == i:
                res += 1
        print(result)
        res /= len(testDataList[i])
        print('success ratio = %.1f%%\n' % (res * 100))
        score.append(res)
    return np.mean(score), X, Y


def Run(namePrefix):
    fileList = []
    for i in range(MODE):
        fileList.append(namePrefix + '_' + LABELS[i])

    rawData = []
    for i in range(MODE):
        rawData.append(Parse(Read(fileList[i])))

    max_score = 0
    seperators = None
    for offset in range(CELL_COUNT):
        score, sep = validate(rawData, offset)
        if score > max_score:
            max_score, seperators = score, sep

    print('optimal mean successful ratios = %.1f%%' % (max_score * 100))
    WriteToFile(seperators)


def Run2(namePrefix):

    fileList = []
    for i in range(MODE):
        fileList.append(namePrefix + '_' + LABELS[i])

    rawData = []
    for i in range(MODE):
        rawData.append(Parse(Read(fileList[i])))

    max_score = 0
    X, Y = None, None
    for offset in range(CELL_COUNT):
        score, x, y = validate2(rawData, offset)
        if score > max_score:
            max_score, X, Y = score, x, y

    print('optimal mean successful ratios = %.1f%%' % (max_score * 100))
    WriteToFile2(X, Y)
    PlotScatter(rawData)


def main(argv):
    if len(argv) == 0:
        print('Error: Please give a filename as a parameter')
        sys.exit(2)
    elif len(argv) > 1:
        print('Error: Only accept at most 1 parameter.')
        sys.exit(2)

    fileName = argv[0]

#    print('>> The machine is training (using GC)...')
#    Run(fileName)
#    print('>> Completed the training (using GC)!')

    print('>> The machine is training (using SVM)...')
    Run2(fileName)
    print('>> Completed the training (using SVM)!')
    plt.show()


if __name__ == '__main__':

    testdata = ['motor_0504_3y_2_COM5']
    for data in testdata:
        main([data])

 #   main(sys.argv[1:])
