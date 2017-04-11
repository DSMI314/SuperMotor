import sys
from lib import *

def Run(trainPrefix, testPrefix):
    
    trainFileList = []
    testFileList = []
    for i in range(MODE):
        trainFileList.append(trainPrefix + '_' + LABELS[i])
        testFileList.append(testPrefix + '_' + LABELS[i])
    
    # preprocess
    trainDataList = []
    testDataList = []
    allTrainData = []
    allTestData = []
    
    
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
        
    X, Y = train2(allTrainData)
    WriteToFile(X, Y)

    
    """
    predict
    """
    clf = SVC(kernel='poly', degree=1)
    clf.fit(X, Y)
    for i in range(MODE):
        # now at mode i
        print('now at mode %d' % i)
        result = []
        res = 0
        for j in range(len(testDataList[i])):
            gap = np.mean(FindGaps(testDataList[i][j]))
            print(gap)
            pd = Predict(gap, clf)
            result.append(pd)
            if pd == i:
                res += 1
        print(result)
        res /= len(testDataList[i])
        print('success ratio = %.1f%%\n' % (res * 100))
    
def main(argv):
    if len(argv) == 0:
        print('Error: Please give a filename as a parameter')
        sys.exit(2)
    elif len(argv) > 2:
        print('Error: Only accept at most 2 parameters.')
        sys.exit(2)
              
    trainFileName = argv[0]
    testFileName = argv[0]
    if len(argv) == 2:
        testFileName = argv[1]
    
    print('>> The machine is training ...')        
    Run(trainFileName, testFileName)
    print('>> Completed the training!')
    plt.show()
    
if __name__ == '__main__':

 #   testdata = ['0411_2']
 #   for data in testdata:
 #           main([data])
    main(sys.argv[1:])
