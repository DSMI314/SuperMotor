import sys
from lib import *


def Run(trainPrefix, testPrefix):
    labels = ['fan0',
              'fan1',
              'fan2',
              'fan3']
    
    trainFileList = []
    testFileList = []
    for i in range(MODE):
        trainFileList.append(trainPrefix + '_' + labels[i])
        testFileList.append(testPrefix + '_' + labels[i])
    
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
    
    peakMeans, peakStds, peakKX, valleyMeans, valleyStds, valleyKX = Train(allTrainData)
    WriteToFile(peakMeans, peakStds, peakKX, valleyMeans, valleyStds, valleyKX)
  
    """
    predict
    """
    for i in range(MODE):
        # now at mode i
        print('now at mode %d' % i)
        result = []
        for j in range(len(testDataList[i])):
            result.append(Predict(testDataList[i][j], peakMeans, peakStds, peakKX, valleyMeans, valleyStds, valleyKX))
        print(result)
        
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
    
if __name__ == '__main__':
    main(sys.argv[1:])