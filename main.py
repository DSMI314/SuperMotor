import sys, serial
from collections import deque

from lib import *

POOL_SIZE = 40

# class that holds analog data for N samples


class AnalogData:
    # constr
    def __init__(self, maxLen):
        self.ax = deque([0.0]*maxLen)
        self.ay = deque([0.0]*maxLen)
        self.az = deque([0.0]*maxLen)
        self.maxLen = maxLen

    # ring buffer
    def addToBuf(self, buf, val):
        if len(buf) < self.maxLen:
            buf.append(val)
        else:
            buf.pop()
            buf.appendleft(val)

    # add data
    def add(self, data):
        assert(len(data) == 3)
        self.addToBuf(self.ax, data[0])
        self.addToBuf(self.ay, data[1])
        self.addToBuf(self.az, data[2])

    def mergeToList(self):
        tmps=[[], [], []]
        tmps[0]=list(self.ax)
        tmps[1]=list(self.ay)
        tmps[2]=list(self.az)
        return tmps

TRAINING_MODEL_FILE = 'motorcycle.txt'
TARGET_FILE = 'prediction.txt'

def ReadModel():
    fp = open(TRAINING_MODEL_FILE, 'r')
    seperators = []
    for token in fp.readline().split(','):
        seperators.append(float(token))
    
    return seperators

def AddToPool(pool, poolCount, val):
    if len(pool) == POOL_SIZE:
        x = pool.pop()
        poolCount[x] -= 1
    pool.appendleft(val)
    poolCount[val] += 1

def TakeResult(poolCount):
    dic = []
    for i in range(MODE):
        dic.append([poolCount[i], i])
    dic.append([poolCount[MODE], -1])
##    print(dic)
    return max(dic)[1]

# main() function
def main():
    # open feature data AND parse them
    seperators = ReadModel()
              
    # plot parameters
    analogData = AnalogData(PAGESIZE)
    dataList = []
    print('>> Start to receive data...')
##    print(seperators)
    # open serial port
    ser = serial.Serial("COM4", 9600)
    for _ in range(20):
        ser.readline()
        
    pool = deque([-1] * POOL_SIZE)
    poolCount = [0, 0, 0, 0, POOL_SIZE] # (mode0, mode1, mode2, mode3, modeNone)

    while True:
        try:
            line = ser.readline()
            try:
                data = [float(val) for val in line.decode().split(',')]
                if(len(data) == 3):
                    analogData.add(data)
                    dataList = analogData.mergeToList()
                    
                    a = []
                    for k in range(len(dataList[0])):
                        a.append([dataList[0][k], dataList[1][k], dataList[2][k]])
                    realData = Parse(a)
               
                    prediction = Predict(realData, seperators)
                    AddToPool(pool, poolCount, prediction)
                                        
                    fp = open(TARGET_FILE, 'w')
                    fp.write(str(TakeResult(poolCount)))
                    fp.close()

            except:
                pass
        except KeyboardInterrupt:
        
            # reset file
            fp = open(TARGET_FILE, 'w')
            fp.write('-1')
            fp.close()    
            
            print('exiting')
            break
    # close serial
    ser.flush()
    ser.close()
    
# call main
if __name__ == '__main__':
    main()