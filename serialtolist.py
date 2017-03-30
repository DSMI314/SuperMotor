import sys, serial
import numpy as np
from collections import deque

from main import *

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
        tmps=[[],[],[]]
        tmps[0]=list(self.ax)
        tmps[1]=list(self.ay)
        tmps[2]=list(self.az)
    ##    dataList = []
    ##    for i in range(maxLen):
    ##        dataList.append([tmps[0], tmps[1], tmps[2]])
        return tmps

      
      
# main() function
def main():
    # open feature data
    fp = open('motorcycle.txt', 'r')
    peakMeans = fp.readline().split(',')
    peakStds = fp.readline().split(',')
    kX = fp.readline().split(',')
    for i in range(4):
        peakMeans[i] = float(peakMeans[i])
        peakStds[i] = float(peakStds[i])
        kX[i] = float(kX[i])
    PEAKMEANS = peakMeans
    PEAKSTDS = peakStds
    KX = kX       
    print(peakMeans)
    print(peakStds)
    print(kX)
    
    # plot parameters
    analogData = AnalogData(100)
    dataList=[]
    print('start to receive data...')
    
    # open serial port
    ser = serial.Serial("COM4", 9600)
    ser.readline()
    while True:
        try:
            line = ser.readline()
            try:
                data = [float(val) for val in line.decode().split(',')]
                if(len(data) == 3):
                    analogData.add(data)
                    dataList=analogData.mergeToList()
                    print('======')
                    a = []
                    for k in range(len(dataList[0])):
                        a.append([dataList[0][k], dataList[1][k], dataList[2][k]])
                    print(_Predict(a))
                    ##print(_Predict(dataList))
                   ## print("-------")
                   ## print(dataList)
                   ## print("-------")
            except:
                pass
        except KeyboardInterrupt:
            print('exiting')
            break
    # close serial
    ser.flush()
    ser.close()

# call main
if __name__ == '__main__':
    
    main()