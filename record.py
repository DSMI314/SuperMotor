import serial
import sys
import time


if len(sys.argv) > 1:
    FILENAME = sys.argv[1]
    out_com4 = open(FILENAME + "_COM4.csv", 'w')
    out_com5 = open(FILENAME + "_COM5.csv", 'w')
    out_com6 = open(FILENAME + "_COM6.csv", 'w')
else:
    exit()

    
if __name__ == '__main__':
    ser_com4=serial.Serial("COM4" , 9600 )
    ser_com5=serial.Serial("COM5" , 9600 )
    ser_com6=serial.Serial("COM6" , 9600 )
    #first raw data may got some problemm, drop it
    for _ in range(20):
        ser_com4.readline()
        ser_com5.readline()
    while True:
        line4 = ser_com4.readline().decode()
        line5 = ser_com5.readline().decode()
        line6 = ser_com6.readline().decode()
        timer = time.strftime("%H:%M:%S", time.localtime())
        if line4 and line5 and line6:
            out_com4.write(timer + "," + line4)
            out_com5.write(timer + "," + line5)
            out_com6.write(timer + "," + line6)
            sys.stdout.write("COM4 >> " + timer + "," + line4 + "COM5 >> " + line5 + "COM6 >> " + line6)


        #print ser.read(1000)