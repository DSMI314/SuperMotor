import serial
import sys
import time


if len(sys.argv) > 1:
    FILENAME = sys.argv[1]
    out = open(FILENAME, 'w')
else:
    exit()

    
if __name__ == '__main__':
    ser=serial.Serial("COM4" , 9600 )
    #first raw data may got some problemm, drop it
    for _ in range(20):
        ser.readline()
    while True:
        line = ser.readline().decode()
        timer = time.strftime("%H:%M:%S", time.localtime())
        if line:
            out.write(timer + "," + line)
            sys.stdout.write(timer + "," + line)
        #print ser.read(1000)