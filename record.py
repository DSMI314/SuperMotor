import serial
import sys
import time


if __name__ == '__main__':
    if len(sys.argv) > 1:
        FILENAME = sys.argv[1]
        # out_com4 = open(FILENAME + "_COM4.csv", 'w')
        out_com3 = open(FILENAME + "_COM3.csv", 'w')
        # out_com6 = open(FILENAME + "_COM6.csv", 'w')
    else:
        exit()

    # ser_com4 = serial.Serial("COM4", 9600)
    ser_com3 = serial.Serial("COM3", 9600)
    # ser_com6 = serial.Serial("COM6", 9600)
    # first raw data may got some problem, drop it
    for _ in range(20):
        # ser_com4.readline()
        ser_com3.readline()
    while True:
        # line4 = ser_com4.readline().decode()
        line3 = ser_com3.readline().decode()
        # line6 = ser_com6.readline().decode()
        timer = time.strftime("%H:%M:%S", time.localtime())
        if line3:
            # out_com4.write(timer + "," + line4)
            out_com3.write(timer + "," + line3)
            # out_com6.write(timer + "," + line6)
            sys.stdout.write(timer + "," + line3)

            # sys.stdout.write("COM4 >> " + timer + "," + line4 + "COM5 >> " + line5 + "COM6 >> " + line6)

        # print ser.read(1000)
