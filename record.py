import serial
import sys
import time


def main(argv):
    """
    After charging, then record sensor data sequentially.

    :param argv:
    argv[0]: output_file_name
    argv[1]: connect_port_name

    :return:
    """
    _FILENAME = argv[0]
    _PORT_NAME = argv[1]

    # write data into this file.
    fp_out = open(_FILENAME + ".csv", 'w')

    # open the port which we entered.
    ser = serial.Serial(_PORT_NAME, 9600)

    # first some raw data may got some problem, drop it.
    for _ in range(20):
        ser.readline()

    while True:
        line = ser.readline().decode()

        # retrieve now time
        timer = time.strftime("%H:%M:%S", time.localtime())

        # "line" has contents.
        if line:
            fp_out.write(timer + "," + line)

            # print at the terminal to debug
            sys.stdout.write(timer + "," + line)


if __name__ == '__main__':
    main(sys.argv[1:])
