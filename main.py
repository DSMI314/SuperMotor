import serial
import sys
import numpy as np

from lib import Parser, PresentationModel, AnalogData


def real_time_process(argv):
    """
    When the model has built, then load data real-time to predict the state at the moment.

    :param argv:
    argv[0]: client ID
    argv[1]: connect_port_name
    argv[2]: K-envelope's K, 5 is the best.

    :return:
    """
    _BT_NAME = argv[0]
    _PORT_NAME = argv[1]
    _K = int(argv[2])

    # access file to read model features.
    fpp = open(PresentationModel.TRAINING_MODEL_FILE, 'r')
    axis_select = int(fpp.readline())
    mu = float(fpp.readline())
    std = float(fpp.readline())
    fpp.close()

    # plot parameters
    analog_data = AnalogData(Parser.PAGESIZE)
    print('>> Start to receive data...')

    # open serial port
    ser = serial.Serial(_PORT_NAME, 9600)
    for _ in range(20):
        ser.readline()

    while True:
        try:
            # retrieve the line
            line = ser.readline().decode()
            data = [float(val) for val in line.split(',')]

            # no missing column in the data
            if len(data) == 3:
                # calculate mean gap
                analog_data.add(data)
                data_list = analog_data.merge_to_list()
                real_data = data_list[axis_select]
                peak_ave = Parser.find_peaks_sorted(real_data)
                valley_ave = Parser.find_valley_sorted(real_data)
                gap = np.mean(peak_ave) - np.mean(valley_ave)

                state = 0
                # is "gap" in K-envelope?
                if gap < mu - _K * std or gap > mu + _K * std:
                    state = 1
                print("OK" if state == 0 else "warning !!!")

                # put result into the target file
                fp = open(PresentationModel.TARGET_FILE, 'w')
                fp.write(_BT_NAME + '\n' + str(state))
                fp.close()

        except KeyboardInterrupt:
            print('>> exiting !')
            break
        except IOError:
            continue


def main(argv):
    if len(argv) == 3:
        real_time_process(argv)
    else:
        print('Error: Only accept exactly 3 parameters.')
        print()
        print(':param argv:')
        print('argv[0]: client ID')
        print('argv[1]: connect_port_name')
        print('argv[2]: K-envelope\'s K, 5 is the best.')
        print()
        sys.exit(2)


if __name__ == '__main__':
    main(sys.argv[1:])
