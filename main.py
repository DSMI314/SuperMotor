import serial
import sys
import numpy as np

from lib import Parser, PresentationModel, AnalogData
import seaborn as sns
import pandas as pd

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
    p_model = PresentationModel(PresentationModel.TRAINING_MODEL_FILE)

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
                real_data = p_model.pca_combine(data_list)
                peak_ave = Parser.find_peaks_sorted(real_data)
                valley_ave = Parser.find_valley_sorted(real_data)
                gap = np.mean(peak_ave) - np.mean(valley_ave)

                # is "gap" in K-envelope?
                state = p_model.predict(gap, _K)
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


def file_process(argv):
    # access file to read model features.
    p_model = PresentationModel(PresentationModel.TRAINING_MODEL_FILE)

    analog_data = AnalogData(Parser.PAGESIZE)

    print('>> Start to receive data from FILE...')

    CONTINOUS_ANOMALY = 40
    count = 0
    xs = []
    ys = []
    for K in range(10, 50 + 1, 2):
        fp = open(argv[0], 'r')
        detected = 0
        K /= 10.0
        line_number = 0
        for file_line in fp:
            line_number += 1
            line = file_line.split(',')
            data = [float(val) for val in line[1:]]

            if len(data) != 3:
                continue
            analog_data.add(data)
            data_list = analog_data.merge_to_list()
            real_data = p_model.pca_combine(data_list)
            peak_ave = Parser.find_peaks_sorted(real_data)
            valley_ave = Parser.find_valley_sorted(real_data)
            gap = np.mean(peak_ave) - np.mean(valley_ave)

            if line_number > Parser.PAGESIZE and p_model.predict(gap, K) != 0:
                count += 1
                if count > CONTINOUS_ANOMALY:
                    print(">> K = %.1f, %d row WARNING!!! %f" % (K, line_number, gap))
                    xs.append(K)
                    ys.append(line_number)
                    break
            else:
                count = 0
    out = pd.DataFrame(data={
        'K': xs,
        '60s': ys
    })
    out.to_csv('result.csv', index=False)

def main(argv):
    if len(argv) == 3:
        real_time_process(argv)
    elif len(argv) == 1:
        file_process(argv)
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
