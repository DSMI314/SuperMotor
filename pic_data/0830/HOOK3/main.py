# import serial
import sys
import numpy as np

from lib import Parser, PresentationModel, AnalogData
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


def real_time_process(argv):
    """
    When the model has built, then load data real-time to predict the state at the moment.

    :param argv:
    argv[0]: client ID
    argv[1]: connect_port_name
    argv[2]: K-envelope's K, 5 is the best.

    :return:
    """
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
"""

def file_process(argv):
    # access file to read model features.
    p_model = PresentationModel(PresentationModel.TRAINING_MODEL_FILE)

    analog_data = AnalogData(Parser.PAGESIZE)

    print('>> Start to receive data from FILE...')

    df = pd.DataFrame(columns={
        'delta_t',
        'K',
        'recorded_time',
        'false_positive_ratio'
    })
    EVENT = 2485
    for c in [1, 5, 10, 15, 20, 25, 30, 35, 40]:
        for K in [10, 15, 20, 25, 30]:
            CONTINOUS_ANOMALY = c
            count = 0
            fp = open(argv[0], 'r')
            K /= 10.0
            line_number = 0
            warning_count = 0
            for file_line in fp:
                line_number += 1
                if line_number > EVENT:
                    break
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

                if line_number >= Parser.PAGESIZE and p_model.predict(gap, K) != 0:
                    count += 1
                    if count >= CONTINOUS_ANOMALY:
                        warning_count += 1
                else:
                    count = 0
            delta_t = c / 20
            rec_time = argv[1]
            e = df.shape[0]
            df.loc[e] = {
                'delta_t': delta_t,
                'recorded_time': rec_time,
                'false_positive_ratio': warning_count / (EVENT - Parser.PAGESIZE) * 100,
                'K': K
            }
            print(delta_t, rec_time, warning_count / (EVENT - Parser.PAGESIZE) * 100, K)
    df = df[['recorded_time', 'delta_t', 'K', 'false_positive_ratio']]
    print(df)
    df.to_csv(argv[0][:-4] + '_res.csv', index=False)

def main(argv):
    if len(argv) == 3:
        real_time_process(argv)
    elif len(argv) == 2:
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
