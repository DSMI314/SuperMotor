import serial
import sys
import numpy as np

from lib import Parser, PresentationModel, AnalogData
from collections import defaultdict
import operator
import math


def real_time_process():
    # open feature data AND parse them
    p_model = PresentationModel(PresentationModel.TRAINING_MODEL_FILE)

    # plot parameters
    analog_data = AnalogData(Parser.PAGESIZE)
    print('>> Start to receive data...')

    # open serial port
    ser = serial.Serial("COM5", 9600)
    for _ in range(20):
        ser.readline()

    while True:
        try:
            line = ser.readline()

            data = [float(val) for val in line.decode().split(',')]
            if len(data) == 3:
                analog_data.add(data)
                data_list = analog_data.merge_to_list()

                a = []
                for k in range(len(data_list[0])):
                    a.append([data_list[0][k], data_list[1][k], data_list[2][k]])

                real_data = Parser.parse(a)
                gap = np.mean(Parser.find_gaps(real_data))
                p_model.add_to_buffer(gap)

                prediction = p_model.predict()

                p_model.add_to_pool(prediction)

                print(p_model._mean_buffer)
                print('%f => res:%d' % (p_model._now_mean, prediction))

                fp = open(PresentationModel.TARGET_FILE, 'w')
                fp.write(str(p_model.take_result()))
                fp.close()

        except KeyboardInterrupt:
            print('exiting')
            break
        finally:
            # close serial
            ser.flush()
            ser.close()

            # reset file
            fp = open(PresentationModel.TARGET_FILE, 'w')
            fp.write('-1')
            fp.close()


def file_process(fp):
    fpp = open(PresentationModel.TRAINING_MODEL_FILE, 'r')
    means = []
    for token in fpp.readline().split(','):
        means.append(float(token))

    components = []
    for token in fpp.readline()[1:-2].split(' '):
        if len(token) > 0:
            components.append(float(token))
    mu = float(fpp.readline())
    std = float(fpp.readline())
    print(components)
    analog_data = AnalogData(Parser.PAGESIZE)

    print('>> Start to receive data from FILE...')

    hit_count = defaultdict(int)
    tot = 0
    for file_line in fp:
        line = file_line.split(',')
        data = [float(val) for val in line[1:]]

        if len(data) != 3:
            continue
        analog_data.add(data)
        data_list = analog_data.merge_to_list()

        a = []
        for k in range(len(data_list[0])):
            a.append([data_list[0][k], data_list[1][k], data_list[2][k]])

        real_data = Parser.parse(a, means, components)
        gap = np.mean(Parser.find_gaps(real_data))
        multiplier = (gap - mu) / (std * 2)
        plot = int(math.floor(multiplier))
        hit_count[plot] += 1
        tot += 1
        # print(multiplier)
    ratios = []
    for w in sorted(hit_count, key=hit_count.get, reverse=True):
        ratios.append((w, round(hit_count[w] / tot * 100, 2)))
    print(ratios)

def main(argv):
    if len(argv) == 0:
        real_time_process()
    elif len(argv) == 1:
        fp = open(argv[0], 'r')
        file_process(fp)
        fp.close()
    else:
        print('Error: Only accept at most 1 parameter.')
        sys.exit(2)


if __name__ == '__main__':
    main(sys.argv[1:])
