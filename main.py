import serial
import numpy as np

from lib import Parser, PresentationModel, AnalogData


def main():
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

if __name__ == '__main__':
    main()
