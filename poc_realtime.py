import serial
import sys
import numpy as np

from lib2 import Parser, Model, PresentationModel, AnalogData, Mode


def main(argv):
    """
    When the model has built, then load data real-time to predict the state at the moment.

    :param argv:
    argv[0]: client_ID
    argv[1]: connect_port_name
    :return:
    """
    _ID = argv[0]
    _PORT_NAME = argv[1]

    model = Model.read_from_file(_ID)
    p_model = PresentationModel.apply(model)
    cache = AnalogData(model.page_size)

    print('>> Start to receive data...')

    # open serial port
    ser = serial.Serial(_PORT_NAME, 9600)
    for _ in range(20):
        ser.readline()

    while True:
        try:
            line = ser.readline()
            data = [float(val) for val in line.decode().split(',')]
            if len(data) == 3:
                cache.add(data)
                data_list = cache.merge_to_list()
                mode = Mode(data_list[0], data_list[1], data_list[2])
                gaps = Model().get_gap_time_series(mode)
                gap = np.mean(gaps)

                p_model.add_to_buffer(gap)

                prediction = p_model.predict()
                p_model.add_to_pool(prediction)

                print(p_model.mean_buffer)
                print('%f => res:%d' % (p_model.now_mean, prediction))

                fp = open(p_model.TARGET_FILE, 'w')
                fp.write(str(p_model.take_result()))
                fp.close()

        except KeyboardInterrupt:
            break
    # close serial
    ser.flush()
    ser.close()


if __name__ == '__main__':
    main(sys.argv[1:])
