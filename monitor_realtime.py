import serial
import sys

from lib import Model, PresentationModel


def main(argv):
    """
    When the model has built, then load data real-time to predict the state at the moment.

    :param argv:
    argv[0]: client ID
    argv[1]: connect_port_name

    :return:
    """
    _ID = argv[0]
    _PORT_NAME = argv[1]

    model = Model.read_from_file(_ID)
    p_model = PresentationModel.apply(model)

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
                p_model.add(data)

                # is "gap" in K-envelope?
                state = p_model.predict()
                print("OK" if state == 0 else "warning !!!")

                # put result into the target file
                fp = open(p_model.TARGET_FILE, 'w')
                fp.write(str(state))
                fp.close()

        except KeyboardInterrupt:
            print('>> exiting !')
            break
        except IOError:
            continue


if __name__ == '__main__':
    main(sys.argv[1:])
