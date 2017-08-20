import sys
from lib import Model

from memory_profiler import profile
import timeit


@profile
def main(argv, self_test):
    if len(argv) == 0:
        print('Error: Please give a filename as a parameter')
        sys.exit(2)
    elif len(argv) > 1:
        print('Error: Only accept at most 1 parameter.')
        sys.exit(2)

    filename = argv[0]
    labels = ['on']
    print('>> Processing file \"' + filename + '\"...')

    if self_test:
        print('>> The machine is training (using ENVELOPE)...')
        timer_start = timeit.default_timer()

        model = Model(filename, labels)
        model.run3(60)

        print('>> Completed the training (using ENVELOPE)!')
        timer_end = timeit.default_timer()
    else:
        print('>> The machine is training (using SVM)...')
        timer_start = timeit.default_timer()

        model = Model(filename, labels)
        model.run()

        print('>> Completed the training (using SVM)!')
        timer_end = timeit.default_timer()

    print('spend ' + str(round(timer_end - timer_start, 2)) + ' seconds')


if __name__ == '__main__':

    test_data = ['motor_0504_4Y7M_2_HOOK']
    for data in test_data:
        main([data], True)
    # main(sys.argv[1:])
