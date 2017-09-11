import sys
from lib2 import Model, Mode, SVMModel
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from memory_profiler import profile
import timeit
import numpy as np


def main(argv):
    if len(argv) == 0:
        print('Error: Please give a filename as a parameter')
        sys.exit(2)

    file_name = argv[0]
    print('>> Processing file \"' + file_name + '\"...')

    print('>> The machine is training (using SVM)...')
    timer_start = timeit.default_timer()

    mode_list = []
    for suffix in argv[1:]:
        mode = Mode.read_csv(file_name + '_' + suffix)
        mode_list.append(mode)
    model = SVMModel(file_name)
    model.fit(mode_list)
    model.save_to_file()
    print('>> Completed the training (using SVM)!')
    timer_end = timeit.default_timer()

    print('spend ' + str(round(timer_end - timer_start, 2)) + ' seconds')


if __name__ == '__main__':
    # train_data = ['motor_0504_2_4Y7M_HOOK',
    #               'motor_0504_2_4Y7M_TOP']
    # for data in train_data:
    #     main([data])
    main(sys.argv[1:])
