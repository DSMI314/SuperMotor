import sys
from lib import Model, PMModel, Mode
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from memory_profiler import profile
import timeit
import numpy as np


# @profile
def main(argv):
    if len(argv) == 0:
        print('Error: Please give a filename as a parameter')
        sys.exit(2)

    file_name = argv[0]
    mode_name = argv[1]
    print('>> Processing file \"' + file_name + '\"...')

    print('>> The machine is training (using ENVELOPE)...')
    timer_start = timeit.default_timer()

    model = PMModel(file_name)
    mode = Mode.read_csv(mode_name)
    model.fit(mode, 60)
    model.save_to_file()

    print('>> Completed the training (using ENVELOPE)!')
    timer_end = timeit.default_timer()

    print('spend ' + str(round(timer_end - timer_start, 2)) + ' seconds')


if __name__ == '__main__':
    main(sys.argv[1:])
