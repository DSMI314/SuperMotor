import sys
from lib import Model
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from memory_profiler import profile
import timeit
import numpy as np


@profile
def main(argv):
    if len(argv) == 0:
        print('Error: Please give a filename as a parameter')
        sys.exit(2)
    elif len(argv) > 1:
        print('Error: Only accept at most 1 parameter.')
        sys.exit(2)

    filename = argv[0]
    labels = ['on']
    print('>> Processing file \"' + filename + '\"...')

    print('>> The machine is training (using ENVELOPE)...')
    timer_start = timeit.default_timer()

    model = Model(filename, labels)
    model.run(240)
    print('>> Completed the training (using ENVELOPE)!')
    timer_end = timeit.default_timer()

    print('spend ' + str(round(timer_end - timer_start, 2)) + ' seconds')


if __name__ == '__main__':

    test_data = ['motor_0504_4Y7M_2_BODY']
    for data in test_data:
        main([data])
    # main(sys.argv[1:])
