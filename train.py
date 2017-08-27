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

    df = pd.DataFrame(columns={
        'recorded_time',
        'mean',
        'std'
    })
    plt.title('gaps distribution depend on recorded time (HOOK)')
    plt.xlabel('gap_value')
    plt.ylabel('percentage (%)')
    for t in range(60, 240+1, 60):
        model = Model(filename, labels)
        mean, std = model.run(t)
        c = df.shape[0]
        df.loc[c] = {
            'recorded_time': int(t/60),
            'mean': mean,
            'std': std
        }
    plt.legend(['1 min', '2 min', '3 min', '4 min'])
    plt.savefig('HOOK' + '.png')
    df = df[['recorded_time', 'mean', 'std']]
    df.to_csv('HOOK.csv', index=False)
    print('>> Completed the training (using ENVELOPE)!')
    timer_end = timeit.default_timer()

    print('spend ' + str(round(timer_end - timer_start, 2)) + ' seconds')


if __name__ == '__main__':

    test_data = ['motor_0504_4Y7M_2_HOOK']
    for data in test_data:
        main([data])
    # main(sys.argv[1:])
