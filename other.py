import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from collections import deque
import statistics
import sys
import seaborn as sns
import pandas as pd
from sklearn.svm import SVC
from sklearn import decomposition


def space_gg(x, a_list, b_list):
    title = 'Memory cost for building the model'
    fig, ax = plt.subplots()

    plt.xlabel('Record time length (second)')
    plt.ylabel('Space needed (MegaBytes)')

    ax.plot(x, a_list, label='storage_memory')
    ax.plot(x, b_list, label='running_memory')
    ax.legend()
    ax.set_title(title)

    plt.savefig(title + 'line_chart.png')
    # plt.show()


def time_gg(x, time_list):
    title = 'Time cost for building the model'
    fig, ax = plt.subplots()

    plt.xlabel('Record time length (second)')
    plt.ylabel('Time needed (second)')

    ax.plot(x, time_list, label='time')
    ax.legend()
    ax.set_title(title)

    plt.savefig(title + 'line_chart.png')
    # plt.show()


def main():
    fp = open('exp.txt', 'r')
    x = []
    y = []
    z1 = []
    z2 = []
    for line in fp:
        items = line.strip().split(' ')
        if len(items) < 4:
            continue
        print(items)
        x.append(int(items[0]))
        y.append(float(items[1]))
        z1.append(float(items[2]))
        z2.append(float(items[3]))
    time_gg(x, y)
    space_gg(x, z1, z2)


def main2():
    inn = pd.read_csv('result - Copy4.csv')
    print(inn)

    f, ax = plt.subplots(1, 1)

    ax.plot(inn.K, inn['60s'], color="blue", label="1min")
    ax.plot(inn.K, inn['120s'], color="red", label="2min")
    ax.plot(inn.K, inn['180s'], color="green", label="3min")
    ax.plot(inn.K, inn['240s'], color="orange", label="4min")
    ax.legend()
    plt.xlabel('K')
    plt.ylabel('detected time stamp (20/s)')
    ax.set_title('The time when anomaly is detected')
    plt.show()


if __name__ == '__main__':
    main2()

