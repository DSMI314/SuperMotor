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


def main3():
    dfl = []
    for t in range(1, 4+1):
        dfl.append(pd.read_csv('motor_HOOK' + str(t) + '_res.csv'))
    df = dfl[0].append(dfl[1]).append(dfl[2]).append(dfl[3]).reset_index()
    df = df.drop(['index'], axis=1)
    df.to_csv('sss.csv', index=False)

    g = sns.factorplot(x="K", y="false_positive_ratio", hue="recorded_time", col="delta_t", col_wrap=3, data=df)
    g.fig.subplots_adjust(top=0.9)
    g.fig.suptitle('K v.s. false_positive_ratio for choice of K (HOOK)', fontsize=16)
    plt.savefig('HOOK_delta_t.png')
    # plt.show()
    print(df)


def main4():
    dfl = []
    for t in range(1, 4+1):
        dfl.append(pd.read_csv('#DRY' + str(t) + '.csv'))
    df = dfl[0].append(dfl[1]).append(dfl[2]).append(dfl[3]).reset_index()
    df = df.drop(['index'], axis=1)

    g = sns.FacetGrid(df, hue="recorded_time", size=5)
    g = g.map(sns.distplot, "gap_value")
    plt.ylabel('percentage (%)')
    plt.title('gaps distribution depends on recorded time (DRY)')
    plt.legend()
    plt.savefig('tm.png')
    plt.show()


def main5():
    # dfl = []
    # for t in range(1, 4 + 1):
    #     dfl.append(pd.read_csv('motor_HOOK' + str(t) + '_res.csv'))
    # df = dfl[0].append(dfl[1]).append(dfl[2]).append(dfl[3]).reset_index()
    # df = df.drop(['index'], axis=1)
    # df.to_csv('sss.csv', index=False)
    for pos in ['HOOK', 'TOP', 'BODY']:
        for t in [1, 2, 3, 4]:
            df = pd.read_csv('motor_' + pos + str(t) + '_res.csv')
            mat = df.pivot("K", "delta_t", "false_positive_ratio")
            ax = plt.axes()
            g = sns.heatmap(mat, ax=ax, vmin=0, vmax=35)
            sns.plt.suptitle('The false positive ratio depends on \"K\" and \"delta_t\" (' + pos + ' ' + str(t) + 'min)',
                             fontsize=14)
            plt.savefig(pos + '_' + str(t) + 'min_heatmap.png')
            plt.clf()
            # plt.show()
            # print(df)

if __name__ == '__main__':
    main4()

