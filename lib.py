import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from collections import deque
import statistics
import sys
import pandas as pd
import seaborn as sns
import bisect
from memory_profiler import profile

from sklearn.svm import SVC
from sklearn import decomposition


class Parser(object):
    """
    Given filename, this static class could parse them into useful information.
    """
    PAGESIZE = 100
    TOP_PEAK_PERCENT = 10
    DATA_FOLDER_PATH = 'recorded_original_data//'
    MEAN_GAP_DIM = 1

    @staticmethod
    def read(filename):
        """
        Load file(.csv) and store them.

        :param filename: filename string "without" extension.
        :return: n*1 dimension list
        """
        records = Parser.__load_csv(filename)
        return np.array(records)

    @staticmethod
    def parse(buffer):
        """
        Do PCA with some filters; e.g. discard a noise axis.

        :param buffer: n*1 dimension list
        :return: n*1 dimension list
        """
        records, means, components = Parser.__get_pca(buffer, 1)
        return records, means, components

    @staticmethod
    def __find_gaps(peaks, valleys):
        pos = int(Parser.PAGESIZE * Parser.TOP_PEAK_PERCENT / 100)
        peak_ave = np.mean(peaks[-pos:])
        valley_ave = np.mean(valleys[:pos])
        return peak_ave - valley_ave

    @staticmethod
    def get_gaps_curve(raw_data):
        """
        Find gaps for the input data.

        :param raw_data:
        :return:
        """
        peaks = []
        valleys = []
        gaps = []
        # process the first window; i.e., the first PAGESIZE rows of data
        for j in range(1, Parser.PAGESIZE):
            if raw_data[j] > raw_data[j - 1] and raw_data[j] > raw_data[j + 1]:
                bisect.insort_left(peaks, raw_data[j], bisect.bisect_left(peaks, raw_data[j]))
            elif raw_data[j] < raw_data[j - 1] and raw_data[j] < raw_data[j + 1]:
                bisect.insort_left(valleys, raw_data[j], bisect.bisect_left(valleys, raw_data[j]))

        gaps.append(Parser.__find_gaps(peaks, valleys))

        # slide from start to end
        for j in range(Parser.PAGESIZE, len(raw_data)):
            s = j - Parser.PAGESIZE + 1
            if raw_data[s] > raw_data[s - 1] and raw_data[s] > raw_data[s + 1]:
                del peaks[bisect.bisect_left(peaks, raw_data[s])]
            elif raw_data[s] < raw_data[s - 1] and raw_data[s] < raw_data[s + 1]:
                del valleys[bisect.bisect_left(valleys, raw_data[s])]

            e = j - 1
            if raw_data[e] > raw_data[e - 1] and raw_data[e] > raw_data[e + 1]:
                bisect.insort_left(peaks, raw_data[e], bisect.bisect_left(peaks, raw_data[e]))
            elif raw_data[e] < raw_data[e - 1] and raw_data[e] < raw_data[e + 1]:
                bisect.insort_left(valleys, raw_data[e], bisect.bisect_left(valleys, raw_data[e]))
            gaps.append(Parser.__find_gaps(peaks, valleys))

        return gaps

    @staticmethod
    def __get_pca(records, n):
        pca = decomposition.PCA(n_components=n)
        pca.fit(records)
        records = pca.transform(records)
        print('mean = ' + str(pca.mean_))
        print('components = ' + str(pca.components_))
        return records, pca.mean_, pca.components_

    @staticmethod
    def __load_csv(filename):
        """
        spider from csv which we experiment, then stored them into a list (n*3 dimension)

        :param filename: filename string "without" extension.
        """
        fp = open(Parser.DATA_FOLDER_PATH + filename + '.csv', 'r')
        records = []
        for line in fp:
            items = line.strip().split(',')
            x, y, z = '0', '0', '0'
            if len(items) > 1:
                x = items[1]
            if len(items) > 2:
                y = items[2]
            if len(items) > 3:
                z = items[3]

            values = [x, y, z]
            records.append(values)

        # Discard some beginning data which may be noisy
        del records[:int(len(records) / 30)]
        n = len(records)

        for i in range(n):
            rec = []
            # Consider X, Y, Z axes
            for k in range(3):
                # If can convert string to float
                try:
                    val = float(records[i][k])
                except ValueError:
                    val = 0
                rec.append(val)

            # Replace it
            records[i] = rec
        return records

    @staticmethod
    def find_peaks_sorted(xs, ratio=TOP_PEAK_PERCENT):
        peaks = []
        pagesize = len(xs)

        for j in range(1, pagesize - 1):
            now = xs[j]
            prevv = xs[j - 1]
            nextt = xs[j + 1]
            # peak detected
            if now > prevv and now > nextt:
                # stored absolute value
                peaks.append(now)
        if len(peaks) == 0:
            peaks.append(0)
        peaks.sort()
        peaks.reverse()
        peaks = peaks[:int(pagesize * ratio / 100)]
        return peaks

    @staticmethod
    def find_valley_sorted(xs, ratio=TOP_PEAK_PERCENT):
        valleys = []
        pagesize = len(xs)

        for j in range(1, pagesize - 1):
            now = xs[j]
            prevv = xs[j - 1]
            nextt = xs[j + 1]
            # valley detected
            if now < prevv and now < nextt:
                valleys.append(now)
        if len(valleys) == 0:
            valleys.append(0)
        valleys.sort()
        valleys = valleys[:int(pagesize * ratio / 100)]
        return valleys


class Model(object):
    """

    """

    _FOLD_COUNT = 5
    _SAMPLE_RATE = 20

    @profile
    def __init__(self, filename, labels):
        self._filename = filename
        self._labels = labels
        self._mode = len(self._labels)

        file_list = []
        for i in range(self._mode):
            file_list.append(filename + '_' + self._labels[i])

        self._original_data = []
        for i in range(self._mode):
            self._original_data.append(Parser.read(file_list[i]))

        self._raw_data = []
        self._components = []
        self._means = []
        for i in range(self._mode):
            result, mean, comp = Parser.parse(self._original_data[i])
            self._raw_data.append(result)
            self._means.append(mean)
            self._components.append(comp)

    def run(self, time_interval):
        """
        PCA

        :return:
        """
        if self._mode > 1:
            print('Error: Only accept at only 1 file.')
            sys.exit(2)
        for i in range(self._mode):
            self._raw_data[i] = self._raw_data[i][:time_interval * Model._SAMPLE_RATE]
        gaps = Parser.get_gaps_curve(self._raw_data[0])
        mean = statistics.mean(gaps)
        std = statistics.pstdev(gaps)
        PresentationModel.write_to_file(self._components, mean, std)


class PresentationModel(object):
    """
    """
    TRAINING_MODEL_FILE = 'motorcycle.txt'
    TARGET_FILE = 'prediction.txt'

    _POOL_SIZE = 20
    _BUFFER_SIZE = 20

    def __init__(self, training_model_file, pool_size=_POOL_SIZE, buffer_size=_BUFFER_SIZE):
        self._pool_size = pool_size
        self._buffer_size = buffer_size

        # read feature to build SVM
        fp = open(training_model_file, 'r')

        self._components = []
        for token in fp.readline().split(','):
            self._components.append(float(token))

        self._mean = float(fp.readline())
        self._std = float(fp.readline())

    @staticmethod
    def write_to_file(components, mean, std):
        fp = open(PresentationModel.TRAINING_MODEL_FILE, 'w')
        for i in range(len(components)):
            PresentationModel.__write_by_line(fp, components[i][0])
        fp.write(str(mean) + '\n')
        fp.write(str(std) + '\n')
        fp.close()

    @staticmethod
    def __write_by_line(fp, xs):
        n = len(xs)
        for i in range(n):
            fp.write(str(xs[i]))
            fp.write(',') if i < n - 1 else fp.write('\n')

    def pca_combine(self, data_list):
        pcas = []
        for i in range(len(data_list[0])):
            pca = 0
            for k in range(3):
                pca += data_list[k][i] * self._components[k]
            pcas.append(pca)
        return pcas

    def predict(self, x, k):
        if abs(x - self._mean) <= k * self._std:
            return 0
        return 1


class AnalogData(object):
    """
    class that holds analog data for N samples
    """
    # con-str
    def __init__(self, max_len):
        self.ax = deque([0.0] * max_len)
        self.ay = deque([0.0] * max_len)
        self.az = deque([0.0] * max_len)
        self.maxLen = max_len

    # ring buffer
    def add_tp_buf(self, buf, val):
        if len(buf) < self.maxLen:
            buf.append(val)
        else:
            buf.pop()
            buf.appendleft(val)

    # add data
    def add(self, data):
        assert(len(data) == 3)
        self.add_tp_buf(self.ax, data[0])
        self.add_tp_buf(self.ay, data[1])
        self.add_tp_buf(self.az, data[2])

    def merge_to_list(self):
        tmps = [[], [], []]
        tmps[0] = list(self.ax)
        tmps[1] = list(self.ay)
        tmps[2] = list(self.az)
        return tmps


class Drawer(object):

    COLORS = ['blue', 'orange', 'green', 'red']

    @staticmethod
    def plot_envelope_prob(mean, std, gaps):
        xs = []
        ys = []
        for K in range(0, 50 + 1, 2):
            K /= 10
            hit = 0
            for j in range(len(gaps)):
                if abs(gaps[j] - mean) <= K * std:
                    hit += 1
            hit_ratio = hit / len(gaps)
            xs.append(K)
            ys.append(hit_ratio)

        df = pd.DataFrame(data={
            'K': xs,
            'hitRatio': ys
        })
        print(df)
        f, ax = plt.subplots(1, 1)
        ax.set_title('Probability of gaps dropping within range(motor_0504_4Y7M_2_HOOK)')
        sns.pointplot('K', 'hitRatio', data=df, title='sss')
        plt.savefig('hitRatio(motor_0504_4Y7M_2_HOOK)240s.png')
        plt.show()

    @staticmethod
    def plot_2d_scatter_origin(raw_data, index, title='', suffix=''):
        # pre-process
        dim = len(raw_data)

        rd = [[], [], []]
        for k in range(dim):
            for j in range(3):
                rd[j].append(raw_data[k][j])

        marks = ['X', 'Y', 'Z']
        for i in range(3):
            for j in range(i + 1, 3):
                fig, ax = plt.subplots()
                x_label = 'acceleration at ' + marks[i] + ' axis (mg)'
                y_label = 'acceleration at ' + marks[j] + ' axis (mg)'
                plt.xlabel(x_label)
                plt.ylabel(y_label)
                ax.set_title('Scatters of Original Data in 2D (' + title + '_' + suffix + ')'
                             + '[' + marks[i] + marks[j] + ']')
                x = rd[i]
                y = rd[j]
                plt.scatter(x, y, label=marks[i] + marks[j], color=Drawer.COLORS[index], alpha=0.2)
                ax.legend()
                plt.savefig(title + '_' + suffix + '[' + marks[i] + marks[j] + ']' + '2d.png')

    @staticmethod
    def plot_2d_scatter_mean_gap(raw_data, title='', labels=[], suffix=''):
        fig, ax = plt.subplots()
        plt.xlabel('meanGap1 (mg)')
        plt.ylabel('meanGap2 (mg)')
        ax.set_title('Scatters of Mean Gaps in 2D (' + title + ')' + '[' + suffix + ']')

        # pre-process
        dim = len(raw_data)
        data_list = []
        for i in range(dim):
            data_list.append(Parser.sliding(raw_data[i]))

        for i in range(dim):
            gap_list = []
            for k in range(len(data_list[i])):
                gap_list.append(Parser.get_gaps_curve(data_list[i][k]))

            now_list = [[], []]
            for j in range(len(gap_list)):
                for k in range(2):
                    now_list[k].append(gap_list[j][k])

            plt.scatter(now_list[0], now_list[1], label=labels[i])

        ax.legend()

        plt.savefig(title + '[' + suffix + ']' + '2D-mean-gap.png')
        # plt.show()

    @staticmethod
    def plot_3d_scatter_origin(raw_data, index, title='', suffix=''):
        fig = plt.figure()
        ax = Axes3D(fig)
        # pre-process
        dim = len(raw_data)
        data_list = []
        for i in range(dim):
            data_list.append(Parser.sliding(raw_data[i]))

        ax.set_xlabel('acceleration at X axis (mg)')
        ax.set_ylabel('acceleration at Y axis (mg)')
        ax.set_zlabel('acceleration at Z axis (mg)')
        ax.set_title('Scatters of Original Data in 3D (' + title + '_' + suffix + ')')

        rd = [[], [], []]
        for k in range(len(raw_data)):
            for j in range(3):
                rd[j].append(raw_data[k][j])

        ax.scatter(rd[0], rd[1], rd[2], color=Drawer.COLORS[index], label='XYZ')

        ax.legend()

        plt.savefig(title + '[' + suffix + ']' + '3D-origin.png')
        # plt.show()

    @staticmethod
    def plot_3d_scatter(raw_data, title='', labels=[], suffix=''):
        fig = plt.figure()
        ax = Axes3D(fig)
        # pre-process
        dim = len(raw_data)
        data_list = []
        for i in range(dim):
            data_list.append(Parser.sliding(raw_data[i]))

        ax.set_xlabel('meanGap1 (mg)')
        ax.set_ylabel('meanGap2 (mg)')
        ax.set_zlabel('meanGap3 (mg)')
        ax.set_title('Scatters of Mean Gaps in 3D (' + title + ')' + '[' + suffix + ']')

        for i in range(dim):
            gap_list = []
            for k in range(len(data_list[i])):
                gap_list.append(Parser.get_gaps_curve(data_list[i][k]))

            now_list = [[], [], []]
            for j in range(len(gap_list)):
                for k in range(3):
                    now_list[k].append(gap_list[j][k])

            ax.scatter(now_list[0], now_list[1], now_list[2], color=Drawer.COLORS[i], label=labels[i])

        ax.legend()
        # plt.show()

    @staticmethod
    def draw_xyz(raw_data, index, filename='', label='', suffix=''):
        x_label = 'time_stamp (s/20)'
        y_label = 'acceleration (mg)'
        title = 'Original Data of X,Y,Z (' + filename + '_' + label + ') [' + suffix + ']'

        fig, ax = plt.subplots(3, sharex='all', sharey='all')

        rd = [[], [], []]
        for k in range(len(raw_data)):
            for j in range(3):
                rd[j].append(raw_data[k][j])

        plt.xlabel(x_label)
        plt.ylabel(y_label)
        # plt.ylim(-50, 50)
        ax[0].set_title(title)

        axis_labels = ['X', 'Y', 'Z']
        for i in range(3):
            x = np.arange(0, len(rd[i]))
            y = rd[i]

            ax[i].plot(x, y, color=Drawer.COLORS[index], label=axis_labels[i])
            ax[i].legend()

        plt.savefig(title + 'xyz.png')
        # plt.show()

    @staticmethod
    def draw_line_chart(raw_data, filename='', labels=[], suffix=''):
        title = 'PCA Value (' + filename + ') [' + suffix + ']'
        fig, ax = plt.subplots()

        data_list = []
        # pre-process
        for i in range(len(raw_data)):
            data_list.append(Parser.sliding(raw_data[i]))

        plt.xlabel('time_stamp (20/s)')
        plt.ylabel('PCA_value (mg)')

        for i in range(len(raw_data)):
            peaks_list = []
            valleys_list = []
            for k in range(len(data_list[i])):
                fragment = data_list[i][k]
                peaks = Parser.find_peaks_sorted(fragment)
                valleys = Parser.find_valley_sorted(fragment)
                if len(peaks) == 0:
                    peaks.append(0)
                if len(valleys) == 0:
                    valleys.append(0)
                peaks_list.append(np.mean(peaks))
                valleys_list.append(np.mean(valleys))

            X = np.arange(0, len(peaks_list))
            ax.plot(X, peaks_list, label='peak_' + labels[i], color=Drawer.COLORS[i])
            ax.plot(X, valleys_list, '--', label='valley_' + labels[i], color=Drawer.COLORS[i])
        ax.legend()
        ax.set_title(title)

        plt.savefig(title + 'line_chart.png')
        # plt.show()
