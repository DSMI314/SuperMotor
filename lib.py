import bisect
import statistics
import sys
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
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
        pca, means, components = Parser.__get_pca(buffer, 1)
        return pca, means, components

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
        print('mean = ' + str(pca.mean_))
        print('components = ' + str(pca.components_))
        return pca, pca.mean_, pca.components_

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
        # del records[:int(len(records) / 30)]
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

    def run(self, time_interval):
        """
        PCA

        :return:
        """
        if self._mode > 1:
            print('Error: Only accept at only 1 file.')
            sys.exit(2)

        pca, mean, comp = Parser.parse(self._original_data[0][:time_interval * Model._SAMPLE_RATE])
        self._raw_data.append(pca.transform(self._original_data[0][:time_interval * Model._SAMPLE_RATE]))
        self._means.append(mean)
        self._components.append(comp)

        gaps = Parser.get_gaps_curve(self._raw_data[0])
        mean = statistics.mean(gaps)
        std = statistics.pstdev(gaps)
        print(mean, std)

        PresentationModel.write_to_file(self._components, mean, std)
        return mean, std


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


class Mode(object):
    """

    """
    def __init__(self, x, y, z):
        assert(len(x) == len(y) and len(y) == len(z))
        self.x = x
        self.y = y
        self.z = z
        pca = decomposition.PCA(n_components=1)
        rec = list(zip(self.x, self.y, self.z))
        self.time_series = pca.fit_transform(rec)
        self.mean = pca.mean_
        self.components = pca.components_


