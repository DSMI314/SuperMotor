import bisect
import statistics
import sys
from collections import deque
from abc import abstractmethod

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import decomposition
from sklearn.svm import SVC


class Parser(object):
    """

    """
    @staticmethod
    def write_by_line(fp, line):
        """

        :param fp:
        :param line:
        :return:
        """
        n = len(line)
        for i in range(n):
            fp.write(str(line[i]))
            fp.write(',') if i < n - 1 else fp.write('\n')


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

    @staticmethod
    def read_csv(file_name):
        """

        :param file_name: filename string "without" extension.
        """
        fp = open(file_name + '.csv', 'r')

        xs, ys, zs = [], [], []

        for line in fp:
            items = line.strip().split(',')
            # discard every row containing missing data
            if len(items) <= 3:
                continue

            x, y, z = items[1], items[2], items[3]

            # discard every row having error
            try:
                x, y, z = float(x), float(y), float(z)
            except ValueError:
                continue

            # preserve original data
            xs.append(x)
            ys.append(y)
            zs.append(z)

        return Mode(xs, ys, zs)


class Model(object):
    """

    """
    _PAGE_SIZE = 100

    def __init__(self, model_name=None, page_size=_PAGE_SIZE):
        self._model_name = model_name
        if model_name is None:
            self._model_name = ""
        self._page_size = page_size

    @property
    def page_size(self):
        return self._page_size

    @abstractmethod
    def save_to_file(self):
        raise NotImplementedError("Please implement method \'save_to_file()\'.")

    @abstractmethod
    def predict(self, x):
        raise NotImplementedError("Please implement method \'predict(x)\'.")

    @staticmethod
    def read_from_file(model_name):
        """

        :param model_name:
        :return:
        """
        fp = open(model_name + '.in', 'r')
        model_type = fp.readline().strip()
        page_size = int(fp.readline())
        fp.close()

        if model_type == 'SVMModel':
            model = SVMModel(model_name, page_size)
            model.read_from_file()
            return model

        if model_type == 'PMModel':
            pass

    def get_gap_time_series(self, mode):
        """
        Find gaps for the input data.

        :param mode:
        :return:
        """
        assert(isinstance(mode, Mode))
        raw_data = mode.time_series

        peaks = []
        valleys = []
        gaps = []

        # process the first window; i.e., the first PAGESIZE rows of data
        for j in range(1, self._page_size - 1):
            if raw_data[j] > raw_data[j - 1] and raw_data[j] > raw_data[j + 1]:
                bisect.insort_left(peaks, raw_data[j], bisect.bisect_left(peaks, raw_data[j]))
            elif raw_data[j] < raw_data[j - 1] and raw_data[j] < raw_data[j + 1]:
                bisect.insort_left(valleys, raw_data[j], bisect.bisect_left(valleys, raw_data[j]))
        gaps.append(self.__find_gaps(peaks, valleys))

        # slide from start to end
        for j in range(self._page_size, len(raw_data)):
            s = j - self._page_size + 1
            if raw_data[s] > raw_data[s - 1] and raw_data[s] > raw_data[s + 1]:
                del peaks[bisect.bisect_left(peaks, raw_data[s])]
            elif raw_data[s] < raw_data[s - 1] and raw_data[s] < raw_data[s + 1]:
                del valleys[bisect.bisect_left(valleys, raw_data[s])]

            e = j - 1
            if raw_data[e] > raw_data[e - 1] and raw_data[e] > raw_data[e + 1]:
                bisect.insort_left(peaks, raw_data[e], bisect.bisect_left(peaks, raw_data[e]))
            elif raw_data[e] < raw_data[e - 1] and raw_data[e] < raw_data[e + 1]:
                bisect.insort_left(valleys, raw_data[e], bisect.bisect_left(valleys, raw_data[e]))
            gaps.append(self.__find_gaps(peaks, valleys))

        assert(len(gaps) > 0)
        return gaps

    def __find_gaps(self, peaks, valleys):
        """

        :param peaks:
        :param valleys:
        :return:
        """
        if len(peaks) == 0:
            peaks = [0]
        if len(valleys) == 0:
            valleys = [0]
        pos = int(self._page_size * 10.0 / 100.0)
        peak_ave = np.mean(peaks[-pos:])
        valley_ave = np.mean(valleys[:pos])
        return peak_ave - valley_ave


class SVMModel(Model):
    """

    """
    __FOLD_COUNT = 5
    __PAGE_SIZE = Model._PAGE_SIZE

    def __init__(self, model_name, page_size=__PAGE_SIZE, fold_count=__FOLD_COUNT):
        super(SVMModel, self).__init__(model_name, page_size)
        self.__FOLD_COUNT = fold_count

        self.__mode_size = 0

        self.__xs, self.__ys = None, None
        self.__clf = SVC(kernel='linear')

    @property
    def mode_size(self):
        return self.__mode_size

    def fit(self, mode_list):
        """

        :param mode_list:
        :return:
        """
        assert(isinstance(mode_list, list))
        assert(isinstance(mode_list[0], Mode))
        assert(len(mode_list) >= 2)
        self.__xs, self.__ys = self.__build(mode_list)
        self.__mode_size = len(mode_list)
        self.__clf.fit(self.__xs, self.__ys)

    def __build(self, mode_list):
        """

        :param mode_list:
        :return:
        """
        max_score = 0
        xs, ys = None, None
        for offset in range(self.__FOLD_COUNT):
            score, xs, ys = self.__validate(offset, mode_list)
            if score > max_score:
                max_score, xs, ys = score, xs, ys

        print('optimal mean successful ratios = %.1f%%' % (max_score * 100))
        return xs, ys

    def __validate(self, offset, mode_list):
        """

        :param offset:
        :param mode_list:
        :return:
        """
        # pre-process
        xs = []
        ys = []
        # read file
        for i in range(len(mode_list)):
            mode = mode_list[i]
            raw_data = Model().get_gap_time_series(mode)
            cell_size = int(len(raw_data) / self.__FOLD_COUNT)
            gap_time_series = raw_data[:cell_size * offset] + raw_data[cell_size * (offset + 1):]
            for gap in gap_time_series:
                xs.append([gap])
                ys.append(i)

        clf = SVC(kernel='linear')
        clf.fit(xs, ys)

        """
        predict
        """

        score = []
        for i in range(len(mode_list)):
            mode = mode_list[i]
            raw_data = Model().get_gap_time_series(mode)
            cell_size = int(len(raw_data) / self.__FOLD_COUNT)

            # now at mode i
            print('now at mode %d' % i)
            gap_time_series = raw_data[cell_size * offset: cell_size * (offset + 1)]
            result = []
            hit = 0
            for gap in gap_time_series:
                y = clf.predict([[gap]])[0]
                result.append(y)
                if y == i:
                    hit += 1
            print(result)
            hit_ratio = hit / len(gap_time_series)
            print('success ratio = %.1f%%\n' % (hit_ratio * 100))
            score.append(hit_ratio)
        return np.mean(score), xs, ys

    def save_to_file(self):
        """

        :return:
        """
        fp = open(self._model_name + '.in', 'w')
        fp.write('SVMModel\n')
        fp.write(str(self._page_size) + '\n')
        Parser.write_by_line(fp, self.__xs)
        Parser.write_by_line(fp, self.__ys)
        fp.close()

    def predict(self, x):
        """

        :return:
        """
        return self.__clf.predict(x)

    def read_from_file(self):
        """

        :return:
        """
        fp = open(self._model_name + '.in', 'r')

        # discard header
        fp.readline()
        fp.readline()

        xs = []
        ys = []
        for token in fp.readline().split(','):
            xs.append([float(token[1:-2])])
        for token in fp.readline().split(','):
            ys.append(int(token))

        fp.close()

        self.__xs, self.__ys = xs, ys
        self.__clf.fit(xs, ys)
        self.__mode_size = len(np.unique(ys))


class PMModel(Model):
    """

    """
    def __init__(self, model_name, page_size, mode):
        assert(isinstance(mode, Mode))
        super(PMModel, self).__init__(model_name, page_size)
        self._mode = mode

    def save_to_file(self):
        pass

    def predict(self, x):
        pass


class PresentationModel(object):
    """

    """
    TARGET_FILE = 'prediction.txt'

    _POOL_SIZE = 20
    _BUFFER_SIZE = 20

    def __init__(self, model, pool_size=_POOL_SIZE, buffer_size=_BUFFER_SIZE):
        self._model = model
        self._pool_size = pool_size
        self._buffer_size = buffer_size

    @staticmethod
    def apply(model):
        if isinstance(model, SVMModel):
            return PresentationSVMModel(model)


class PresentationSVMModel(PresentationModel):
    """

    """
    __POOL_SIZE = PresentationModel._POOL_SIZE
    __BUFFER_SIZE = PresentationModel._BUFFER_SIZE

    def __init__(self, model, pool_size=__POOL_SIZE, buffer_size=__BUFFER_SIZE):
        super(PresentationSVMModel, self).__init__(model, pool_size, buffer_size)

        self.__model = model
        self.__mode_size = model.mode_size
        self.__pool = deque([-1] * pool_size)

        # (mode0, mode1, ..~, modeNone)
        self.__pool_count = [0 for _ in range(self.__mode_size)] + [pool_size]
        self.__mean_buffer = deque([0] * self._buffer_size)
        self.__now_mean = 0

    @property
    def mean_buffer(self):
        return self.__mean_buffer

    @property
    def now_mean(self):
        return self.__now_mean

    def add_to_pool(self, val):
        """

        :param val:
        :return:
        """
        if len(self.__pool) == self._POOL_SIZE:
            x = self.__pool.pop()
            self.__pool_count[x] -= 1
            self.__pool.appendleft(val)
        self.__pool_count[val] += 1

    def add_to_buffer(self, val):
        """

        :param val:
        :return:
        """
        if len(self.__mean_buffer) == self._buffer_size:
            x = self.__mean_buffer.pop()
            self.__now_mean = (self.__now_mean * self._buffer_size - x) / len(self.__mean_buffer)
        self.__mean_buffer.appendleft(val)
        self.__now_mean = (self.__now_mean * (len(self.__mean_buffer) - 1) + val) / len(self.__mean_buffer)

    def take_result(self):
        """

        :return:
        """
        dic = []
        for i in range(self.__mode_size):
            dic.append([self.__pool_count[i], i])
        dic.append([self.__pool_count[self.__mode_size], -1])
        return max(dic)[1]

    def predict(self):
        return int(self.__model.predict(self.__now_mean))


class PresentationPMModel(PresentationModel):
    """

    """
    def __init__(self):
        pass


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
    def add_to_buf(self, buf, val):
        if len(buf) < self.maxLen:
            buf.append(val)
        else:
            buf.pop()
            buf.appendleft(val)

    # add data
    def add(self, data):
        assert(len(data) == 3)
        self.add_to_buf(self.ax, data[0])
        self.add_to_buf(self.ay, data[1])
        self.add_to_buf(self.az, data[2])

    def merge_to_list(self):
        tmps = [[], [], []]
        tmps[0] = list(self.ax)
        tmps[1] = list(self.ay)
        tmps[2] = list(self.az)
        return tmps