import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from collections import deque
import statistics
import sys

from sklearn.svm import SVC
from sklearn import decomposition


class Parser(object):
    """
    Given filename, this static class could parse them into useful information.
    """
    PAGESIZE = 100
    TOP_PEAK_PERCENT = 10
    DATA_FOLDER_PATH = 'recorded_original_data//'
    MEAN_GAP_DIM = 2
    ORIGINAL_ZERO_ADJUST = False

    HARD_COMP = [[-0.91985032, -0.06703556,  0.3864992],
                 [-0.91985032, -0.06703556, 0.3864992]]
    """
    [0.71000981,  0.15992345,  0.68579192], # HOOK
    [-0.17194572, -0.17941319,  0.96863078], # BODY
    [0.84898579,  0.39951426, -0.34584893]] # TOP
    [-0.91985032, -0.06703556,  0.3864992 ] # DRY
    """

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
    def parse(buffer, means=None, components=None):
        """
        Do PCA with some filters; e.g. discard a noise axis.

        :param buffer: n*1 dimension list
        :return: n*1 dimension list
        """
        # for k in range(len(buffer)):
        #     buffer[k][0] = 0.0
        if components is not None:
            res = []
            for i in range(len(buffer)):
                pca = 0
                for k in range(3):
                    pca += (buffer[i][k] - means[k]) * components[k]
                res.append([pca])
            return res
        else:
            records, means, components = Parser.__get_pca(buffer, 1)
            # print(records)
            return records, means, components

    @staticmethod
    def slice(buffer, axis_index):
        res = []
        mean = [0.0] * 3
        for i in range(len(buffer)):
            res.append([buffer[i][axis_index]])
            mean[axis_index] += buffer[i][axis_index]
        mean[axis_index] /= len(buffer)
        components = [0.0] * 3
        components[axis_index] = 1.0
        components = np.array([components])
        return res, mean, components

    @staticmethod
    def sliding(buffer):
        """
        Split data into n-1 segments sequentially where each segment has "_PAGESIZE" size.

        :param buffer: n*1 dimension list
        :return: (n-1) * _PAGESIZE dimension list
        """
        result = []
        for j in range(Parser.PAGESIZE, len(buffer)):
            result.append(buffer[j - Parser.PAGESIZE: j])
        return result

    @staticmethod
    def paging(buffer):
        """
        Split it into several pages indepently which every page size is "_PAGESIZE".

        :param buffer: n*1 dimension list
        :return: ceil(n/_PAGESIZE)*1 dimension list
        """
        result = []
        for j in range(Parser.PAGESIZE, len(buffer), Parser.PAGESIZE):
            result.append(buffer[j - Parser.PAGESIZE: j])
        return result

    @staticmethod
    def find_gaps(data):
        """
        Find gaps for the input data.

        :param data: _PAGESIZE*1 dimension list
        :return: 1*3 dimension list
        """
        gap = []
        for j in range(Parser.MEAN_GAP_DIM):
            fragment = data[int(Parser.PAGESIZE * j / Parser.MEAN_GAP_DIM):
                            int(Parser.PAGESIZE * (j + 1) / Parser.MEAN_GAP_DIM)]
            peaks = Parser.find_peaks_sorted(fragment)
            valleys = Parser.find_valley_sorted(fragment)
            if len(peaks) == 0:
                peaks.append(0)
            if len(valleys) == 0:
                valleys.append(0)
            gap.append(np.mean(peaks) - np.mean(valleys))
        return gap

    @staticmethod
    def __get_pca(records, n):
        pca = decomposition.PCA(n_components=n)
        pca.fit(records)
        records = pca.transform(records)
        print('mean = ' + str(pca.mean_))
        # print coefficient
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
    def do_zero_adjust(records):
        rd = [[], [], []]
        for k in range(len(records)):
            for j in range(3):
                rd[j].append(records[k][j])
        for j in range(3):
            mu = statistics.mean(rd[j])
            for k in range(len(rd[j])):
                rd[j][k] -= mu
        result = []
        for k in range(len(rd[0])):
            result.append([rd[j][k] for j in range(3)])
        return result

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

        valleys.sort()
        valleys = valleys[:int(pagesize * ratio / 100)]
        return valleys

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

        peaks.sort()
        peaks.reverse()
        peaks = peaks[:int(pagesize * ratio / 100)]
        return peaks


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
            if Parser.ORIGINAL_ZERO_ADJUST:
                self._original_data[i] = Parser.do_zero_adjust(self._original_data[i])

        self._raw_data = []
        self._components = []
        self._means = []
        for i in range(self._mode):
            result, mean, comp = Parser.parse(self._original_data[i])
            # result, mean, comp = Parser.slice(self._original_data[i], 2)
            self._raw_data.append(result)
            self._means.append(mean)
            self._components.append(comp)

    def run(self):
        # max_score = 0
        # best_envelope = []
        # xs, ys = None, None
        # for offset in range(Model._FOLD_COUNT):
        #     score, envelope, x, y = self.__validate(offset)
        #     if score > max_score:
        #         max_score, best_envelope, xs, ys = score, envelope, x, y
        # print('optimal mean successful ratios = %.1f%%' % (max_score * 100))
        # PresentationModel.write_to_file(self._mode, self._means, self._components, xs, ys)

        suffix = 'XYZ'
        if Parser.MEAN_GAP_DIM == 3:
            Drawer.plot_3d_scatter(self._raw_data, self._filename, self._labels, suffix)
        else:
            Drawer.plot_2d_scatter_mean_gap(self._raw_data, self._filename, self._labels, suffix)

        # for i in range(self._mode):
        #     Drawer.plot_2d_scatter_origin(self._original_data[i], i, self._filename, self._labels[i])
        # for i in range(self._mode):
        #     Drawer.plot_3d_scatter_origin(self._original_data[i], i, self._filename, self._labels[i])
        # for i in range(self._mode):
        #     Drawer.draw_xyz(self._original_data[i], i, self._filename, self._labels[i], suffix)
        Drawer.draw_line_chart(self._raw_data, self._filename, self._labels, suffix)

    def run2(self, time_interval):
        """
        Consider after PCA

        :param time_interval: time range in seconds
        :return:
        """
        if self._mode > 1:
            print('Error: Only accept at only 1 file.')
            sys.exit(2)
        train_data_list = []
        train_data = []
        for i in range(self._mode):
            train_data.append(self._raw_data[i][:time_interval * Model._SAMPLE_RATE])
            train_data_list.append(Parser.sliding(train_data[i]))
        xs, ys = self.train(train_data_list)
        gaps = []
        for j in range(len(xs)):
            gaps.append(xs[j][0])
        mean = statistics.mean(gaps)
        std = statistics.pstdev(gaps, mean)
        PresentationModel.write_to_file2(self._means, self._components, mean, std)

    def run3(self, time_interval):
        """
        Consider only one axis.

        :param time_interval: time range in seconds
        :return:
        """
        if self._mode > 1:
            print('Error: Only accept at only 1 file.')
            sys.exit(2)
        now_max_gap = 0
        now_max_gap_index = None
        now_raw_data = None
        for axis_index in range(3):
            raw_data, _, _ = Parser.slice(self._original_data[0][:time_interval * Model._SAMPLE_RATE], axis_index)
            raw_data = Parser.sliding(raw_data)
            gaps = []
            for k in range(len(raw_data)):
                gaps.append(Parser.find_gaps(raw_data[k]))
            gap = np.mean(gaps)
            print(gap)
            if gap > now_max_gap:
                now_max_gap, now_max_gap_index, now_raw_data = gap, axis_index, raw_data
        xs, ys = self.train([now_raw_data])
        gaps = []
        for j in range(len(xs)):
            gaps.append(xs[j][0])
        mean = statistics.mean(gaps)
        print(mean)
        std = statistics.pstdev(gaps, mean)
        PresentationModel.write_to_file3(now_max_gap_index, self._means, self._components, mean, std)
        print("!!!!!!!!!!! " + str(now_max_gap_index) + " !!!!!!!!!!!!!!!")

    def train(self, train_data_list):
        xs = []
        ys = []
        # split every file
        for i in range(self._mode):
            for k in range(len(train_data_list[i])):
                gap = Parser.find_gaps(train_data_list[i][k])
                xs.append([np.mean(gap)])
                ys.append(i)
        return xs, ys

    def __validate(self, offset):
        # pre-process
        train_data_list = []
        test_data_list = []
        train_data = []
        test_data = []

        # read file
        for i in range(self._mode):
            cell_size = int(len(self._raw_data[i]) / Model._FOLD_COUNT)

            train_data.append(np.concatenate([self._raw_data[i][:cell_size * offset],
                                              self._raw_data[i][cell_size * (offset + 1):]]))
            test_data.append(self._raw_data[i][cell_size * offset: cell_size * (offset + 1)])

            train_data_list.append(Parser.sliding(train_data[i]))
            test_data_list.append(Parser.sliding(test_data[i]))

        xs, ys = self.train(train_data_list)

        # predict block
        clf = SVC(kernel='linear')
        clf.fit(xs, ys)
        score = []
        envelope = []
        for i in range(self._mode):
            print('now at mode %d' % i)
            result = []
            gaps = []
            res = 0
            for j in range(len(test_data_list[i])):
                gap = np.mean(Parser.find_gaps(test_data_list[i][j]))
                # print(gap)
                gaps.append(gap)
                pd = Model.predict(gap, clf)
                result.append(pd)
                if pd == i:
                    res += 1
            # print(result)
            res /= len(test_data_list[i])
            print('success ratio = %.1f%%\n' % (res * 100))
            score.append(res)
            envelope.append((statistics.mean(gaps), statistics.variance(gaps)))
        return np.mean(score), envelope, xs, ys

    @staticmethod
    def predict(target_gap, clf):
        return int(clf.predict([[target_gap]])[0])


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
        self._mode = int(fp.readline())

        self._components = []
        for token in fp.readline().split(','):
            self._components.append(float(token))

        xs = []
        ys = []
        for token in fp.readline().split(','):
            xs.append([float(token[1: -2])])
        for token in fp.readline().split(','):
            ys.append(int(token))
        self._clf = SVC(kernel='linear')
        self._clf.fit(xs, ys)

        self._pool = deque([-1] * self._pool_size)

        # (mode0, mode1, ..~, modeNone)
        self._pool_count = [0 for _ in range(self._mode)] + [self._pool_size]

        self._mean_buffer = deque([0] * self._buffer_size)

        self._now_mean = 0

        print(xs)

    @staticmethod
    def write_to_file(mode_count, components, xs, ys):
        fp = open(PresentationModel.TRAINING_MODEL_FILE, 'w')
        fp.write(str(mode_count) + '\n')
        PresentationModel.__write_by_line(fp, components)
        PresentationModel.__write_by_line(fp, xs)
        PresentationModel.__write_by_line(fp, ys)
        fp.close()

    @staticmethod
    def write_to_file2(means, components, mean, std):
        fp = open(PresentationModel.TRAINING_MODEL_FILE, 'w')
        for i in range(len(means)):
            PresentationModel.__write_by_line(fp, means[i])
        for i in range(len(components)):
            PresentationModel.__write_by_line(fp, components[i])
        fp.write(str(mean) + '\n')
        fp.write(str(std) + '\n')
        fp.close()

    @staticmethod
    def write_to_file3(index, means, components, mean, std):
        fp = open(PresentationModel.TRAINING_MODEL_FILE, 'w')
        fp.write(str(index) + '\n')
        for i in range(len(means)):
            PresentationModel.__write_by_line(fp, means[i])
        for i in range(len(components)):
            PresentationModel.__write_by_line(fp, components[i])
        fp.write(str(mean) + '\n')
        fp.write(str(std) + '\n')
        fp.close()

    @staticmethod
    def __write_by_line(fp, xs):
        n = len(xs)
        for i in range(n):
            fp.write(str(xs[i]))
            fp.write(',') if i < n - 1 else fp.write('\n')

    def add_to_pool(self, val):
        if len(self._pool) == self._POOL_SIZE:
            x = self._pool.pop()
            self._pool_count[x] -= 1
            self._pool.appendleft(val)
        self._pool_count[val] += 1

    def add_to_buffer(self, val):
        if len(self._mean_buffer) == self._buffer_size:
            x = self._mean_buffer.pop()
            now_mean = (self._now_mean * self._buffer_size - x) / len(self._mean_buffer)
        self._mean_buffer.appendleft(val)
        self._now_mean = (self._now_mean * (len(self._mean_buffer) - 1) + val) / len(self._mean_buffer)

    def take_result(self):
        dic = []
        for i in range(self._mode):
            dic.append([self._pool_count[i], i])
        dic.append([self._pool_count[self._mode], -1])
        return max(dic)[1]

    def predict(self):
        return Model.predict(self._now_mean, self._clf)


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
                gap_list.append(Parser.find_gaps(data_list[i][k]))

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
                gap_list.append(Parser.find_gaps(data_list[i][k]))

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
