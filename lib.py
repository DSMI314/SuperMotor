import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from collections import deque

from sklearn.svm import SVC
from sklearn import decomposition


class Parser(object):
    """
    Given filename, this static class could parse them into useful information.
    """
    PAGESIZE = 100
    TOP_PEAK_PERCENT = 10
    DATA_FOLDER_PATH = 'recorded_original_data//'

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
        for k in range(len(buffer)):
            buffer[k][2] = 0.0
        records = Parser.__get_pca(buffer, 1)
        return records

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
        for j in range(3):
            fragment = data[int(Parser.PAGESIZE * j / 3): int(Parser.PAGESIZE * (j + 1) / 3)]
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
        return records

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
    def find_valley_sorted(xs, ratio=TOP_PEAK_PERCENT):
        valleys = []
        pagesize = len(xs)

        for j in range(1, pagesize - 1):
            now = xs[j]
            prevv = xs[j - 1]
            nextt = xs[j + 1]
            # valley detected
            if now < prevv and now < nextt:
                valleys.extend(now)

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
                peaks.extend(now)

        peaks.sort()
        peaks.reverse()
        peaks = peaks[:int(pagesize * ratio / 100)]
        return peaks


class Model(object):
    """

    """

    _FOLD_COUNT = 5

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
        for i in range(self._mode):
            self._raw_data.append(Parser.parse(self._original_data[i]))

    def run(self):
        """
        max_score = 0
        xs, ys = None, None
        for offset in range(Model._FOLD_COUNT):
            score, x, y = self.__validate(offset)
            if score > max_score:
                max_score, xs, ys = score, x, y
        print('optimal mean successful ratios = %.1f%%' % (max_score * 100))
        PresentationModel.write_to_file(self._mode, xs, ys)
        """
        SUFFIX = 'XY'
        Drawer.plot_scatter(self._raw_data, self._filename, self._labels, SUFFIX)
        # for i in range(self._mode):
        #     Drawer.draw_xyz(self._original_data[i], self._filename, self._labels[i], SUFFIX)
        Drawer.draw_line_chart(self._raw_data, self._filename, self._labels, SUFFIX)

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
        for i in range(self._mode):
            print('now at mode %d' % i)
            result = []
            res = 0
            for j in range(len(test_data_list[i])):
                gap = np.mean(Parser.find_gaps(test_data_list[i][j]))
                print(gap)
                pd = Model.predict(gap, clf)
                result.append(pd)
                if pd == i:
                    res += 1
            print(result)
            res /= len(test_data_list[i])
            print('success ratio = %.1f%%\n' % (res * 100))
            score.append(res)
        return np.mean(score), xs, ys

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
    def write_to_file(mode_count, xs, ys):
        fp = open(PresentationModel.TRAINING_MODEL_FILE, 'w')
        fp.write(str(mode_count) + '\n')
        PresentationModel.__write_by_line(fp, xs)
        PresentationModel.__write_by_line(fp, ys)
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

    @staticmethod
    def plot_scatter(raw_data, title='', labels=[], suffix=''):
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

            ax.scatter(now_list[0], now_list[1], now_list[2], label=labels[i])

        ax.legend()

        plt.savefig(title + '[' + suffix + ']' + '.png')
        # plt.show()

    @staticmethod
    def draw_xyz(raw_data,  filename='', label='', suffix=''):
        x_label = 'time_stamp (s/20)'
        y_label = 'acceleration (mg)'
        title = 'Original Data of X,Y,Z (' + filename + '_' + label + ') [' + suffix + ']'

        fig, ax = plt.subplots(3, sharex='all')

        rd = [[], [], []]
        for k in range(len(raw_data)):
            for j in range(3):
                rd[j].append(raw_data[k][j])

        plt.xlabel(x_label)
        plt.ylabel(y_label)
        ax[0].set_title(title)

        axis_labels = ['X', 'Y', 'Z']
        for i in range(3):
            x = np.arange(0, len(rd[i]))
            y = rd[i]
            ax[i].plot(x, y, label=axis_labels[i])
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

        colors = ['blue', 'orange', 'green']
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
            ax.plot(X, peaks_list, label='peak_' + labels[i], color=colors[i])
            ax.plot(X, valleys_list, '--', label='valley_' + labels[i], color=colors[i])
        ax.legend()
        ax.set_title(title)

        plt.savefig(title + 'line_chart.png')
        # plt.show()
