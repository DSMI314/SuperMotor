import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

from sklearn.svm import SVC
from sklearn import decomposition


class Parser(object):
    """
    Given filename, this static class could parse them into useful information.
    """
    _PAGESIZE = 100
    _TOP_PEAK_PERCENT = 10

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
        # for k in range(len(buffer)):
        #     buffer[k][1] = 0.0
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
        for j in range(Parser._PAGESIZE, len(buffer)):
            result.append(buffer[j - Parser._PAGESIZE: j])
        return result

    @staticmethod
    def paging(buffer):
        """
        Split it into several pages indepently which every page size is "_PAGESIZE".

        :param buffer: n*1 dimension list
        :return: ceil(n/_PAGESIZE)*1 dimension list
        """
        result = []
        for j in range(Parser._PAGESIZE, len(buffer), Parser._PAGESIZE):
            result.append(buffer[j - Parser._PAGESIZE: j])
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
            fragment = data[int(Parser._PAGESIZE * j / 3): int(Parser._PAGESIZE * (j + 1) / 3)]
            peaks = Parser.__find_peaks_sorted(fragment)
            valleys = Parser.__find_valley_sorted(fragment)
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
        fp = open(filename + '.csv', 'r')
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
    def __find_valley_sorted(xs, ratio=_TOP_PEAK_PERCENT):
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
    def __find_peaks_sorted(xs, ratio=_TOP_PEAK_PERCENT):
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
    TRAINING_MODEL_FILE = 'motorcycle.txt'
    TARGET_FILE = 'prediction.txt'

    _MODE = 0
    _LABELS = []
    _FOLD_COUNT = 5

    _original_data = []
    _raw_data = []

    def __init__(self, filename, labels):
        self._LABELS = labels
        self._MODE = len(self._LABELS)

        file_list = []
        for i in range(self._MODE):
            file_list.append(filename + '_' + self._LABELS[i])

        for i in range(self._MODE):
            self._original_data.append(Parser.read(file_list[i]))

        for i in range(self._MODE):
            self._raw_data.append(Parser.parse(self._original_data[i]))

    def run(self):
        max_score = 0
        xs, ys = None, None
        for offset in range(Model._FOLD_COUNT):
            score, x, y = self.__validate(offset)
            if score > max_score:
                max_score, xs, ys = score, x, y

        print('optimal mean successful ratios = %.1f%%' % (max_score * 100))
        Model.write_to_file(xs, ys)

        # PlotScatter(_raw_data)
        # draw_line_chart(_raw_data)

    def train(self, train_data_list):
        xs = []
        ys = []
        # split every file
        for i in range(self._MODE):
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
        for i in range(self._MODE):
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
        for i in range(self._MODE):
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

    @staticmethod
    def write_by_line(fpp, xs):
        n = len(xs)
        for i in range(n):
            fpp.write(str(xs[i]))
            fpp.write(',') if i < n - 1 else fpp.write('\n')

    @staticmethod
    def write_to_file(xs, ys):
        fpp = open(Model.TRAINING_MODEL_FILE, 'w')
        Model.write_by_line(fpp, xs)
        Model.write_by_line(fpp, ys)
        fpp.close()

"""
def PlotScatter(data, filenamePrefix = ''):
    fig = plt.figure()
    ax = Axes3D(fig)
    # preprocess
    dataList = []
    for i in range(MODE):
        dataList.append(Paging(data[i]))

    ax.set_xlabel('meanGap1')
    ax.set_ylabel('meanGap2')
    ax.set_zlabel('meanGap3')
    ax.set_title('Scatters of Mean Gaps in 3D (' + filenamePrefix + ')')    

    for i in range(MODE):
        gapList = []
        for k in range(len(dataList[i])):
            gap = []
            for j in range(3):
                fragment = dataList[i][k][int(PAGESIZE * j / 3): int(PAGESIZE * (j + 1) / 3)]
                peaks = FindPeaksSorted(fragment)
                valleys = FindValleysSorted(fragment)
                if len(peaks) == 0:
                    peaks.append(0)
                if len(valleys) == 0:
                    valleys.append(0)
                gap.append(np.mean(peaks) - np.mean(valleys))
            gapList.append(gap)
            
        nowList = [[], [], []]
        for j in range(len(gapList)):
            for k in range(3):
                nowList[k].append(gapList[j][k])

        ax.scatter(nowList[0], nowList[1], nowList[2], label = LABELS[i])
    
    ax.legend()

    plt.savefig(filenamePrefix +'.png')
    plt.show()



def Draw(fig, ax, X, label0):
    x = np.arange(0, len(X))
    y = X
    ax.plot(x, y, label=label0)


def draw_xyz(data):
    XLABEL = 'time_stamp'
    YLABEL = 'acceleration'
    TITLE = 'Original Data of X,Y,Z'

    fig, ax = plt.subplots(3, sharex=True)

    rd = [[], [], []]
    for k in range(len(data)):
        for j in range(3):
            rd[j].append(data[k][j])

    plt.xlabel(XLABEL)
    plt.ylabel(YLABEL)
    ax[0].set_title(TITLE)

    LL = ['X', 'Y', 'Z']
    for i in range(3):
        Draw(fig, ax[i], rd[i], LL[i])
        ax[i].legend()


def draw_line_chart(data):
    fig, ax = plt.subplots()

    dataList = []
    # preprocess
    for i in range(MODE):
        dataList.append(Paging(data[i]))

    plt.xlabel('time_stamp')
    plt.ylabel('PCA_value')
    
    colors = ['blue', 'orange', 'green']
    for i in range(MODE):
        peaksList = []
        valleysList = []
        for k in range(len(dataList[i])):
            gap = []
            fragment = dataList[i][k]
            peaks = FindPeaksSorted(fragment)
            valleys = FindValleysSorted(fragment)
            if len(peaks) == 0:
                peaks.append(0)
            if len(valleys) == 0:
                valleys.append(0)
            peaksList.append(np.mean(peaks))
            valleysList.append(np.mean(valleys))

        X = np.arange(0, len(peaksList))
        ax.plot(X, peaksList, label='peak_' + LABELS[i], color=colors[i])
        ax.plot(X, valleysList, label='valley_' + LABELS[i], color=colors[i])
    ax.legend()
    ax.set_title('PCA Value (for peaks and valley in model)')

"""