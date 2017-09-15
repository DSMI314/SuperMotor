import bisect
import statistics
import numpy as np

from collections import deque
from abc import abstractmethod
from sklearn import decomposition
from sklearn.svm import SVC


class Parser(object):
    """
    Handle special data type
    """
    @staticmethod
    def write_by_line(fp, line):
        """
        Write line into file with splitting comma.

        :param fp: file source
        :param line: the target line we wanna write into file
        :return:
        """
        n = len(line)
        for i in range(n):
            fp.write(str(line[i]))
            fp.write(',') if i < n - 1 else fp.write('\n')


class Mode(object):
    """
    The unit data frame

    :private attributes:
        x: acceleration at x-axis
        y: acceleration at y-axis
        z: acceleration at z-axis
        time_series: the sequence after combined with x,y,z by PCA
        components: the components when wanna combine x,y,z into time_series
        mean: the mean vector of (x,y,z)
    """
    def __init__(self, x, y, z, components=None):
        assert(len(x) == len(y) and len(y) == len(z))
        self.__x = x
        self.__y = y
        self.__z = z

        self.__time_series = []
        if components is None:
            pca = decomposition.PCA(n_components=1)
            rec = list(zip(self.__x, self.__y, self.__z))
            self.__time_series = pca.fit_transform(rec)
            self.__mean = pca.mean_
            self.__components = pca.components_
        else:
            assert(len(components) == 3)
            assert(isinstance(components[0], float))
            self.__components = components
            self.__mean = [np.mean(x), np.mean(y), np.mean(z)]
            for i in range(len(x)):
                self.__time_series.append(x[i] * components[0] + y[i] * components[1] + z[i] * components[2])

    @property
    def components(self):
        return self.__components

    @property
    def time_series(self):
        return self.__time_series

    @staticmethod
    def read_csv(file_name):
        """
        Read data from specific format .csv file

        :param file_name: filename string "without" extension.
        :return: encrypted as Mode class
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
    The base class which will analyze data into information.

    :protected attributes:
        model_name: model's name
        page_size: the size of a window; a window generates a gap
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
        Read the specific format from .in file to recover the model.

        :param model_name: mode's file name
        :return: the appropriate recovered model
        """
        fp = open(model_name + '.in', 'r')
        model_type = fp.readline().strip()
        page_size = int(fp.readline())
        fp.close()

        model = None
        if model_type == 'SVMModel':
            model = SVMModel(model_name, page_size)

        if model_type == 'PMModel':
            model = PMModel(model_name, page_size)

        assert(model is not None)
        model.read_from_file()
        return model

    def get_gap_time_series(self, mode):
        """
        Get gap curve for the mode by using this model's parameter.

        :param mode: wanna be retrieved
        :return: [gap1, gap2, ...]
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
        Given lists of peak and valley, i.e., the window information , translate them into the feature "gap"

        :param peaks: [peak1, peak2, ...]
        :param valleys: [valley1, valley2, ...]
        :return: gap
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
    The model identifying different modes by using SVM.

    :attr mode_size: the total count of modes
    :attr xs: train_X
    :attr ys: predict_X
    :attr clf: SVM classifier
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
        Given a list of mode we wanna identify, this model will train automatically.

        :param mode_list: [mode1, mode2, ...]
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
        Given a list of mode we wanna identify, this model will do __FOLD_COUNT-fold cross validation.

        :param mode_list: [mode1, mode2, ...]
        :return: [train_X], [predict_X]
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
        Split data into __FOLD_COUNT cells equally, then put "offset"-th cell as test data, otherwise as train data.

        :param offset: #-th as test data
        :param mode_list: [mode1, mode2, ...]
        :return: accuracy, [train_X], [predict_X]
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
        return self.__validate_score(clf, offset, mode_list), xs, ys

    def __validate_score(self, clf, offset, mode_list):
        """
        Given a specific validation method, calculate the performance score.

        :param clf: classifier
        :param offset: #-th as test data
        :param mode_list: [mode1, mode2, ...]
        :return: score
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
        return np.mean(score)

    def save_to_file(self):
        """
        Save the features.

        :return:
        """
        fp = open(self._model_name + '.in', 'w')
        fp.write('SVMModel\n')
        fp.write(str(self._page_size) + '\n')
        Parser.write_by_line(fp, self.__xs)
        Parser.write_by_line(fp, self.__ys)
        fp.close()

    def read_from_file(self):
        """
        Whenever model's name is set, recover the model by reading the feature file.

        :return:
        """
        fp = open(self._model_name + '.in', 'r')

        # discard header
        fp.readline()
        fp.readline()

        # read features
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

    def predict(self, x):
        """
        Return the classification of "x".

        :param x: gap
        :return: prediction
        """
        return self.__clf.predict(x)


class PMModel(Model):
    """
    The model monitoring machine continuously to detect anomaly.

    :private attributes:
        sample_rate: # rows per every second
        cof_k: the coefficient of k
        components: the components applied in translating incoming data rows
        mean: the mean value of the gap series from the mode
        std: the standard deviation of the gap series from the mode
    """
    __PAGE_SIZE = Model._PAGE_SIZE
    __SAMPLE_RATE = 20
    __COF_K = 2

    def __init__(self, model_name, page_size=__PAGE_SIZE, sample_rate=__SAMPLE_RATE, coef_k=__COF_K):
        super(PMModel, self).__init__(model_name, page_size)
        self.__sample_rate = sample_rate
        self.__cof_k = coef_k
        self.__components = None
        self.__mean = None
        self.__std = None

    @property
    def components(self):
        return self.__components

    def fit(self, mode, interval):
        """
        Consider the first "interval" seconds of data from "mode".

        :param mode: normal mode of the machine
        :param interval: retrieve # seconds from beginning as considered
        :return:
        """
        assert(isinstance(mode, Mode))
        assert(isinstance(interval, int))

        x = mode.x[:interval * self.__sample_rate]
        y = mode.y[:interval * self.__sample_rate]
        z = mode.z[:interval * self.__sample_rate]
        capture_mode = Mode(x, y, z)
        self.__components = capture_mode.components
        gap_time_series = self.get_gap_time_series(capture_mode)
        self.__mean = statistics.mean(gap_time_series)
        self.__std = statistics.pstdev(gap_time_series)

    def save_to_file(self):
        """
        Save the features.

        :return:
        """
        fp = open(self._model_name + '.in', 'w')

        # place header
        fp.write('PMModel\n')
        fp.write(str(self._page_size) + '\n')

        # place features
        Parser.write_by_line(fp, self.__components[0])
        fp.write(str(self.__mean) + '\n')
        fp.write(str(self.__std) + '\n')

        fp.close()

    def read_from_file(self):
        """
        Whenever model's name is set, recover the model by reading the feature file.

        :return:
        """
        fp = open(self._model_name + '.in', 'r')

        # discard header
        fp.readline()
        fp.readline()

        # read features
        self.__components = []
        for token in fp.readline().split(','):
            self.__components.append(float(token))

        self.__mean = float(fp.readline())
        self.__std = float(fp.readline())

    def predict(self, x):
        """
        Return the classification of "x".

        :param x: gap
        :return: {0 -> normal, 1 -> anomaly}
        """
        return 0 if abs(x - self.__mean) <= self.__cof_k * self.__std else 1


class PresentationModel(object):
    """
    The intermediate model to maintain I/O.

    :protected attributes:
        model: model's name
        pool_size: the size of pool
        buffer_size: the size of buffer
        cache: reading data
    """
    TARGET_FILE = 'prediction.txt'

    _POOL_SIZE = 20
    _BUFFER_SIZE = 20

    def __init__(self, model, pool_size=_POOL_SIZE, buffer_size=_BUFFER_SIZE):
        self._model = model
        self._pool_size = pool_size
        self._buffer_size = buffer_size
        self._cache = AnalogData(model.page_size)

    @staticmethod
    def apply(model):
        """
        Apply the appropriate model to operate by using factory design pattern.

        :param model: target model
        :return: appropriate model
        """
        if isinstance(model, SVMModel):
            return PresentationSVMModel(model)
        if isinstance(model, PMModel):
            return PresentationPMModel(model)


class PresentationSVMModel(PresentationModel):
    """
    The intermediate SVMModel to maintain I/O.

    :private attributes:
        model: SVMModel
        pool: pool buffer to vote
        pool_count: how many the specific predictions have been output
        mean_buffer: mean of gaps in the cache
        now_mean: mean of gaps in the buffer now
    """
    __POOL_SIZE = PresentationModel._POOL_SIZE
    __BUFFER_SIZE = PresentationModel._BUFFER_SIZE

    def __init__(self, model, pool_size=__POOL_SIZE, buffer_size=__BUFFER_SIZE):
        super(PresentationSVMModel, self).__init__(model, pool_size, buffer_size)

        self.__model = model
        self.__pool = deque([-1] * pool_size)

        # (mode0, mode1, ..~, modeNone)
        self.__pool_count = [0 for _ in range(model.mode_size)] + [pool_size]
        self.__mean_buffer = deque([0] * self._buffer_size)
        self.__now_mean = 0

    @property
    def mean_buffer(self):
        return self.__mean_buffer

    @property
    def now_mean(self):
        return self.__now_mean

    def add_to_pool(self, label):
        """
        Add prediction "label" to pool.

        :param label: prediction
        :return:
        """
        assert(isinstance(label, int))

        if len(self.__pool) == self._POOL_SIZE:
            x = self.__pool.pop()
            self.__pool_count[x] -= 1
            self.__pool.appendleft(label)
        self.__pool_count[label] += 1

    def add_to_buffer(self, data):
        """
        Translate new data (x, y, z) and add it to the cache. Then update the buffer.

        :param data: (x, y, z)
        :return:
        """
        assert(len(data) == 3)
        assert(isinstance(data[0], float))

        self._cache.add(data)
        data_list = self._cache.merge_to_list()
        mode = Mode(data_list[0], data_list[1], data_list[2])
        gaps = Model().get_gap_time_series(mode)
        gap = np.mean(gaps)

        if len(self.__mean_buffer) == self._buffer_size:
            x = self.__mean_buffer.pop()
            self.__now_mean = (self.__now_mean * self._buffer_size - x) / len(self.__mean_buffer)
        self.__mean_buffer.appendleft(gap)
        self.__now_mean = (self.__now_mean * (len(self.__mean_buffer) - 1) + gap) / len(self.__mean_buffer)

    def take_result(self):
        """
        Return the most occurrence of label in the pool.

        :return: label
        """
        dic = []
        for i in range(self.__model.mode_size):
            dic.append([self.__pool_count[i], i])
        dic.append([self.__pool_count[self.__model.mode_size], -1])
        return max(dic)[1]

    def predict(self):
        """
        Return the prediction from the buffer by using this model.

        :return: label
        """
        return int(self.__model.predict(self.__now_mean))


class PresentationPMModel(PresentationModel):
    """
    The intermediate PMModel to maintain I/O.

    :private attributes:
        model: PMModel
        now_gap: mean of gaps in the cache now
    """
    __POOL_SIZE = PresentationModel._POOL_SIZE
    __BUFFER_SIZE = PresentationModel._BUFFER_SIZE

    def __init__(self, model, pool_size=__POOL_SIZE, buffer_size=__BUFFER_SIZE):
        super(PresentationPMModel, self).__init__(model, pool_size, buffer_size)
        self.__model = model
        self.__now_gap = None

    def add(self, data):
        """
        Translate new data (x, y, z) and add it to the cache.

        :param data: (x, y, z)
        :return:
        """
        assert(len(data) == 3)
        assert(isinstance(data[0], float))

        self._cache.add(data)
        data_list = self._cache.merge_to_list()
        mode = Mode(data_list[0], data_list[1], data_list[2], self._model.components)
        gap_time_series = self.__model.get_gap_time_series(mode)
        self.__now_gap = np.mean(gap_time_series)

    def predict(self):
        """
        Return the prediction from the cache by using this model.

        :return: label
        """
        return int(self.__model.predict(self.__now_gap))


class AnalogData(object):
    """
    Hold analog data for "max_len" samples.

    :private attributes:
        ax: the buffer having acceleration at x-axis for "max_len" size
        ay: the buffer having acceleration at y-axis for "max_len" size
        az: the buffer having acceleration at z-axis for "max_len" size
        max_len: the size of the deque structure
    """
    def __init__(self, max_len):
        self.__ax = deque([0.0] * max_len)
        self.__ay = deque([0.0] * max_len)
        self.__az = deque([0.0] * max_len)
        self.__max_len = max_len

    def add(self, data):
        """
        Push data into the buffer.

        :param data: (x, y, z)
        :return:
        """
        assert(len(data) == 3)

        self.__add_to_buf(self.__ax, data[0])
        self.__add_to_buf(self.__ay, data[1])
        self.__add_to_buf(self.__az, data[2])

    def merge_to_list(self):
        return [list(self.__ax), list(self.__ay), list(self.__az)]

    def __add_to_buf(self, buf, val):
        """
        Add "val" to the newest position of deque "buf". If overflow, pop out the oldest position one.

        :param buf: the one-axis buffer
        :param val: original new value
        :return:
        """
        if len(buf) < self.__max_len:
            buf.appendleft(val)
        else:
            buf.pop()
            buf.appendleft(val)
