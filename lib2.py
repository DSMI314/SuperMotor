import bisect
import statistics
import sys
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import decomposition


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


class Model(object):
    """

    """
    def __init__(self, page_size):
        self.page_size = page_size


class SVMModel(Model):
    """

    """
    def __init__(self, page_size, mode_list):
        assert(isinstance(mode_list, list))
        assert(isinstance(mode_list[0], Mode))
        assert(len(mode_list) >= 2)
        super(SVMModel, self).__init__(page_size)
        self.mode_list = mode_list


class PMModel(Model):
    """

    """
    def __init__(self, page_size, mode):
        assert(isinstance(mode, Mode))
        super(PMModel, self).__init__(page_size)
        self.mode = mode
