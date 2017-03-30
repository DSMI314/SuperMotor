import matplotlib.pyplot as plt
import numpy as np

from sklearn import decomposition
from sklearn import datasets

"""
Given filelist, this class could parse them.
"""
PAGESIZE = 1500
        
def LoadCSV(filename):
    """
    spider from csv which we experiment, then stored them into a list (n*3 dimesion)
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
            
        values = [x, 0, z]
        records.append(values)

    # Discard front data which may be noisy
    n = len(records)
    del records[:int(n/30)]
    
    for i in range(len(records)):
        rec = []
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


def GetPCA(records, n):
    pca = decomposition.PCA(n_components = 1)
    pca.fit(records)
    records = pca.transform(records)
    return records

def Read(file):
    records = np.array(LoadCSV(file))
    return records

def Parse(buffer):
    records = GetPCA(buffer, 1)
    return records

def Paging(buffer, pagesize = PAGESIZE):
    """
    Split it into several pages which every page size is "pagesize".
    """
    result = []        
    for j in range(pagesize, len(buffer), pagesize):
        result.append(buffer[j - pagesize: j])
        
    return result