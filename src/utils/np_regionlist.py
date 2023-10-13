import numpy as np

class NpRegionList:
    def __init__(self, dims):
        self.dims = dims
        self.data = 2*-np.ones((20,) + dims, dtype=np.float64)
        self.capacity = 20
        self.size = 0

    def __len__(self):
        return self.size

    def add(self, x):
        if self.size == self.capacity:
            self.capacity *= 4
            newdata = 2*-np.ones((self.capacity,)+self.dims, dtype=np.float64)
            newdata[:self.size] = self.data
            self.data = newdata

        self.data[self.size] = x
        self.size += 1
