import torch as t
from torch.utils.data import Dataset
import numpy as np


class CQT_Dataset(Dataset):
    def __init__(self, data, mode='train'):
        super().__init__()
        self.data = data
        self.mode = mode
        self.k_max = 8
        self.k_min = 0
        self.F = 128

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        '''
            index: here refers to the time 
                instant T or to a Slice of CQT 
        '''
        tri_tuple = []
        ss = self.data.iloc[index]
        cqt_full = t.tensor(ss)
        # make two slices
        k_1 = np.random.randint(self.k_min, self.k_max)
        k_2 = np.random.randint(self.k_min, self.k_max)
        # return two slices plus the difference for loss function
        tri_tuple.append(np.abs(k_1 - k_2))
        tri_tuple.append(cqt_full[k_1: k_1 + self.F])
        tri_tuple.append(cqt_full[k_2: k_2 + self.F])
        
        return tri_tuple