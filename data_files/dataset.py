import torch as t
from torch.utils.data import Dataset
import numpy as np


class CQT_Dataset(Dataset):
    def __init__(self, data, mode='train'):
        super().__init__()
        self.data = data#.iloc[:, -1:]
        #self.labels = data.iloc[:, :-1]
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
            data: last column of data are f0 labels
        '''
        four_tuple = []
        # get the features
        s_features = self.data.iloc[index]
        s_features = t.tensor(s_features)
        # get cqt slice and label
        cqt_full = s_features[:-1]
        labels = s_features[-1:].squeeze()
        # make two slices
        k_1 = np.random.randint(self.k_min, self.k_max)
        k_2 = np.random.randint(self.k_min, self.k_max)
        # return two slices plus the difference for loss function
        four_tuple.append(np.abs(k_1 - k_2))
        four_tuple.append(cqt_full[k_1: k_1 + self.F])
        four_tuple.append(cqt_full[k_2: k_2 + self.F])
        four_tuple.append(labels)
        
        return four_tuple