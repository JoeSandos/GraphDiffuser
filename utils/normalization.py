import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from collections import namedtuple

Batch = namedtuple('Batch', 'trajectories conditions')

#-----------------------------------------------------------------------------#
#-------------------------- single-field normalizers -------------------------#
#-----------------------------------------------------------------------------#

class Normalizer:
    '''
        parent class, subclass by defining the `normalize` and `unnormalize` methods
    '''

    def __init__(self, X):
        self.X = X.astype(np.float32)
        self.mins = X.min(axis=0)
        self.maxs = X.max(axis=0)

    def __repr__(self):
        return (
            f'''[ Normalizer ] dim: {self.mins.size}\n    -: '''
            f'''{np.round(self.mins, 2)}\n    +: {np.round(self.maxs, 2)}\n'''
        )

    def __call__(self, x):
        return self.normalize(x)

    def normalize(self, *args, **kwargs):
        raise NotImplementedError()

    def unnormalize(self, *args, **kwargs):
        raise NotImplementedError()

class LimitsNormalizer(Normalizer):
    '''
        maps [ xmin, xmax ] to [ -1, 1 ]
    '''

    def normalize(self, x):
        ## [ 0, 1 ]
        x = (x - self.mins) / (self.maxs - self.mins)
        ## [ -1, 1 ]
        x = 2 * x - 1
        return x

    def unnormalize(self, x, eps=1e-4):
        '''
            x : [ -1, 1 ]
        '''
        # if x.max() > 1 + eps or x.min() < -1 - eps:
        #     x = np.clip(x, -1, 1)

        ## [ -1, 1 ] --> [ 0, 1 ]
        x = (x + 1) / 2.

        return x * (self.maxs - self.mins) + self.mins

class identity_normalizer(Normalizer):
    '''
        identity normalizer
    '''

    def normalize(self, x):
        return x

    def unnormalize(self, x):
        return x

#-----------------------------------------------------------------------------#
#--------------------------- PyTorch Normalizer Adapter ----------------------#
#-----------------------------------------------------------------------------#

class PyTorchNormalizer:
    def __init__(self, normalizer):
        self.normalizer = normalizer

    def normalize(self, tensor):
        if hasattr(tensor,'numpy'):
            array = tensor.numpy()
        else:
            array = tensor
        normalized_array = self.normalizer.normalize(array)
        return torch.tensor(normalized_array, dtype=torch.float32)

    def unnormalize(self, tensor):
        if hasattr(tensor,'numpy'):
            array = tensor.numpy()
        else:
            array = tensor
        unnormalized_array = self.normalizer.unnormalize(array)
        return torch.tensor(unnormalized_array, dtype=torch.float32)
    
def flatten(array):
    if len(array.shape)>=3:
        flattened = np.concatenate([
            x for x in array
            ],axis=0)
    assert len(flattened.shape)==2
    return flattened

#-----------------------------------------------------------------------------#
#------------------------------- TrainData Class -----------------------------#
#-----------------------------------------------------------------------------#



#-----------------------------------------------------------------------------#
#------------------------------- Usage Example -------------------------------#
#-----------------------------------------------------------------------------#

# # Create synthetic data
# U = np.random.rand(100, 10, 3)
# Y_bar = np.random.rand(100, 9, 2)
# Y_f = np.random.rand(100, 2)

# # Initialize the dataset
# train_data = TrainData(U, Y_bar, Y_f, normalizer='LimitsNormalizer')

# # Create a DataLoader
# data_loader = DataLoader(train_data, batch_size=16, shuffle=True)

# # Iterate through the DataLoader
# for batch in data_loader:
#     print(batch.trajectories, batch.conditions)
