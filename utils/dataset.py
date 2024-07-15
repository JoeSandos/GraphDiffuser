import torch
from torch.utils.data import Dataset, DataLoader
from utils.normalization import PyTorchNormalizer, LimitsNormalizer, identity_normalizer
from collections import namedtuple

Batch = namedtuple('Batch', 'trajectories conditions')

class TrainData(Dataset):
    def __init__(self, U, Y_bar, Y_f, normalizer='identity_normalizer'):
        super().__init__()
        assert len(U.shape)==3 and len(Y_bar.shape)==3 and len(Y_f.shape)==2 # N, T, m; N, T-1, p; N, p
        U_tensor = torch.tensor(U, dtype=torch.float32)
        U_zeros = torch.zeros(U_tensor.shape[0], 1, U_tensor.shape[2])
        self.U = torch.cat([U_zeros, U_tensor], dim=1) # N, T+1, m
        self.U = self.U.flip(1) # u is originally from T to 0
        Y_bar_tensor = torch.tensor(Y_bar, dtype=torch.float32)
        Y_f_tensor = torch.tensor(Y_f, dtype=torch.float32)
        Y_0_tensor = torch.zeros(Y_bar_tensor.shape[0], 1, Y_bar_tensor.shape[2])
        self.Y = torch.cat([Y_0_tensor, Y_bar_tensor, Y_f_tensor.unsqueeze(1)], dim=1) # N, T+1, p
        self.horizon = self.U.shape[1]

        if isinstance(normalizer, str):
            normalizer = eval(normalizer)

        # Flatten the dataset to apply normalization
        dataset = {
            'U': self.U.view(-1, self.U.shape[-1]).numpy(),
            'Y': self.Y.view(-1, self.Y.shape[-1]).numpy()
        }
        self.normalizer = {
            'U': PyTorchNormalizer(normalizer(dataset['U'])),
            'Y': PyTorchNormalizer(normalizer(dataset['Y']))
        }

        self.U = self.normalizer['U'].normalize(self.U)
        self.Y = self.normalizer['Y'].normalize(self.Y)
        
        
    def __getitem__(self, index):
        u, y = self.U[index], self.Y[index] # T+1, m; T+1, p
        x = torch.cat([u, y], dim=1) # T+1, m+p
        cond = {0: y[0], self.horizon-1: y[-1]}
        # cond = {0: y[0]}
        # cond={}
        return Batch(x, cond)
    def __len__(self):
        return self.U.shape[0]


class TrainData2(Dataset):
    def __init__(self, U, Y_bar, Y_f, horizon=8, normalizer='identity_normalizer'):
        super().__init__()
        assert len(U.shape)==3 and len(Y_bar.shape)==3 and len(Y_f.shape)==2 # N, T, m; N, T-1, p; N, p
        U_tensor = torch.tensor(U, dtype=torch.float32)
        U_zeros = torch.zeros(U_tensor.shape[0], 1, U_tensor.shape[2])
        Us = torch.cat([U_zeros, U_tensor], dim=1) # N, T+1, m
        Us = Us.flip(1) # u is originally from T to 0
        U_cuts = [Us[:, i:i+horizon] for i in range(Us.shape[1]-horizon+1)] # length: T+1-horizon+1
        self.U = torch.cat(U_cuts, dim=0)  # (T+1-horizon+1)*N, horizon, m
        Y_bar_tensor = torch.tensor(Y_bar, dtype=torch.float32)
        Y_f_tensor = torch.tensor(Y_f, dtype=torch.float32)
        Y_0_tensor = torch.zeros(Y_bar_tensor.shape[0], 1, Y_bar_tensor.shape[2])
        Ys = torch.cat([Y_0_tensor, Y_bar_tensor, Y_f_tensor.unsqueeze(1)], dim=1) # N, T+1, p
        Y_cuts = [Ys[:, i:i+horizon] for i in range(Ys.shape[1]-horizon+1)]
        self.Y = torch.cat(Y_cuts, dim=0) # (T+1-horizon+1)*N, horizon, p
        self.horizon = horizon
        self.num=0
        
        if isinstance(normalizer, str):
            normalizer = eval(normalizer)

        # Flatten the dataset to apply normalization
        dataset = {
            'U': self.U.view(-1, self.U.shape[-1]).numpy(),
            'Y': self.Y.view(-1, self.Y.shape[-1]).numpy()
        }
        self.normalizer = {
            'U': PyTorchNormalizer(normalizer(dataset['U'])),
            'Y': PyTorchNormalizer(normalizer(dataset['Y']))
        }

        self.U = self.normalizer['U'].normalize(self.U)
        self.Y = self.normalizer['Y'].normalize(self.Y)
        
    def __getitem__(self, index):
        u, y = self.U[index], self.Y[index] # horizon, m; horizon, p
        x = torch.cat([u, y], dim=1)
        # randomint = torch.randint(low=1, high=self.horizon-1, size=(1,))  # 生成一个0到10之间的随机整数
        cond = {0: y[0], self.horizon-1: y[1]}
        # cond = {0: y[0]}
        # cond={}
        # cond = {0: y[0], self.num%(self.horizon-1): y[self.num%(self.horizon-1)]}
        self.num+=1
        return Batch(x, cond)
    def __len__(self):
        return self.U.shape[0]

class TrainData_norm(Dataset):
    def __init__(self, U, Y_bar, Y_f, normalizer='LimitsNormalizer'):
        super().__init__()
        assert len(U.shape)==3 and len(Y_bar.shape)==3 and len(Y_f.shape)==2 # N, T, m; N, T-1, p; N, p
        U_tensor = torch.tensor(U, dtype=torch.float32)
        U_zeros = torch.zeros(U_tensor.shape[0], 1, U_tensor.shape[2])
        self.U = torch.cat([U_zeros, U_tensor], dim=1)
        self.U = self.U.flip(1) # u is originally from T to 0
        Y_bar_tensor = torch.tensor(Y_bar, dtype=torch.float32)
        Y_f_tensor = torch.tensor(Y_f, dtype=torch.float32)
        Y_0_tensor = torch.zeros(Y_bar_tensor.shape[0], 1, Y_bar_tensor.shape[2])
        self.Y = torch.cat([Y_0_tensor, Y_bar_tensor, Y_f_tensor.unsqueeze(1)], dim=1) # N, T+1, p
        self.horizon = self.U.shape[1]

        # Apply normalization
        if isinstance(normalizer, str):
            normalizer = eval(normalizer)

        # Flatten the dataset to apply normalization
        dataset = {
            'U': self.U.view(-1, self.U.shape[-1]).numpy(),
            'Y': self.Y.view(-1, self.Y.shape[-1]).numpy()
        }
        self.normalizer = {
            'U': PyTorchNormalizer(normalizer(dataset['U'])),
            'Y': PyTorchNormalizer(normalizer(dataset['Y']))
        }

        self.U = self.normalizer['U'].normalize(self.U)
        self.Y = self.normalizer['Y'].normalize(self.Y)

    def __getitem__(self, index):
        u, y = self.U[index], self.Y[index] # T+1, m; T+1, p
        x = torch.cat([u, y], dim=1) # T+1, m+p
        cond = {0: y[0], self.horizon-1: y[-1]}
        # cond = {0: y[0]}
        # cond={}
        return Batch(x, cond)
    
    def __len__(self):
        return self.U.shape[0]

class TrainData_norm2(Dataset):
    def __init__(self, U, Y_bar, Y_f, normalizer='LimitsNormalizer',horizon=8):
        super().__init__()
        assert len(U.shape)==3 and len(Y_bar.shape)==3 and len(Y_f.shape)==2 # N, T, m; N, T-1, p; N, p
        U_tensor = torch.tensor(U, dtype=torch.float32)
        U_zeros = torch.zeros(U_tensor.shape[0], 1, U_tensor.shape[2])
        Us = torch.cat([U_zeros, U_tensor], dim=1) # N, T+1, m
        Us = Us.flip(1) # u is originally from T to 0
        U_cuts = [Us[:, i:i+horizon] for i in range(Us.shape[1]-horizon+1)] # length: T+1-horizon+1
        self.U = torch.cat(U_cuts, dim=0)  # (T+1-horizon+1)*N, horizon, m
        Y_bar_tensor = torch.tensor(Y_bar, dtype=torch.float32)
        Y_f_tensor = torch.tensor(Y_f, dtype=torch.float32)
        Y_0_tensor = torch.zeros(Y_bar_tensor.shape[0], 1, Y_bar_tensor.shape[2])
        Ys = torch.cat([Y_0_tensor, Y_bar_tensor, Y_f_tensor.unsqueeze(1)], dim=1) # N, T+1, p
        Y_cuts = [Ys[:, i:i+horizon] for i in range(Ys.shape[1]-horizon+1)]
        self.Y = torch.cat(Y_cuts, dim=0) # (T+1-horizon+1)*N, horizon, p
        self.horizon = horizon
        self.num=0
        
        # Apply normalization
        if isinstance(normalizer, str):
            normalizer = eval(normalizer)

        # Flatten the dataset to apply normalization
        dataset = {
            'U': self.U.view(-1, self.U.shape[-1]).numpy(),
            'Y': self.Y.view(-1, self.Y.shape[-1]).numpy()
        }
        self.normalizer = {
            'U': PyTorchNormalizer(normalizer(dataset['U'])),
            'Y': PyTorchNormalizer(normalizer(dataset['Y']))
        }

        self.U = self.normalizer['U'].normalize(self.U)
        self.Y = self.normalizer['Y'].normalize(self.Y)

    def __getitem__(self, index):
        u, y = self.U[index], self.Y[index] # T+1, m; T+1, p
        x = torch.cat([u, y], dim=1) # T+1, m+p
        cond = {0: y[0], self.horizon-1: y[-1]}
        # cond = {0: y[0]}
        # cond={}
        return Batch(x, cond)
    
    def __len__(self):
        return self.U.shape[0]