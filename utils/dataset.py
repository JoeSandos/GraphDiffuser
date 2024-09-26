import torch
from torch.utils.data import Dataset, DataLoader
from utils.normalization import PyTorchNormalizer, LimitsNormalizer, identity_normalizer
from collections import namedtuple
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from joblib import Parallel, delayed
import pdb
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
Batch = namedtuple('Batch', 'trajectories conditions')
Batch2 = namedtuple('Batch2', 'trajectories conditions denoiser_conditions')


def clustering(tensor, type='DBSCAN', eps=0.5, min_samples=5):
    if type == 'DBSCAN':
        clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean').fit(tensor)
        labels = clustering.labels_
        assert -1 not in labels, 'DBSCAN failed to cluster the data'
        num_clusters = len(set(labels)) 
        print(f"确定的聚类数: {num_clusters}")
        print("聚类标签:", labels)
        return labels
    elif type == 'KMeans':
        # use silhouette_score
        silhouette_avg = -1
        best_k = 2
        best_cluster = None
        len_tensor = len(tensor)
        def compute_kmeans(k, tensor):
            clustering = KMeans(n_clusters=k, n_init=10, random_state=42).fit(tensor)
            labels = clustering.labels_
            silhouette_avg_new = silhouette_score(tensor, labels)
            return (k, silhouette_avg_new, clustering)

        # results = Parallel(n_jobs=10)(delayed(compute_kmeans)(k, tensor) for k in range(200, 2000, 200))
        print('range:', len_tensor//100,len_tensor//20+1,len_tensor//100)
        results = [compute_kmeans(k, tensor) for k in range(len_tensor//100,len_tensor//20+1,len_tensor//100)]
        # pdb.set_trace()
        # 寻找最佳的 k
        best_k, silhouette_avg, best_cluster = max(results, key=lambda x: x[1])

        # 最终的最佳聚类结果
        clustering = best_cluster
        labels = clustering.labels_
        assert -1 not in labels, 'KMeans failed to cluster the data'
        num_clusters = len(set(labels)) 
        print(f"num clusters: {num_clusters}")
        # count number of samples in each cluster using one line
        # numofdata = [sum(labels==i) for i in range(num_clusters)]
        
        # pca down to 2D and plot

        # pca = PCA(n_components=2)
        # pca_result = pca.fit_transform(tensor)
        # plt.figure(figsize=(10, 10))
        # plt.scatter(pca_result[:, 0], pca_result[:, 1], c=labels)
        # plt.colorbar()
        # plt.savefig('kmeans.png')
        
        return labels
        

    else:
        raise NotImplementedError('Clustering type not implemented')

def cal_smoothness(tensor, order=2, universal=True):
    """
    tensor: N, T, p
    
    """
    if order == 1:
        # diffs = (tensor[:, 1:] - tensor[:, :-1])/(tensor[:,-1:]-tensor[:,0:1]+1e-6) # normalize
        diffs = (tensor[:, 1:] - tensor[:, :-1]) - (tensor[:,-1:]-tensor[:,0:1])/(tensor.shape[1]-1) # normalize
        # add zero to the first element
        diffs = torch.cat([torch.zeros_like(diffs[:, 0:1]), diffs], dim=1)
        if universal:
            smoothness = torch.mean(torch.norm(diffs, p=2,dim=-1), dim=-1) # N
        else:
            smoothness = torch.norm(diffs, p=2,dim=-1) # N, T
    elif order == 2:
        # diffs = (tensor[:, 2:] - 2*tensor[:, 1:-1] + tensor[:, :-2])/(tensor[:,-1:]+tensor[:,0:1]-2*tensor[:,tensor.shape[1]//2:tensor.shape[1]//2+1]+1e-6) # normalize
        diffs = (tensor[:, 2:] - 2*tensor[:, 1:-1] + tensor[:, :-2])
        # add zero to the first two elements
        diffs = torch.cat([torch.zeros_like(diffs[:, 0:1]), diffs, torch.zeros_like(diffs[:, 0:1])], dim=1)
        if universal:
            smoothness = torch.mean(torch.norm(diffs, p=2,dim=-1), dim=-1)
        else:
            smoothness = torch.norm(diffs, p=2,dim=-1) # N, T
    else:
        raise NotImplementedError('Order not implemented')            

    return smoothness #lower is smoother

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
        
        raise NotImplementedError('This class is not implemented yet')
        
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
        raise NotImplementedError('This class is not implemented yet')
        
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
    def __init__(self, U, Y_bar, Y_f, normalizer='identity_normalizer', kuramoto=False):
        super().__init__()
        assert len(U.shape)==3 and len(Y_bar.shape)==3 and len(Y_f.shape)==2 # N, T, m; N, T-1, p; N, p
        U_tensor = torch.tensor(U, dtype=torch.float32)
        U_zeros = torch.zeros(U_tensor.shape[0], 1, U_tensor.shape[2])
        self.U = torch.cat([U_zeros, U_tensor], dim=1)
        self.U = self.U.flip(1) # u is originally from T to 0
        Y_bar_tensor = torch.tensor(Y_bar, dtype=torch.float32)
        Y_f_tensor = torch.tensor(Y_f, dtype=torch.float32)
        Y_0_tensor = torch.zeros(Y_bar_tensor.shape[0], 1, Y_bar_tensor.shape[2])
        if not kuramoto:
            self.Y = torch.cat([Y_0_tensor, Y_bar_tensor, Y_f_tensor.unsqueeze(1)], dim=1) # N, T+1, p
        else:
            self.Y = torch.cat([Y_bar_tensor, Y_f_tensor.unsqueeze(1)], dim=1)
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
    
class TrainData_norm_free(Dataset):
    def __init__(self, U, Y_bar, Y_f, normalizer='identity_normalizer', kuramoto=False, train=False, use_clustering=True, use_smoothness=None):
        super().__init__()
        assert len(U.shape)==3 and len(Y_bar.shape)==3 and len(Y_f.shape)==2 # N, T, m; N, T-1, p; N, p
        U_tensor = torch.tensor(U, dtype=torch.float32)
        U_zeros = torch.zeros(U_tensor.shape[0], 1, U_tensor.shape[2])
        self.U = torch.cat([U_zeros, U_tensor], dim=1)
        self.U = self.U.flip(1) # u is originally from T to 0
        Y_bar_tensor = torch.tensor(Y_bar, dtype=torch.float32)
        Y_f_tensor = torch.tensor(Y_f, dtype=torch.float32)
        Y_0_tensor = torch.zeros(Y_bar_tensor.shape[0], 1, Y_bar_tensor.shape[2])
        if not kuramoto:
            self.Y = torch.cat([Y_0_tensor, Y_bar_tensor, Y_f_tensor.unsqueeze(1)], dim=1) # N, T+1, p
        else:
            self.Y = torch.cat([Y_bar_tensor, Y_f_tensor.unsqueeze(1)], dim=1)
        self.horizon = self.U.shape[1]
        self.train = train
        if use_clustering:
            assert use_smoothness is None, 'Clustering and smoothness cannot be used at the same time'
        if use_smoothness is not None:
            assert not use_clustering, 'Clustering and smoothness cannot be used at the same time'
        self.use_clustering = use_clustering
        assert use_smoothness in ['uni_first', 'uni_second', 'first', 'second', None]
        self.use_smoothness = use_smoothness
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
        energies = [i.norm() for i in self.U]
        self.energies = torch.tensor(energies)
        self.min_energy = torch.min(self.energies)
        self.max_energy = torch.max(self.energies)
        self.energies = (self.energies-self.min_energy)/(self.max_energy-self.min_energy)
        
        self.ranks = -torch.ones_like(self.energies) # -1 means not assigned
        self.smoothness = -torch.ones_like(self.energies) # -1 means not assigned
        if self.train and self.use_clustering:
            # y_0_fs = torch.cat([self.Y[:, 0, :], self.Y[:, -1, :]], dim=-1)
            y_0_fs = (self.Y[:, 0, :]-self.Y[:, -1, :]).abs()
            self.km_label = torch.tensor(clustering(y_0_fs.cpu().numpy(), type='KMeans'))
            self.km_label_set = set(self.km_label.numpy())

            for i in list(self.km_label_set):
                # print(f'Label {i} has {sum(self.km_label==i)} samples')
                indexs = torch.where(self.km_label==i)
                rank_index = torch.argsort(torch.argsort(self.energies[indexs], descending=False)).type(torch.float32)+1e-6 # from min to max
                rank_index /= torch.max(rank_index.type(torch.float32))
                self.ranks[indexs] = rank_index 
            num=10
            fig,ax = plt.subplots(2, num, figsize=(num*5, 10))
            for i in range(num):
                idx = torch.where(self.km_label==i)
                ranks = self.ranks[idx]
                Y_s = self.Y[idx]
                index_0 = torch.where(ranks<1e-5)
                index_0 = index_0[0][torch.randperm(len(index_0[0]))[:1]]
                
                index_1 = torch.where(ranks>1-1e-5)
                # randomly pick 10 index
                index_1 = index_1[0][torch.randperm(len(index_1[0]))[:1]]
            
            
                ax[0, i].plot(Y_s[index_0[0]].cpu().numpy())
                ax[1, i].plot(Y_s[index_1[0]].cpu().numpy())
                # set label
                ax[0, i].set_title(f'Rank: {ranks[index_0[0]]:.2f}, Cluster: {i}')
                ax[1, i].set_title(f'Rank: {ranks[index_1[0]]:.2f}, Cluster: {i}')
            fig.savefig('clustering.png')
            
            assert -1 not in self.ranks, 'Ranking failed to assign'
        if self.train and self.use_smoothness is not None:
            if self.use_smoothness == 'uni_first':
                smoothness = cal_smoothness(self.Y, order=1, universal=True)
            elif self.use_smoothness == 'uni_second':
                smoothness = cal_smoothness(self.Y, order=2, universal=True)
            elif self.use_smoothness == 'first':
                smoothness = cal_smoothness(self.Y, order=1, universal=False)
            elif self.use_smoothness == 'second':
                smoothness = cal_smoothness(self.Y, order=2, universal=False)
            self.smoothness = smoothness
            
            # randomly pick 10 index for low smoothness and high smoothness
            num=10
            fig,ax = plt.subplots(2, num, figsize=(num*5, 10))
            for i in range(num):
                index_0 = torch.where(self.smoothness<min(1,self.smoothness.min()+5e-3))
                index_0 = index_0[0][torch.randperm(len(index_0[0]))[:1]]
                
                index_1 = torch.where(self.smoothness>min(1,self.smoothness.max()-5e-3))
                # randomly pick 10 index
                index_1 = index_1[0][torch.randperm(len(index_1[0]))[:1]]
                ax[0, i].plot(self.Y[index_0[0]].cpu().numpy())
                ax[1, i].plot(self.Y[index_1[0]].cpu().numpy())
                # set label
                ax[0, i].set_title(f'Smoothness: {self.smoothness[index_0[0]]:.2f} index: {index_0[0]} energy: {self.energies[index_0[0]]:.2f}')
                ax[1, i].set_title(f'Smoothness: {self.smoothness[index_1[0]]:.2f} index: {index_1[0]} energy: {self.energies[index_1[0]]:.2f}')
            fig.savefig('smoothness.png')
            
            # normalize smoothness to [0, 1]
            self.smoothness = 1-((self.smoothness-self.smoothness.min())/(self.smoothness.max()-self.smoothness.min()+1e-8)) # the bigger the smoother
        self.alpha = 2

    def __getitem__(self, index):
        u, y = self.U[index], self.Y[index] # T+1, m; T+1, p
        x = torch.cat([u, y], dim=1) # T+1, m+p
        cond = {0: y[0], self.horizon-1: y[-1]}
        if self.use_clustering:
            denoiser_cond =torch.exp(-self.alpha*self.ranks[index])
        elif self.use_smoothness is not None:
            denoiser_cond = self.alpha*(self.smoothness[index]-1)+1
        else:
            denoiser_cond = torch.exp(-self.alpha*self.energies[index])
        # cond = {0: y[0]}
        # cond={}
        return Batch2(x, cond, denoiser_cond)
    
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
        raise NotImplementedError('This class is not implemented yet')

    def __getitem__(self, index):
        u, y = self.U[index], self.Y[index] # T+1, m; T+1, p
        x = torch.cat([u, y], dim=1) # T+1, m+p
        cond = {0: y[0], self.horizon-1: y[-1]}
        # cond = {0: y[0]}
        # cond={}
        return Batch(x, cond)
    
    def __len__(self):
        return self.U.shape[0]