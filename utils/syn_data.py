import numpy as np
import scipy.linalg as la
import networkx as nx
import time
import pdb
import pickle
# also set parent directory ..
import sys
sys.path.append('/home/joe/projects/GraphDiffuser')
from env.env import LinearEnv
n = 20 # num nodes
m = 5  # input dimension
p = 20 # output dimension
T = 15  # control horizon
N = 1000  # samples
# sigma = 1
# yfs = np.random.randn(N, p, 1)*sigma  # target state

# # adjacency matrix E-R graph

# G = nx.erdos_renyi_graph(n, np.log(n)/n + 0.05)
# A = nx.adjacency_matrix(G).astype(float).todense()
# print(A.sum())
# adj = np.copy(A)
# A /= np.sqrt(n)

# ## input matrix
# B = np.zeros((n, m))
# # randomly set the driver node of each input signal
# b = np.random.permutation(n)[:m] 
# B[b, np.arange(m)] = 1

# # output matrix
# C = np.zeros((p, n))

# c = np.random.permutation(n)[:p]
# C[np.arange(p), c] = 1
# if n==p:
#     C =np.identity(n)

# # state space system
# sys_A = A.astype(np.float32)
# sys_B = B.astype(np.float32)
# sys_C = C.astype(np.float32)
path = '/home/joe/projects/GraphDiffuser/data/synthetic_data/'+ f'erdos_renyi_{n}_{m}_{p}_{T}_{N}.pkl'
with open(path, 'rb') as f:
    pickle_data = pickle.load(f)

sys_A, sys_B, sys_C = pickle_data['sys']['A'], pickle_data['sys']['B'], pickle_data['sys']['C']
adj = pickle_data['adj']
assert (n, m, p, T, N) == (pickle_data['meta_data']['num_nodes'], pickle_data['meta_data']['input_dim'], pickle_data['meta_data']['output_dim'], pickle_data['meta_data']['control_horizon'], pickle_data['meta_data']['num_samples'])
yfs = pickle_data['data']['Y_f'][...,np.newaxis]
yfs = yfs.astype(np.float32)
print(yfs.shape)
env = LinearEnv(sys_A, sys_B, sys_C, adj, T)
for i in range(N):
    yf = yfs[i]
    model_based_u, _ = env.calculate_model_based_control(yf)
    model_based_y = env.from_actions_to_obs_direct(model_based_u)
    # print(model_based_u)
    if i == 0:
        U = model_based_u[np.newaxis, ...]
        Y = model_based_y[np.newaxis, ...]
    else:
        U = np.concatenate((U, model_based_u[np.newaxis, ...]), axis=0)
        Y = np.concatenate((Y, model_based_y[np.newaxis, ...]), axis=0)
    
    data_driven_u, _, _, _ = env.calculate_data_driven_control(yf)
    data_driven_y = env.from_actions_to_obs_direct(data_driven_u)
    
    if i == 0:
        U_data_driven = data_driven_u[np.newaxis, ...]
        Y_data_driven = data_driven_y[np.newaxis, ...]
        
    else:
        U_data_driven = np.concatenate((U_data_driven, data_driven_u[np.newaxis, ...]), axis=0)
        Y_data_driven = np.concatenate((Y_data_driven, data_driven_y[np.newaxis, ...]), axis=0)
    
print(U.shape, U_data_driven.shape)
print(Y.shape, Y_data_driven.shape)
Y_bar = Y[:,:-1,:]
Y_f =  Y[:,-1,:]

Y_data_driven_bar = Y_data_driven[:,:-1,:]
Y_data_driven_f = Y_data_driven[:,-1,:]

data_name = f'erdos_renyi_{n}_{m}_{p}_{T}_{N}_model_based.pkl'
data_dict = {}
data_dict['sys'] = {'A': sys_A, 'B': sys_B, 'C': sys_C}
data_dict['adj'] = adj
data_dict['meta_data'] = {'num_nodes': n, 'input_dim': m, 'output_dim': p, 'control_horizon': T, 'num_samples': N, 'num_edges': adj.sum()}
data_dict['data'] = {'U': U, 'Y_bar': Y_bar, 'Y_f': Y_f}
with open('/home/joe/projects/GraphDiffuser/data/synthetic_data/'+data_name, 'wb') as f:
    pickle.dump(data_dict, f)
    
data_name = f'erdos_renyi_{n}_{m}_{p}_{T}_{N}_data_driven.pkl'
data_dict = {}
data_dict['sys'] = {'A': sys_A, 'B': sys_B, 'C': sys_C}
data_dict['adj'] = adj
data_dict['meta_data'] = {'num_nodes': n, 'input_dim': m, 'output_dim': p, 'control_horizon': T, 'num_samples': N, 'num_edges': adj.sum()}
data_dict['data'] = {'U': U_data_driven, 'Y_bar': Y_data_driven_bar, 'Y_f': Y_data_driven_f}
with open('/home/joe/projects/GraphDiffuser/data/synthetic_data/'+data_name, 'wb') as f:
    pickle.dump(data_dict, f)
# # compute T-steps system Hankel matrix
# C_mat = np.dot(sys_C, sys_B)
# for i in range(T):
#     C_mat = np.concatenate((C_mat, np.dot(sys_C, np.linalg.matrix_power(sys_A, i+1)).dot(sys_B)), axis=-1)
# print(C_mat.shape)
# u,s,v = np.linalg.svd(C_mat)
# if s.min()<1e-5:
#     raise ValueError
# H = np.zeros((p * T, m * T))
# for r in range(1, T+1):
#     for k in range(1, T+1):
#         if k > T - r:
#             H[(r-1)*p:r*p, (k-1)*m:k*m] = np.dot(sys_C, np.linalg.matrix_power(sys_A, r-T+k-1)).dot(sys_B)

# U = np.random.randn(m * T, N)*sigma
# U_3d = np.reshape(U, (T, m, N)).transpose(2, 0, 1)
# Y = np.dot(H, U)
# Y_bar = Y[:(T - 1) * p, :]
# Y_bar_3d = np.reshape(Y_bar, ((T - 1), p, N)).transpose(2, 0, 1)
# Y_f = Y[(T - 1) * p:, :]
# Y_f_3d = Y_f.transpose(1, 0)
# # output shape
# print(U_3d.shape)
# print(Y_bar_3d.shape, Y_f_3d.shape)

# save as .pkl
# data_name = f'erdos_renyi_{n}_{m}_{p}_{T}_{N}_data_driven.pkl'
# data_dict = {}
# data_dict['sys'] = {'A': sys_A, 'B': sys_B, 'C': sys_C}
# data_dict['adj'] = adj
# data_dict['meta_data'] = {'num_nodes': n, 'input_dim': m, 'output_dim': p, 'control_horizon': T, 'num_samples': N, 'num_edges': adj.sum()}
# data_dict['data'] = {'U': U_3d, 'Y_bar': Y_bar_3d, 'Y_f': Y_f_3d}
# with open('/home/joe/projects/GraphDiffuser/data/synthetic_data/'+data_name, 'wb') as f:
#     pickle.dump(data_dict, f)