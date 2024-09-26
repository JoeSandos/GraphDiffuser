import numpy as np
from scipy.linalg import null_space, pinv, cholesky
from scipy.sparse import lil_matrix
import matplotlib.pyplot as plt
import pickle
# %% generate ring network and set Kuramoto parameters

# network size
n = 8
# control node size
m = 8
# control nodes
#m_set = [4, 5, 6]
m_set = [1, 2, 3, 4, 5, 6, 7, 8]
# number of neighbors per node
k = 2

# # define patterns
theta1 = np.zeros((n, 1))  # initial phase-locked pattern
# theta2 = np.zeros((n, 1))  # final phase-locked pattern


# natural frequencies
omega = np.zeros((n, 1))

# Construct a regular ring lattice: a graph with n nodes, each connected to k neighbors, k/2 on each side
A = lil_matrix((n, n))
kHalf = k // 2
for i in range(n):
    for j in range(1, kHalf + 1):
        A[i, (i + j) % n] = 1
        A[i, (i - j) % n] = 1
A = A.toarray()

# %% generate data

# number of data 
N = 500

X0 = np.array([]).reshape(n, 0)
X_f = np.array([]).reshape(n, 0)
X_bar = np.array([]).reshape(n * (int(0.16 / 0.01) - 2), 0)
U = np.array([]).reshape(m * (int(0.16 / 0.01) - 1), 0)

tspan = [0, 0.16]  # control times
h = 0.01  # discretization step
T = int(tspan[-1] / h)
theta = np.zeros((n, T))
u = np.zeros((m, T-1))
sigma=1
for l in range(N):
    
    theta[:,0:1] = theta1 + 1 * np.random.randn(n, 1)  # initial condition
    
    for t in range(int(tspan[-1] / h) - 1):
        
        u[:,t:t+1] = sigma * np.random.randn(m, 1)
        
        p = 0
        
        for node in range(n):
            
            if node + 1 in m_set:
                theta[node, t+1] = theta[node, t] + h * omega[node] + h * u[p,t]
                p += 1
            else:
                theta[node, t+1] = theta[node, t] + h * omega[node]
            
            for neighbor in range(n):
                theta[node, t+1] += h * A[node, neighbor] * np.sin(theta[neighbor, t] - theta[node, t])
    
    X0 = np.hstack((X0, theta[:, [0]]))
    U = np.hstack((U, u.flatten('F')[::-1].reshape(-1, 1)))
    X_bar = np.hstack((X_bar, theta[:, 1:-1].reshape(-1, 1)))
    X_f = np.hstack((X_f, theta[:, [-1]]))

# Null space calculations for matrices
K_X0 = null_space(X0, rcond=1e-10)
K_U = null_space(U, rcond=1e-10)
# read data

with open(f'/data2/chenhongyi/diffcon/GraphDiffuser/data/synthetic_data/kuramoto_8_8_15_100000_2_sigma={sigma}.pkl', 'rb') as f:
    data = pickle.load(f)
theta1s = data['data']['Y_bar'][:10000,0]
theta2s = data['data']['Y_f'][:10000]
numofdata = len(theta1s)
print('numofdata:', numofdata,' theta1s:', theta1s.shape, ' theta2s:', theta2s.shape)
import tqdm
numofdata_tqdm = tqdm.tqdm(range(numofdata))
U_opt = np.zeros((len(theta1s), T-1, m))
Y_bar_opt = np.zeros((len(theta1s), T-1, n))
Y_f_opt = np.zeros((len(theta1s), n))
for i in numofdata_tqdm:
    theta1 = theta1s[i].reshape(-1, 1)
    theta2 = theta2s[i].reshape(-1, 1)
        
    # xf_c calculation
    xf_c = (theta2 - (X_f @ K_U @ pinv(X0 @ K_U, rcond=1e-10)) @ theta1)

    # print('norm of U:', np.linalg.norm(U, 2))
    # print('norm of X_0:', np.linalg.norm(X0, 2))
    # print('norm of X_f:', np.linalg.norm(X_f, 2))

    U_mul = U @ K_X0
    X_bar_mul = X_bar @ K_X0
    X_f_mul = X_f @ K_X0

    # Compute data-driven input
    K_f = null_space(X_f_mul, rcond=1e-10)
    Q = 50 * np.eye(n * (T - 2))
    R = np.eye(m * (T - 1))
    epsilon = 1e-9
    L = cholesky(X_bar_mul.T @ Q @ X_bar_mul + U_mul.T @ R @ U_mul + epsilon * np.eye(X_bar_mul.shape[1]), lower=True)
    W, S, V = np.linalg.svd(L @ K_f, full_matrices=False)

    u_opt = U_mul @ pinv(X_f_mul, rcond=1e-10) @ xf_c - U_mul @ K_f @ pinv(W @ np.diag(S) @ V, rcond=1e-10) @ L @ pinv(X_f_mul, rcond=1e-10) @ xf_c

    # print('norm of uopt:', np.linalg.norm(u_opt, 2))
    # print('norm of theta1:', np.linalg.norm(theta1, 2))
    # print('norm of theta2:', np.linalg.norm(theta2, 2))

    # %% simulate controlled system

    u_opt_seq = np.fliplr(u_opt.reshape(m, int(tspan[-1] / h - 1)))
    U_opt[i] = np.fliplr(u_opt_seq).T
    print(f'norm of u_opt_seq: {np.linalg.norm(U_opt[i], 2)}')
    
    # Simulate the controlled system

    # Time steps
    T_new = T
    theta = np.zeros((n, T_new))
    theta[:, 0] = theta1.flatten()
    u = np.zeros((m, T_new - 1))
    for t in range(T_new - 1):
        if t < T-1:
            u[:, t] = u_opt_seq[:, t]
        else:
            u[:, t] = np.zeros((m, 1)).flatten()
            if t == T-1:
                mse = np.sum((theta[:, t] - theta2.flatten()) ** 2) / n
                mape = np.mean(np.abs((theta[:, t] - theta2.flatten()) / theta[:, t])) * 100
                u_norm = 16 * np.linalg.norm(u_opt_seq, 2)
                # print(f'MSE: {mse}')
                # print(f'MAPE: {mape}%')
                # print(f'16 * norm(u_opt_seq, 2): {u_norm}')
        
        p = 0
        for node in range(n):
            if node + 1 in m_set:
                theta[node, t + 1] = theta[node, t] + h * omega[node] + h * u[p, t]
                p += 1
            else:
                theta[node, t + 1] = theta[node, t] + h * omega[node]
            
            for neighbor in range(n):
                theta[node, t + 1] += h * A[node, neighbor] * np.sin(theta[neighbor, t] - theta[node, t])
    mse = np.sum((theta[:, T-1] - theta2.flatten()) ** 2) / n
    mape = np.mean(np.abs((theta[:, T-1] - theta2.flatten()) / theta[:, T-1])) * 100
    u_norm = 16 * np.linalg.norm(u_opt_seq, 2)
    # print(f'MSE: {mse}')
    # print(f'MAPE: {mape}%')
    print(f'16 * norm(u_opt_seq, 2): {u_norm}')
    Y_bar_opt[i] = theta[:, 0:T-1].T
    Y_f_opt[i] = theta[:, -1]

data_name = f'kuramoto_{n}_{m}_{T-1}_{numofdata}_{k}_sigma={sigma}_data_driven.pkl'
data_dict = {}
data_dict['sys'] = {'A': data['sys']['A'], 'B': data['sys']['B'], 'C': data['sys']['C'], 'k': data['sys']['k']}
data_dict['adj'] = np.copy(data['adj'])
data_dict['meta_data'] = {'num_nodes': n, 'input_dim': m, 'control_horizon': T-1, 'num_samples': numofdata, 'num_edges': data['adj'].sum()}
data_dict['data'] = { 'U': U_opt, 'Y_bar': Y_bar_opt, 'Y_f': Y_f_opt}
with open('/data2/chenhongyi/diffcon/GraphDiffuser/data/synthetic_data/'+data_name, 'wb') as f:
    pickle.dump(data_dict, f)

# # Simulate the controlled system
# theta[:, 0] = theta1.flatten()
# tspan = [0, 0.30]  # Extend the control time

# # Time steps
# T_new = int(tspan[-1] / h)
# theta = np.zeros((n, T_new))
# u = np.zeros((m, T_new - 1))
# for t in range(T_new - 1):
#     if t < T-1:
#         u[:, t] = u_opt_seq[:, t]
#     else:
#         u[:, t] = np.zeros((m, 1)).flatten()
#         if t == T-1:
#             mse = np.sum((theta[:, t] - theta2.flatten()) ** 2) / n
#             mape = np.mean(np.abs((theta[:, t] - theta2.flatten()) / theta[:, t])) * 100
#             u_norm = 16 * np.linalg.norm(u_opt_seq, 2)
#             # print(f'MSE: {mse}')
#             # print(f'MAPE: {mape}%')
#             # print(f'16 * norm(u_opt_seq, 2): {u_norm}')
    
#     p = 0
#     for node in range(n):
#         if node + 1 in m_set:
#             theta[node, t + 1] = theta[node, t] + h * omega[node] + h * u[p, t]
#             p += 1
#         else:
#             theta[node, t + 1] = theta[node, t] + h * omega[node]
        
#         for neighbor in range(n):
#             theta[node, t + 1] += h * A[node, neighbor] * np.sin(theta[neighbor, t] - theta[node, t])
