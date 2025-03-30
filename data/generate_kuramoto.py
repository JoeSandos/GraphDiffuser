import numpy as np
from scipy.sparse import csr_matrix
import pickle
import tqdm
import os

# Generate ring network and set Kuramoto parameters

# Network size
n = 8
# Number of control nodes
m = 8
# Control nodes
m_set = np.array([1, 2, 3,4,5,6,7,8])
# Control matrix
B = np.zeros((n, m))
B[m_set-1, np.arange(m)] = 1
assert m_set.shape[0] == m
# Number of neighbors for each node
k = 2

# Define initial and final phase-locked patterns
theta1 = np.mod(0 * np.pi * np.arange(n) / n, 2 * np.pi)
# theta2 = np.mod(4 * np.pi * np.arange(n) / n, 2 * np.pi)

# Natural frequency
omega = np.zeros(n)
sigma = 2
# Construct a regular ring lattice graph
A = np.zeros((n, n))
kHalf = k // 2
for i in range(n):
    for j in range(-kHalf, kHalf + 1):
        if j != 0:
            A[i, (i + j) % n] = 1

# Number of data samples
N = 100000

X0 = []
X_f = []
X_bar = []
Y_bar = []
U = []

tspan = [0.01, 0.16]  # Control time interval
h = 0.01  # Discretization step size
T = int(tspan[-1] / h)  # Number of time steps

for l in tqdm.tqdm(range(N)):
    theta = np.zeros((n, T))
    theta[:, 0] = theta1 + np.random.randn(n)
    
    u = np.zeros((m, T-1))  # Initialize control input
    
    for t in range(T-1):
        u[:, t] = np.random.randn(m)* sigma  # Randomly generate control input
        
        p = 0
        
        for node in range(n):
            if node + 1 in m_set:
                theta[node, t+1] = theta[node, t] + h * omega[node] + h * u[p, t]
                p += 1
            else:
                theta[node, t+1] = theta[node, t] + h * omega[node]
            
            for neighbor in range(n):
                theta[node, t+1] += h * A[node, neighbor] * np.sin(theta[neighbor, t] - theta[node, t])
    
    X0.append(theta[:, 0])
    U.append(np.fliplr(u).T)  # (T-1), #order: u(T-2), u(T-3), ..., u(0)
    X_bar.append(theta[:, 1:T-1].reshape(n * (T-2), 1)) # (T-1)*(n-1), 1
    Y_bar.append(theta[:, 0:T-1].T) # T-1, n
    X_f.append(theta[:, -1])

# Convert to array format
X0 = np.stack(X0).T
X_f = np.stack(X_f).T
X_bar = np.stack(X_bar).T
Y_bar = np.stack(Y_bar)
Y_f = np.copy(X_f).T
U = np.stack(U)

data_name = f'kuramoto_{n}_{m}_{T-1}_{N}_{k}_sigma={sigma}.pkl'
data_dict = {}
data_dict['sys'] = {'A': A, 'k': k, 'B': B, "C": np.identity(n)}
data_dict['adj'] = np.copy(A)
data_dict['meta_data'] = {'num_nodes': n, 'input_dim': m, 'control_horizon': T-1, 'num_samples': N, 'num_edges': A.sum()}
data_dict['data'] = {'X0': X0, 'X_bar': X_bar, 'X_f': X_f, 'U': U, 'Y_bar': Y_bar, 'Y_f': Y_f}
# Create directory if not exist
if not os.path.exists('./data/synthetic_data'):
    os.makedirs('./data/synthetic_data')
with open('./data/synthetic_data/'+data_name, 'wb') as f:
    pickle.dump(data_dict, f)