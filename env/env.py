import numpy as np
import torch
import networkx as nx

class EnvBase:
    def __init__(self):
        pass
    def reset(self):
        pass
    def step(self):
        pass
    def get_state(self):
        pass

class LinearEnv(EnvBase):
    def __init__(self, A, B, C, adj, T, device='cpu'):
        super().__init__()
        assert type(A)==type(B)==type(C)==type(adj)
        self.device = device
        
        if type(A)==np.ndarray or type(A)==np.matrix:
            self.A = torch.tensor(A, dtype=torch.float32).to(device)
            self.B = torch.tensor(B, dtype=torch.float32).to(device)
            self.C = torch.tensor(C, dtype=torch.float32).to(device)
            self.adj = torch.tensor(adj, dtype=torch.float32).to(device)
        else:
            self.A = A.to(device)
            self.B = B.to(device)
            self.C = C.to(device)
            self.adj = adj.to(device)
        self.num_edges = torch.sum(self.adj)

        self.max_T = T
        self.num_nodes = A.shape[0]
        self.num_driver = B.shape[1]
        self.num_observation = C.shape[0]
        self.reset()
        self.calculate_C_0()
    def reset(self, start=None):
        if start is None:
            self.x = torch.zeros(self.num_nodes).to(self.device)
        else:
            assert len(start) == self.num_nodes, "Invalid state"
            self.x = start
        self.T = 0
        return torch.matmul(self.C, self.x), self.x, self.T
    
    
    def step(self, u, target=None):
        terminal = False
        if target is not None:
            assert target.shape == (self.num_observation,), "Invalid target"
        # if self.T >= self.max_T:
        #     return None, None, True
        self.T += 1
        if target is None:
            if self.T == self.max_T:
                terminal = True
        else:
            y = torch.matmul(self.C, self.x)
            if torch.norm(y - target) < 1e-3:
                terminal = True
        assert len(u) == self.num_driver, f"Invalid input, {len(u)},{self.num_driver}"
        self.x = torch.matmul(self.A, self.x) + torch.matmul(self.B, u)
        return torch.matmul(self.C, self.x), self.x, terminal
    
    def get_state(self):
        return self.x
    
    def from_actions_to_obs(self, actions, start=None):
        assert len(actions) == self.max_T, "Invalid actions"
        if start is not None:
            self.reset(start)
        else:
            self.reset()
        observations = []
        for a in actions:
            obs, _, _ = self.step(a)
            observations.append(obs)
        return torch.stack(observations).to(self.device)
    
    def from_actions_to_obs_direct(self, actions, start=None):
        """
        actions u(0), u(1), ..., u(T-1)
        """

        H = torch.zeros((self.num_observation * self.max_T, self.num_driver * self.max_T)).to(self.device)

        for r in range(1, self.max_T + 1):
            for k in range(1, self.max_T + 1):
                if k > self.max_T - r:
                    H[(r-1)*self.num_observation:r*self.num_observation, (k-1)*self.num_driver:k*self.num_driver] = \
                        torch.matmul(self.C, torch.matrix_power(self.A, r-self.max_T+k-1)).matmul(self.B)
        actions_c = torch.flip(actions, [0]) #NOTE: u need to be inversed (i.e. u(T-1), u(T-2),...u(0))
        U = actions_c.reshape(-1) 
        Y = torch.matmul(H, U)
        Y = Y.reshape(self.max_T, self.num_observation)
        return Y

    def calculate_C_0(self):
        C_o = torch.zeros(self.num_nodes, self.num_driver * self.max_T).to(self.device)
        C_o[:, :self.num_driver] = self.B
        for k in range(1, self.max_T):
            C_o[:, self.num_driver*k:(k+1)*self.num_driver] = self.A @ C_o[:, self.num_driver*(k-1):self.num_driver*k]
        self.C_o = self.C @ C_o
        
    def calculate_model_based_control(self, y_f):

        # # Initialize the output controllability matrix
        # C_o = torch.zeros(self.num_nodes, self.num_driver * self.max_T)

        # # First block is simply the B matrix
        # C_o[:, :self.num_driver] = self.B

        # # Compute the controllability matrix for the whole time horizon
        # for k in range(1, self.max_T):
        #     C_o[:, self.num_driver*k:(k+1)*self.num_driver] = self.A @ C_o[:, self.num_driver*(k-1):self.num_driver*k]

        # Apply the output matrix C to the controllability matrix
        if self.C_o is None:
            self.calculate_C_0()
        C_o = self.C_o

        # Minimum-energy model-based control

        u = torch.pinverse(C_o.cpu()).to(self.device) @ y_f # shape: (num_driver * max_T, num_observation) @ (num_observation, 1) = (num_driver * max_T, 1)
        y_f_hat = torch.matmul(C_o, u) # shape: (num_nodes, num_driver * max_T) @ (num_driver * max_T, 1) = (num_nodes, 1)
        u_reshape = u.reshape(self.max_T, self.num_driver).flip([0])
        return u_reshape, y_f_hat
    
    def calculate_data_driven_control(self, y_f):
        U = torch.randn(self.num_driver * self.max_T, 80).to(self.device)
        Y = torch.matmul(self.C_o, U)
        u_dd = torch.pinverse(Y @ torch.pinverse(U)) @ y_f
        u_dd_r = u_dd.reshape(self.max_T, self.num_driver).flip([0])
        u_dd_approx = U @ torch.pinverse(Y) @ y_f
        u_dd_approx_r = u_dd_approx.reshape(self.max_T, self.num_driver).flip([0])
        y_f_hat = torch.matmul(self.C_o, u_dd)
        y_f_hat_approx = torch.matmul(self.C_o, u_dd_approx)
        
        return u_dd_r, y_f_hat, u_dd_approx_r, y_f_hat_approx


#generate kuramoto code:
# import numpy as np
# from scipy.sparse import csr_matrix
# import pickle
# import tqdm
# # 生成环形网络并设置Kuramoto参数

# # 网络大小
# n = 16
# # 控制节点的数量
# m = 16
# # 控制节点
# m_set = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
# # control matrix
# B = np.zeros((n, m))
# B[m_set-1, np.arange(m)] = 1
# assert m_set.shape[0] == m
# # 每个节点的邻居数
# k = 2

# # 定义初始和最终的相位锁定模式
# theta1 = np.mod(0 * np.pi * np.arange(n) / n, 2 * np.pi)
# theta2 = np.mod(2 * np.pi * np.arange(n) / n, 2 * np.pi)

# # 自然频率
# omega = np.zeros(n)

# # 构建一个规则的环形格子图
# A = np.zeros((n, n))
# kHalf = k // 2
# for i in range(n):
#     for j in range(-kHalf, kHalf + 1):
#         if j != 0:
#             A[i, (i + j) % n] = 1

# # 数据量
# N = 1000

# X0 = []
# X_f = []
# X_bar = []
# Y_bar = []
# U = []

# tspan = [0.01, 0.16]  # 控制时间段
# h = 0.01  # 离散化步长
# T = int(tspan[-1] / h)  # 时间步数



# for l in tqdm.tqdm(range(N)):
#     theta = np.zeros((n, T))
#     theta[:, 0] = theta1 + 0.1 * np.random.randn(n)
    
#     u = np.zeros((m, T-1))  # 控制输入初始化
    
#     for t in range(T-1):
#         u[:, t] = 0.1 * np.random.randn(m)  # 随机生成控制输入
        
#         p = 0
        
#         for node in range(n):
#             if node + 1 in m_set:
#                 theta[node, t+1] = theta[node, t] + h * omega[node] + h * u[p, t]
#                 p += 1
#             else:
#                 theta[node, t+1] = theta[node, t] + h * omega[node]
            
#             for neighbor in range(n):
#                 theta[node, t+1] += h * A[node, neighbor] * np.sin(theta[neighbor, t] - theta[node, t])
    
#     X0.append(theta[:, 0])
#     U.append(np.fliplr(u).T)  # (T-1), m
#     X_bar.append(theta[:, 1:T-1].reshape(n * (T-2), 1)) # (T-1)*(n-1), 1
#     Y_bar.append(theta[:, 0:T-1].T) # T-1, n
#     X_f.append(theta[:, -1])

# # 转换为数组格式
# X0 = np.stack(X0).T
# X_f = np.stack(X_f).T
# X_bar = np.stack(X_bar).T
# Y_bar = np.stack(Y_bar)
# Y_f = np.copy(X_f).T
# U = np.stack(U)

# data_name = f'kuramoto_{n}_{m}_{T}_{N}_{k}.pkl'
# data_dict = {}
# data_dict['sys'] = {'A': A, 'k': k, 'B': B, "C": np.identity(n)}
# data_dict['adj'] = np.copy(A)
# data_dict['meta_data'] = {'num_nodes': n, 'input_dim': m, 'control_horizon': T, 'num_samples': N, 'num_edges': A.sum()}
# data_dict['data'] = {'X0': X0, 'X_bar': X_bar, 'X_f': X_f, 'U': U, 'Y_bar': Y_bar, 'Y_f': Y_f}
# with open('/data2/chenhongyi/diffCon/GraphDiffuser/data/synthetic_data/'+data_name, 'wb') as f:
#     pickle.dump(data_dict, f)
        
class Kuramoto(EnvBase):
    def __init__(self, A, B, C, k, T, device='cpu'):
        super().__init__()
        if type(A)==np.ndarray or type(A)==np.matrix:
            self.A = torch.tensor(A, dtype=torch.float32).to(device)
            self.B = torch.tensor(B, dtype=torch.float32).to(device)
            self.C = torch.tensor(C, dtype=torch.float32).to(device)
        else:
            self.A = A.to(device)
            self.B = B.to(device)
            self.C = C.to(device)
        self.k = k
        self.num_nodes = A.shape[0]
        self.num_driver = B.shape[1]
        self.num_observation = C.shape[0]
        self.max_T = T
        self.device = device
        self.h = torch.tensor(0.01, dtype=torch.float32).to(device)
        self.reset()
    
    def reset(self, start=None):
        theta1 = np.mod(0 * np.pi * np.arange(self.num_nodes) / self.num_nodes, 2 * np.pi) # initial phase
        theta2 = np.mod(4 * np.pi * np.arange(self.num_nodes) / self.num_nodes, 2 * np.pi) # final phase
        self.omega = torch.zeros(self.num_nodes).to(self.device)
        if start is None:
            self.current_phase = torch.tensor(theta1, dtype=torch.float32).to(self.device)
        else:
            self.current_phase = start
        self.T = 0
        return self.current_phase, self.current_phase, self.T
    
    def step(self, u):
        terminal = False

        self.T += 1
        if self.T == self.max_T:
            terminal = True
        assert len(u) == self.num_driver, f"Invalid input, {len(u)},{self.num_driver}"
        u_to_x = self.B @ u
        next_phase = self.current_phase + self.h * self.omega + self.h * u_to_x
        for node in range(self.num_nodes):
            for neighbor in range(self.num_nodes):
                next_phase[node] += self.h * self.A[node, neighbor] * torch.sin(self.current_phase[neighbor] - self.current_phase[node])
        self.current_phase = next_phase
        return torch.matmul(self.C, self.current_phase), self.current_phase, terminal
    
    def get_state(self):
        return self.current_phase
    
    def from_actions_to_obs(self, actions, start=None):
        assert len(actions) == self.max_T, f"Invalid actions, {len(actions)},{self.max_T}"
        if start is not None:
            self.reset(start)
        else:
            self.reset()
        observations = []
        for a in actions:
            obs, _, _ = self.step(a)
            observations.append(obs)
        return torch.stack(observations).to(self.device)
    
    def from_actions_to_obs_longer(self, actions, start=None, continues=2):
        assert len(actions) == self.max_T, "Invalid actions"
        if start is not None:
            self.reset(start)
        else:
            self.reset()
        observations = []
        for a in actions:
            obs, _, _ = self.step(a)
            observations.append(obs)
        for i in range(continues*self.max_T):
            obs, _, _ = self.step(torch.zeros(self.num_driver))
            observations.append(obs)
        return torch.stack(observations).to(self.device)
    
    def from_actions_to_obs_direct(self, actions, start=None):
        return self.from_actions_to_obs(actions, start)
    
    def calculate_model_based_control(self, y_f):
        # Minimum-energy model-based control
        # print("Not implemented")
        return torch.zeros(self.max_T, self.num_driver), torch.zeros(self.num_nodes)
    
    def calculate_data_driven_control(self, y_f):
        # print("Not implemented")
        return torch.zeros(self.max_T, self.num_driver), torch.zeros(self.num_nodes), torch.zeros(self.max_T, self.num_driver), torch.zeros(self.num_nodes)
    
    def calculate_C_0(self):
        # print("Not implemented")
        pass
        
