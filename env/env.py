import numpy as np
import torch
import networkx as nx
# from data.generate_power import swing, T_fault_in, T_fault_end
from data.generate_burgers import burgers_update
# from data.generate_ip import inverted_pendulum as ip
from matplotlib import pyplot as plt


def null_space(A, rtol=1e-10):
    if A.numel() == 0:
        return torch.empty(A.shape[1], 0)
    u, s, v = torch.svd(A)
    rank = (s > rtol * s.max()).sum().item()
    return v[:, rank:]

def compute_optimal_input(n, m, T, X0, U, X_f, y_0, y_f, X_bar):

    # Compute K_X0 and K_U
    K_X0 = null_space(X0, 1e-10)
    K_U = null_space(U, 1e-10)

    # Compute xf_c
    if K_U.shape[1] > 0:
        xf_c = y_f - (X_f @ K_U @ torch.pinverse(X0 @ K_U, rcond=1e-10)) @ y_0
    else:
        xf_c = y_f

    # Update U, X_bar, X_f
    if K_X0.shape[1] > 0:
        U = U @ K_X0
        X_bar = X_bar @ K_X0
        X_f = X_f @ K_X0

    # Ensure the dimensions of Q and R are compatible with X_bar and U
    Q_dim = X_bar.shape[1]
    R_dim = U.shape[1]
    Q = 50 * torch.eye(Q_dim)
    R = torch.eye(R_dim)

    # Compute data-driven input
    K_f = null_space(X_f, 1e-10)
    if K_f.shape[1] > 0:
        L = torch.linalg.cholesky(X_bar.T @ Q @ X_bar + U.T @ R @ U, upper=False)
        W, S, V = torch.svd_lowrank(L @ K_f, q=min(m * (T - 1) - n, K_f.shape[1]))
        u_opt = U @ torch.pinverse(X_f, rcond=1e-10) @ xf_c - U @ K_f @ torch.pinverse(W @ S @ V.T, rcond=1e-10) @ L @ torch.pinverse(X_f, rcond=1e-10) @ xf_c
    else:
        u_opt = U @ torch.pinverse(X_f, rcond=1e-10) @ xf_c

    return u_opt

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
        
class Power(EnvBase):
    def __init__(self, C,m,n, T, device='cpu'):
        super().__init__()
        if type(C)==np.ndarray or type(C)==np.matrix:
            self.C = torch.tensor(C, dtype=torch.float32).to(device)
        else:
            self.C = C.to(device)
        self.num_nodes = n
        self.num_driver = m
        self.num_observation = 2*n
        self.max_T = T #action len
        self.device = device
        self.delta_t = 0.01
        self.reset()
    
    def reset(self, start=None):
        theta1 = np.zeros(self.num_observation) # initial phase
        # theta2 = np.mod(4 * np.pi * np.arange(self.num_nodes) / self.num_nodes, 2 * np.pi) # final phase
        self.omega = torch.zeros(self.num_observation).to(self.device)
        if start is None:
            self.current_phase = torch.tensor(theta1, dtype=torch.float32).to(self.device)
        else:
            self.current_phase = start
        self.T = 0
        return self.current_phase, self.current_phase, self.T
    
    def step(self, u):
        terminal = False

        if self.T == self.max_T*self.delta_t:
            terminal = True
        assert len(u) == self.num_driver, f"Invalid input, {len(u)},{self.num_driver}"
        
        out=swing(self.T, self.delta_t, self.current_phase.cpu().numpy(), u.cpu().numpy())
        self.T += self.delta_t
        if not isinstance(out,torch.Tensor):
            out = torch.tensor(out, dtype=torch.float32).to(u.device)
        self.current_phase = out
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
        return torch.zeros(self.max_T, self.num_driver), torch.zeros(self.num_observation)
    
    def calculate_data_driven_control(self, y_f):
        # print("Not implemented")
        return torch.zeros(self.max_T, self.num_driver), torch.zeros(self.num_observation), torch.zeros(self.max_T, self.num_driver), torch.zeros(self.num_observation)
    
    def calculate_C_0(self):
        # print("Not implemented")
        pass
    
    
    def generate_case(self, u, stage, name=None):
        T_control_in = T_fault_end+0.1
        T_control_end = T_control_in+31*self.delta_t
        T_control_in_dis = int(T_control_in/self.delta_t)
        T_control_end_dis = int(T_control_end/self.delta_t)
        u_0 = np.zeros((self.num_driver, T_control_in_dis))
        
        y = np.zeros((self.num_observation, 8*T_control_end_dis))
        for t in range(T_control_in_dis):
            y[:, t+1] = swing(t*self.delta_t, self.delta_t, y[:, t], u_0[:, t])
        
        y_0 = y[:, T_control_in_dis]
        y_f = np.zeros((self.num_observation))
        if stage==0:
            return y_0, y_f
        elif stage==1:
            assert u.size(1) == T_control_end_dis-T_control_in_dis, f"Invalid input, {u.size(1)},{T_control_end_dis-T_control_in_dis}"
            u_T = np.concatenate((u_0, u), axis=1)
            u_max_T = np.concatenate((u_T, np.zeros((self.num_driver, 8*T_control_end_dis-T_control_end_dis))), axis=1)
            for t in range(T_control_in_dis, 8*T_control_end_dis-1):
                y[:, t+1] = swing(t*self.delta_t, self.delta_t, y[:, t], u_max_T[:, t])
        
            # plot y[:9] in one figure
            
            fig1, ax1 = plt.subplots(2, 1, figsize=(10, 6))
            for i in range(9):
                # plot till T_control_end_dis*2
                ax1[0].plot(np.arange(T_control_end_dis+100)*self.delta_t, y[i, :T_control_end_dis+100], label=f'node {i+2}')
                # title
                ax1[0].set_xlabel('Time (s)')
                ax1[0].set_ylabel('Phase')
                # draw vertical area between T_control_in and T_control_end; T_fault_in and T_fault_end
                ax1[0].axvspan(T_control_in, T_control_end, color='lightgreen', alpha=0.2)
                ax1[0].axvspan(T_fault_in, T_fault_end, color='pink', alpha=0.2)
                # plot till 8*T_control_end_dis
                ax1[1].plot(np.arange(8*T_control_end_dis)*self.delta_t, y[i, :], label=f'node {i+2}')
                # title
                ax1[1].set_xlabel('Time (s)')
                ax1[1].set_ylabel('Phase')
                ax1[1].axvspan(T_control_in, T_control_end, color='lightgreen', alpha=0.2)
                ax1[1].axvspan(T_fault_in, T_fault_end, color='pink', alpha=0.2)
                # save fig
            fig1.savefig(f'figures/{name}_phase.png')
            plt.close(fig1)
            plt.clf()
            
            # plot y[9:] in one figure
            fig2, ax2 = plt.subplots(2, 1, figsize=(10, 6))
            for i in range(9, self.num_observation):
                # plot till T_control_end_dis*2
                ax2[0].plot(np.arange(T_control_end_dis+100)*self.delta_t, y[i, :T_control_end_dis+100], label=f'node {i+2}')
                # title
                ax2[0].set_xlabel('Time (s)')
                ax2[0].set_ylabel('freq')
                ax2[0].axvspan(T_control_in, T_control_end, color='lightgreen', alpha=0.2)
                ax2[0].axvspan(T_fault_in, T_fault_end, color='pink', alpha=0.2)
                # plot till 8*T_control_end_dis
                ax2[1].plot(np.arange(8*T_control_end_dis)*self.delta_t, y[i, :], label=f'node {i+2}')
                # title
                ax2[1].set_xlabel('Time (s)')
                ax2[1].set_ylabel('freq')
                ax2[1].axvspan(T_control_in, T_control_end, color='lightgreen', alpha=0.2)
                ax2[1].axvspan(T_fault_in, T_fault_end, color='pink', alpha=0.2)
                # save fig
            fig2.savefig(f'./figures/{name}_freq.png')
            plt.close(fig2)
            plt.clf()
            return y_0, y
    

            
# class Burger(EnvBase):
#     def __init__(self, C,m,n, T, device='cpu'):
#         super().__init__()
#         if type(C)==np.ndarray or type(C)==np.matrix:
#             self.C = torch.tensor(C, dtype=torch.float32).to(device)
#         else:
#             self.C = C.to(device)
#         self.num_nodes = n
#         self.num_driver = m
#         self.num_observation = n
#         assert n==m, "Invalid input"
#         self.max_T = T #action len
#         self.device = device
#         self.delta_t = 1
#         self.reset()
    
#     def reset(self, start=None):
#         theta1 = np.zeros(self.num_nodes) # initial phase
#         # theta2 = np.mod(4 * np.pi * np.arange(self.num_nodes) / self.num_nodes, 2 * np.pi) # final phase
#         self.omega = torch.zeros(self.num_nodes).to(self.device)
#         if start is None:
#             self.current_state = torch.tensor(theta1, dtype=torch.float32).to(self.device)
#         else:
#             self.current_state = start
#         self.T = 0
#         return self.current_state, self.current_state, self.T
    
#     def step(self, u):
#         terminal = False

#         if self.T == self.max_T*self.delta_t:
#             terminal = True
#         assert len(u) == self.num_driver, f"Invalid input, {len(u)},{self.num_driver}"
#         # TODO: implement the burger's equation
        
#         out = burger(self.T, self.delta_t, self.current_state.cpu().numpy(), u.cpu().numpy())
        
#         self.T += self.delta_t
#         if not isinstance(out,torch.Tensor):
#             out = torch.tensor(out, dtype=torch.float32).to(u.device)
#         self.current_state = out
#         return torch.matmul(self.C, self.current_state), self.current_state, terminal
    
#     def get_state(self):
#         return self.current_state
    
#     def from_actions_to_obs(self, actions, start=None):
#         assert len(actions) == self.max_T, f"Invalid actions, {len(actions)},{self.max_T}"
#         if start is not None:
#             self.reset(start)
#         else:
#             self.reset()
#         observations = []
#         for a in actions:
#             obs, _, _ = self.step(a)
#             observations.append(obs)
#         return torch.stack(observations).to(self.device)
    
#     def from_actions_to_obs_longer(self, actions, start=None, continues=2):
#         assert len(actions) == self.max_T, "Invalid actions"
#         if start is not None:
#             self.reset(start)
#         else:
#             self.reset()
#         observations = []
#         for a in actions:
#             obs, _, _ = self.step(a)
#             observations.append(obs)
#         for i in range(continues*self.max_T):
#             obs, _, _ = self.step(torch.zeros(self.num_driver))
#             observations.append(obs)
#         return torch.stack(observations).to(self.device)
    
#     def from_actions_to_obs_direct(self, actions, start=None):
#         return self.from_actions_to_obs(actions, start)
    
#     def calculate_model_based_control(self, y_f):
#         # Minimum-energy model-based control
#         # print("Not implemented")
#         return torch.zeros(self.max_T, self.num_driver), torch.zeros(self.num_nodes)
    
#     def calculate_data_driven_control(self, y_f):
#         return torch.zeros(self.max_T, self.num_driver), torch.zeros(self.num_nodes), torch.zeros(self.max_T, self.num_driver), torch.zeros(self.num_nodes)
#         U = torch.randn(self.num_driver * self.max_T, 80).to(self.device) # 
#         Y = torch.matmul(self.C_o, U)
#         u_dd = torch.pinverse(Y @ torch.pinverse(U)) @ y_f
#         u_dd_r = u_dd.reshape(self.max_T, self.num_driver).flip([0])
#         u_dd_approx = U @ torch.pinverse(Y) @ y_f
#         u_dd_approx_r = u_dd_approx.reshape(self.max_T, self.num_driver).flip([0])
#         y_f_hat = self.from_actions_to_obs_direct(u_dd_r)
#         y_f_hat_approx = self.from_actions_to_obs_direct(u_dd_approx_r)
        
#         return u_dd_r, y_f_hat, u_dd_approx_r, y_f_hat_approx
    
#     def calculate_data_driven_control_with_data(self, y_f, U, Y):
#         u_dd = torch.pinverse(Y @ torch.pinverse(U)) @ y_f
#         u_dd_r = u_dd.reshape(self.max_T, self.num_driver).flip([0])
#         u_dd_approx = U @ torch.pinverse(Y) @ y_f
#         u_dd_approx_r = u_dd_approx.reshape(self.max_T, self.num_driver).flip([0])
#         y_f_hat = self.from_actions_to_obs_direct(u_dd_r)
#         y_f_hat_approx = self.from_actions_to_obs_direct(u_dd_approx_r)
        
#         return u_dd_r, y_f_hat, u_dd_approx_r, y_f_hat_approx
    
#     def calculate_data_driven_control_with_data_complete(self, y_f, y_0, U, Y_bar, Y_f):
#         X_0 = Y_bar[:, 0].reshape(Y_bar.shape[0], -1).transpose()
#         X_bar = Y_bar[:, 1:].reshape(Y_bar.shape[0], -1).transpose()
#         X_f = Y_f.reshape(Y_f.shape[0], -1).transpose()
#         U = U.reshape(U.shape[0], -1).transpose()

#         # change to torch tensor
#         X_0 = torch.tensor(X_0, dtype=torch.float32).to(self.device)
#         X_bar = torch.tensor(X_bar, dtype=torch.float32).to(self.device)
#         X_f = torch.tensor(X_f, dtype=torch.float32).to(self.device)
#         U = torch.tensor(U, dtype=torch.float32).to(self.device)
#         y_0 = torch.tensor(y_0, dtype=torch.float32).to(self.device)
#         y_f = torch.tensor(y_f, dtype=torch.float32).to(self.device)
        
#         u_dd = compute_optimal_input(self.num_nodes, self.num_driver, self.max_T, X_0, U, X_f, y_0, y_f, X_bar)
#         u_dd_r = u_dd.reshape(self.max_T, self.num_driver).flip([0])
#         y_f_hat = self.from_actions_to_obs_direct(u_dd_r, y_0)
#         return u_dd_r, y_f_hat
    
#     def calculate_C_0(self):
#         # print("Not implemented")
#         pass

class Burgers(EnvBase):
    def __init__(self, C,m,n, T, device='cpu'):
        super().__init__()
        if type(C)==np.ndarray or type(C)==np.matrix:
            self.C = torch.tensor(C, dtype=torch.float32).to(device)
        else:
            self.C = C.to(device)
        self.num_nodes = n
        self.num_driver = m
        self.num_observation = n
        assert n==m, "Invalid input"
        self.max_T = T #action len
        self.device = device
        self.dt = 0.0001
        self.t_sample_interval = 1000
        self.nt = 11000
        self.nt1 = 11
        self.nx = 128
        self.dx = 1/self.nx
        self.x_range = (0,1)
        self.reset()
    
    def reset(self, start=None):
        theta1 = np.zeros(self.num_nodes) # initial phase
        if start is None:
            self.current_state = torch.tensor(theta1, dtype=torch.float32).to(self.device)
        else:
            self.current_state = start
        self.T = 0
        return self.current_state, self.current_state, self.T
    
    def step(self, u):
        terminal = False

        if self.T == self.nt1-1:
            terminal = True
        assert len(u) == self.num_driver, f"Invalid input, {len(u)},{self.num_driver}"
        # TODO: implement the burger's equation
        if isinstance(self.current_state, torch.Tensor):
            y = self.current_state.cpu().numpy()
        if isinstance(u, torch.Tensor):
            u = u.cpu().numpy()
        for ts in range(self.T*self.t_sample_interval, (self.T+1)*self.t_sample_interval):
            y=burgers_update(y, u, self.dx, self.dt)
            
        out = y
        self.T += 1
        if not isinstance(out,torch.Tensor):
            out = torch.tensor(out, dtype=torch.float32).to(self.device)
        self.current_state = out
        return torch.matmul(self.C, self.current_state), self.current_state, terminal
    
    def get_state(self):
        return self.current_state
    
    def from_actions_to_obs(self, actions, start=None):
        assert len(actions) == self.max_T, f"Invalid actions, {len(actions)},{self.max_T}"
        if start is not None:
            self.reset(start)
        else:
            self.reset()
        observations = []
        for a in actions:
            obs, _, terminal = self.step(a)
            observations.append(obs)
        assert terminal, "Invalid actions"
        return torch.stack(observations).to(self.device)
    
    def from_actions_to_obs_longer(self, actions, start=None, continues=2):
        # assert len(actions) == self.max_T, "Invalid actions"
        # if start is not None:
        #     self.reset(start)
        # else:
        #     self.reset()
        # observations = []
        # for a in actions:
        #     obs, _, _ = self.step(a)
        #     observations.append(obs)
        # for i in range(continues*self.max_T):
        #     obs, _, _ = self.step(torch.zeros(self.num_driver))
        #     observations.append(obs)
        raise
        return torch.stack(observations).to(self.device)
    
    def from_actions_to_obs_direct(self, actions, start=None):
        return self.from_actions_to_obs(actions, start)
    
    def calculate_model_based_control(self, y_f):
        # Minimum-energy model-based control
        # print("Not implemented")
        return torch.zeros(self.max_T, self.num_driver), torch.zeros(self.num_nodes)
    
    def calculate_data_driven_control(self, y_f):
        return torch.zeros(self.max_T, self.num_driver), torch.zeros(self.num_nodes), torch.zeros(self.max_T, self.num_driver), torch.zeros(self.num_nodes)
        U = torch.randn(self.num_driver * self.max_T, 80).to(self.device) # 
        Y = torch.matmul(self.C_o, U)
        u_dd = torch.pinverse(Y @ torch.pinverse(U)) @ y_f
        u_dd_r = u_dd.reshape(self.max_T, self.num_driver).flip([0])
        u_dd_approx = U @ torch.pinverse(Y) @ y_f
        u_dd_approx_r = u_dd_approx.reshape(self.max_T, self.num_driver).flip([0])
        y_f_hat = self.from_actions_to_obs_direct(u_dd_r)
        y_f_hat_approx = self.from_actions_to_obs_direct(u_dd_approx_r)
        
        return u_dd_r, y_f_hat, u_dd_approx_r, y_f_hat_approx
    
    def calculate_data_driven_control_with_data(self, y_f, U, Y):
        u_dd = torch.pinverse(Y @ torch.pinverse(U)) @ y_f
        u_dd_r = u_dd.reshape(self.max_T, self.num_driver).flip([0])
        u_dd_approx = U @ torch.pinverse(Y) @ y_f
        u_dd_approx_r = u_dd_approx.reshape(self.max_T, self.num_driver).flip([0])
        y_f_hat = self.from_actions_to_obs_direct(u_dd_r)
        y_f_hat_approx = self.from_actions_to_obs_direct(u_dd_approx_r)
        
        return u_dd_r, y_f_hat, u_dd_approx_r, y_f_hat_approx
    
    def calculate_data_driven_control_with_data_complete(self, y_f, y_0, U, Y_bar, Y_f):
        X_0 = Y_bar[:, 0].reshape(Y_bar.shape[0], -1).transpose()
        X_bar = Y_bar[:, 1:].reshape(Y_bar.shape[0], -1).transpose()
        X_f = Y_f.reshape(Y_f.shape[0], -1).transpose()
        U = U.reshape(U.shape[0], -1).transpose()

        # change to torch tensor
        X_0 = torch.tensor(X_0, dtype=torch.float32).to(self.device)
        X_bar = torch.tensor(X_bar, dtype=torch.float32).to(self.device)
        X_f = torch.tensor(X_f, dtype=torch.float32).to(self.device)
        U = torch.tensor(U, dtype=torch.float32).to(self.device)
        y_0 = torch.tensor(y_0, dtype=torch.float32).to(self.device)
        y_f = torch.tensor(y_f, dtype=torch.float32).to(self.device)
        
        u_dd = compute_optimal_input(self.num_nodes, self.num_driver, self.max_T, X_0, U, X_f, y_0, y_f, X_bar)
        u_dd_r = u_dd.reshape(self.max_T, self.num_driver).flip([0])
        y_f_hat = self.from_actions_to_obs_direct(u_dd_r, y_0)
        return u_dd_r, y_f_hat
    
    def calculate_C_0(self):
        # print("Not implemented")
        pass
    

class IP(EnvBase):
    def __init__(self, C,m,n, T, device='cpu'):
        super().__init__()
        if type(C)==np.ndarray or type(C)==np.matrix:
            self.C = torch.tensor(C, dtype=torch.float32).to(device)
        else:
            self.C = C.to(device)
        self.num_nodes = n
        self.num_driver = m
        self.num_observation = n
        self.max_T = T #action len
        self.device = device
        self.delta_t = 0.01
        self.reset()
    
    def reset(self, start=None):
        theta1 = np.zeros(self.num_nodes) # initial phase
        # theta2 = np.mod(4 * np.pi * np.arange(self.num_nodes) / self.num_nodes, 2 * np.pi) # final phase
        self.omega = torch.zeros(self.num_nodes).to(self.device)
        if start is None:
            self.current_state = torch.tensor(theta1, dtype=torch.float32).to(self.device)
        else:
            self.current_state = start
        self.T = 0
        return self.current_state, self.current_state, self.T
    
    def step(self, u):
        terminal = False

        if self.T == self.max_T*self.delta_t:
            terminal = True
        assert len(u) == self.num_driver, f"Invalid input, {len(u)},{self.num_driver}"
        # TODO: implement the burger's equation
        
        out = ip(self.T, self.delta_t, self.current_state.cpu().numpy(), u.cpu().numpy())
        
        self.T += self.delta_t
        if not isinstance(out,torch.Tensor):
            out = torch.tensor(out, dtype=torch.float32).to(u.device)
        self.current_state = out
        return torch.matmul(self.C, self.current_state), self.current_state, terminal
    
    def get_state(self):
        return self.current_state
    
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
    
    def calculate_data_driven_control_with_data(self, y_f, U, Y):
        u_dd = torch.pinverse(Y @ torch.pinverse(U)) @ y_f
        u_dd_r = u_dd.reshape(self.max_T, self.num_driver).flip([0])
        u_dd_approx = U @ torch.pinverse(Y) @ y_f
        u_dd_approx_r = u_dd_approx.reshape(self.max_T, self.num_driver).flip([0])
        
        y_f_hat = self.from_actions_to_obs_direct(u_dd_r)
        y_f_hat_approx = self.from_actions_to_obs_direct(u_dd_approx_r)
        
        return u_dd_r, y_f_hat, u_dd_approx_r, y_f_hat_approx

    def calculate_data_driven_control_with_data_complete(self, y_f, y_0, U, Y_bar, Y_f):
        X_0 = Y_bar[:, 0].reshape(Y_bar.shape[0], -1).transpose()
        X_bar = Y_bar[:, 1:].reshape(Y_bar.shape[0], -1).transpose()
        X_f = Y_f.reshape(Y_f.shape[0], -1).transpose()
        U = U.reshape(U.shape[0], -1).transpose()
        # change to torch tensor
        X_0 = torch.tensor(X_0, dtype=torch.float32).to(self.device)
        X_bar = torch.tensor(X_bar, dtype=torch.float32).to(self.device)
        X_f = torch.tensor(X_f, dtype=torch.float32).to(self.device)
        U = torch.tensor(U, dtype=torch.float32).to(self.device)
        y_0 = torch.tensor(y_0, dtype=torch.float32).to(self.device)
        y_f = torch.tensor(y_f, dtype=torch.float32).to(self.device)
        
        u_dd = compute_optimal_input(self.num_nodes, self.num_driver, self.max_T, X_0, U, X_f, y_0, y_f, X_bar)
        u_dd_r = u_dd.reshape(self.max_T, self.num_driver).flip([0])
        y_f_hat = self.from_actions_to_obs_direct(u_dd_r, y_0)
        return u_dd_r, y_f_hat