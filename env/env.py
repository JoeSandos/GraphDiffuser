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
    def __init__(self, A, B, C, adj, T):
        super().__init__()
        assert type(A)==type(B)==type(C)==type(adj)
        if type(A)==np.ndarray or type(A)==np.matrix:
            self.A = torch.tensor(A, dtype=torch.float32)
            self.B = torch.tensor(B, dtype=torch.float32)
            self.C = torch.tensor(C, dtype=torch.float32)
            self.adj = torch.tensor(adj, dtype=torch.float32)
        else:
            self.A = A
            self.B = B
            self.C = C
            self.adj = adj
        self.num_edges = torch.sum(self.adj)

        self.max_T = T
        self.num_nodes = A.shape[0]
        self.num_driver = B.shape[1]
        self.num_observation = C.shape[0]
        self.reset()
        self.calculate_C_0()
        
    def reset(self, state=None):
        if state is None:
            self.x = torch.zeros(self.num_nodes)
        else:
            assert len(state) == self.num_nodes, "Invalid state"
            self.x = state
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
    
    def from_actions_to_obs(self, actions):
        assert len(actions) == self.max_T, "Invalid actions"
        observations = []
        for a in actions:
            obs, _, _ = self.step(a)
            observations.append(obs)
        return torch.stack(observations)
    
    def from_actions_to_obs_direct(self, actions):

        H = torch.zeros((self.num_observation * self.max_T, self.num_driver * self.max_T))

        for r in range(1, self.max_T + 1):
            for k in range(1, self.max_T + 1):
                if k > self.max_T - r:
                    H[(r-1)*self.num_observation:r*self.num_observation, (k-1)*self.num_driver:k*self.num_driver] = \
                        torch.matmul(self.C, torch.matrix_power(self.A, r-self.max_T+k-1)).matmul(self.B)
        actions_c = torch.flip(actions, [0]) #NOTE: u vector is originally inversed (i.e. u(T-1), u(T-2),...u(0))
        U = actions_c.reshape(-1) 
        Y = torch.matmul(H, U)
        Y = Y.reshape(self.max_T, self.num_observation)
        return Y

    def calculate_C_0(self):
        C_o = torch.zeros(self.num_nodes, self.num_driver * self.max_T)
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
        u = torch.pinverse(C_o) @ y_f # shape: (num_driver * max_T, num_observation) @ (num_observation, 1) = (num_driver * max_T, 1)
        y_f_hat = torch.matmul(C_o, u) # shape: (num_nodes, num_driver * max_T) @ (num_driver * max_T, 1) = (num_nodes, 1)
        u_reshape = u.reshape(self.max_T, self.num_driver).flip([0])
        return u_reshape, y_f_hat
    
    def calculate_data_driven_control(self, y_f):
        U = torch.randn(self.num_driver * self.max_T, 100)
        Y = torch.matmul(self.C_o, U)
        u_dd = torch.pinverse(Y @ torch.pinverse(U)) @ y_f
        u_dd_r = u_dd.reshape(self.max_T, self.num_driver).flip([0])
        u_dd_approx = U @ torch.pinverse(Y) @ y_f
        u_dd_approx_r = u_dd_approx.reshape(self.max_T, self.num_driver).flip([0])
        y_f_hat = torch.matmul(self.C_o, u_dd)
        y_f_hat_approx = torch.matmul(self.C_o, u_dd_approx)
        
        return u_dd_r, y_f_hat, u_dd_approx_r, y_f_hat_approx
        