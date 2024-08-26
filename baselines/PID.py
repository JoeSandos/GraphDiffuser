import sys

sys.path.append('/home/joe/projects/GraphDiffuser')
print(sys.path)
import numpy as np
import torch
import pickle
import networkx as nx
from utils.dataset import TrainData, TrainData2, TrainData_norm, TrainData_norm2
from utils.trainer import Trainer
from torch.utils.data import DataLoader
from torch import nn
from env.env import LinearEnv
from model.diffusion import GaussianDiffusion, GaussianInvDynDiffusion, GaussianDiffusionClassifierGuided
from model.temporal import TemporalUnet
import copy
# tensorboard
# from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt
import pdb
from torch.nn.functional import mse_loss
import os
def set_seeds(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



# set_seeds(44)
# train_ratio = 0.6
# batch_size = int(16)
# lr = 2e-3
# train_savepath = './results/toy_noguide'
# n_train_steps = int(1e5)
# n_steps_per_epoch = int(1e3)
# sw_name = 'sample_without_cond_data_10_3_debug'
# resample = False
# horizon = 8
# sample_use_test = False
# test_ratio = 0.2
# no_cond = True
# data_name = 'erdos_renyi_10_5_5_15_1000'

import argparse

# 创建一个解析器
parser = argparse.ArgumentParser()

# 添加参数
parser.add_argument('--seed', type=int, default=44)
parser.add_argument('--train_ratio', type=float, default=0.2)
parser.add_argument('--valid_ratio', type=float, default=0.1)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--train_savepath', type=str, default='./results/toy_noguide')
parser.add_argument('--n_train_steps', type=int, default=int(4e3))
parser.add_argument('--n_steps_per_epoch', type=int, default=int(2e3))
parser.add_argument('--sw_dir', type=str, default='./runs/')
parser.add_argument('--sw_name', type=str, default='debug1')
parser.add_argument('--resample', type=int, default=0)
parser.add_argument('--horizon', type=int, default=8)
parser.add_argument('--sample_use_test', type=int, default=1)
parser.add_argument('--test_ratio', type=float, default=0.2)
parser.add_argument('--no_cond', type=int, default=0)
parser.add_argument('--data_name', type=str, default='simple_2_1_2_7_1000_sigma_1')
parser.add_argument('--use_invdyn', type=int, default=0)
parser.add_argument('--normalized', type=int, default=0)
parser.add_argument('--pred_eps', type=int, default=0)
parser.add_argument('--sigma', type=float, default=1)
parser.add_argument('--apply_guide', type=int, default=1)
parser.add_argument('--guide_clean', type=int, default=1)
parser.add_argument('--scale', type=float, default=1)
parser.add_argument('--loops', type=int, default=0)
parser.add_argument('--concat', type=int, default=1)
parser.add_argument('--ID', type=int, default=0)


# 解析参数
args = parser.parse_args()
print(args)
set_seeds(args.seed)
# import mse function
from torch.nn.functional import mse_loss, l1_loss
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



with open('data/synthetic_data/'+args.data_name+'.pkl', 'rb') as f:
    pickle_data = pickle.load(f)
    
# data_dict['sys'] = {'A': sys_A, 'B': sys_B, 'C': sys_C}
# data_dict['adj'] = adj
# data_dict['meta_data'] = {'num_nodes': n, 'input_dim': m, 'output_dim': p, 'control_horizon': T, 'num_samples': N, 'num_edges': adj.sum()}
# data_dict['data'] = {'U': U_3d, 'Y_bar': Y_bar_3d, 'Y_f': Y_f_3d}

sys_A, sys_B, sys_C = pickle_data['sys']['A'], pickle_data['sys']['B'], pickle_data['sys']['C']
adj = pickle_data['adj']
n, m, p, T, N = pickle_data['meta_data']['num_nodes'], pickle_data['meta_data']['input_dim'], pickle_data['meta_data']['output_dim'], pickle_data['meta_data']['control_horizon'], pickle_data['meta_data']['num_samples']
U_3d, Y_bar_3d, Y_f_3d = pickle_data['data']['U'], pickle_data['data']['Y_bar'], pickle_data['data']['Y_f']
env = LinearEnv(sys_A, sys_B, sys_C, adj, T, device=device)

class NeuralPID(nn.Module):
    def __init__(self, input_dim=2, output_dim=1):
        super(NeuralPID, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.nn = nn.Sequential(
            nn.Linear(input_dim, 16*input_dim*output_dim), 
            nn.LayerNorm(16*input_dim*output_dim),
            nn.Sigmoid(),
            nn.Linear(16*input_dim*output_dim, 16*input_dim*output_dim),
            nn.LayerNorm(16*input_dim*output_dim),
            nn.Sigmoid(),
            nn.Linear(16*input_dim*output_dim, 3*input_dim*output_dim),  # 输出PID的三个参数：Kp, Ki, Kd
            # nn.Sigmoid(),

        )

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        
        output = self.nn(x)/100
        pid_param = output.view(-1, 3, self.output_dim, self.input_dim)    
        # if pid_param.norm(1)<1e-6:
        #     pdb.set_trace()
        return pid_param



num_train = int(N*args.train_ratio)
num_val = int(N*args.valid_ratio)
num_test = int(N*args.test_ratio)
if args.resample and args.normalized:  
    train_data = TrainData_norm2(U_3d[:num_train], Y_bar_3d[:num_train], Y_f_3d[:num_train], horizon=args.horizon)
    val_data = TrainData_norm2(U_3d[num_train:num_train+num_val], Y_bar_3d[num_train:num_train+num_val], Y_f_3d[num_train:num_train+num_val], horizon=args.horizon)
    
elif args.resample and (not args.normalized):
    train_data = TrainData2(U_3d[:num_train], Y_bar_3d[:num_train], Y_f_3d[:num_train], horizon=args.horizon)
    val_data = TrainData2(U_3d[num_train:num_train+num_val], Y_bar_3d[num_train:num_train+num_val], Y_f_3d[num_train:num_train+num_val], horizon=args.horizon)
    
elif args.normalized:
    train_data = TrainData_norm(U_3d[:num_train], Y_bar_3d[:num_train], Y_f_3d[:num_train])
    val_data = TrainData_norm(U_3d[num_train:num_train+num_val], Y_bar_3d[num_train:num_train+num_val], Y_f_3d[num_train:num_train+num_val])
    
else:
    train_data = TrainData(U_3d[:num_train], Y_bar_3d[:num_train], Y_f_3d[:num_train])
    val_data = TrainData(U_3d[num_train:num_train+num_val], Y_bar_3d[num_train:num_train+num_val], Y_f_3d[num_train:num_train+num_val])

if not args.sample_use_test:
    if args.normalized:
        test_data = TrainData_norm(U_3d[num_train+num_val:num_train+num_val+num_test], Y_bar_3d[num_train+num_val:num_train+num_val+num_test], Y_f_3d[num_train+num_val:num_train+num_val+num_test])
    else:
        test_data = TrainData(U_3d[num_train+num_val:num_train+num_val+num_test], Y_bar_3d[num_train+num_val:num_train+num_val+num_test], Y_f_3d[num_train+num_val:num_train+num_val+num_test])
else:
    if args.normalized:
        test_data = TrainData_norm(U_3d[:num_train], Y_bar_3d[:num_train], Y_f_3d[:num_train])
    else:
        test_data = TrainData(U_3d[:num_train], Y_bar_3d[:num_train], Y_f_3d[:num_train])


def pid_controller(PID=None,error=None,last_error=None,error_sum=None):
    # control=PID[:,0,:]*error+PID[:,1,:]*error_sum+PID[:,2,:]*(error-last_error) 
    # PID[:,0,:] [batch_size,nc,ns]* error [batch_size,ns]=[batch_size,nc]
    control = torch.einsum('bnc,bc->bn', PID[:,0,:], error)+torch.einsum('bnc,bc->bn', PID[:,1,:], error_sum)+torch.einsum('bnc,bc->bn', PID[:,2,:], error-last_error)
    return control

import torch.nn.functional as F
class Controller(nn.Module):
    def __init__(self,model,loss_type= 'l1'):
        super(Controller, self).__init__()
        self.loss_type=loss_type
        self.model=model
        self.last_error=0
        self.error_sum=0
    @property
    def loss_fn(self):
        if self.loss_type == 'l1':
            return F.l1_loss
        elif self.loss_type == 'l2':
            return F.mse_loss
        else:
            raise ValueError(f'invalid loss type {self.loss_type}')
    def forward(self,error):
        PID=self.model(error)
        self.error_sum=self.error_sum+error
        control=pid_controller(PID=PID,error=error,last_error=self.last_error,error_sum=self.error_sum)
        # pdb.set_trace()
        self.last_error=error
        return control

def cycle(dl):
    while True:
        for data in dl:
            yield data
            
import tqdm

class train(object):
    def __init__(
            self,
            controller=None,
            max_training_iters=100,
            Ud=None,
            U0=None,
            max_eval_iters=200,
            lr=1e-4,
            save_iters=20,
            exp_path=None,
            dataset=None,
            device=None,
            env=env
    ):
        self.controller=controller
        self.Ud=Ud
        self.max_training_iters=max_training_iters
        self.max_eval_iters= max_eval_iters
        
        self.U0=U0
        self.save_iters=save_iters
        self.optimizer = torch.optim.Adam(self.controller.parameters(), lr=lr)
        self.exp_path=exp_path
        self.device=device
        self.controller = self.controller.to(self.device)
        self.train_batch_size=1
        dl=DataLoader(dataset, batch_size = 1, shuffle = True)
        self.dl=cycle(dl)
        self.env=env
        self.max_time = self.env.max_T
        self.max_iter_steps=self.max_time
        # self.scheduler = torch.optim.lr_scheduler.StepLR(self.opt, step_size=40000, gamma=0.5)
    def train(self):
        self.controller.model.train()
        loss_list=torch.zeros((self.max_training_iters)).flatten()
        trajectory=None
        data = next(self.dl).trajectories.to(self.device)
        self.U0=data[:,0, -self.env.num_observation:]
        self.Ud=data[:, -1, -self.env.num_observation:]
        print(self.max_training_iters)
        for j in range(self.max_training_iters):
            self.controller.error_sum=0
            self.controller.last_error=0
            loss_sum=0
            data = next(self.dl).trajectories.to(self.device)
            self.env.reset()
            
            for i in range(self.max_iter_steps):
                if i==0:
                    
                    data = next(self.dl).trajectories.to(self.device)
                    self.U0=data[:,0, -self.env.num_observation:]
                    self.Ud=data[:, -1, -self.env.num_observation:]
                    ut=self.U0
                    control=self.controller(ut-self.Ud)
                    control=control.reshape(control.shape[0],-1)
                else:
                    # u0=ut.clone().detach()
                    control=self.controller(ut-self.Ud)
                    control=control.reshape(control.shape[0],-1)##[batch_size,1,ns]
                #burgers_numeric_solve 
                #u0 [batch_size,ns]
                # pdb.set_trace()
                next_state,_,_=self.env.step(control.squeeze(0))                
                ut=next_state
                loss=self.controller.loss_fn(self.Ud,ut)
                loss=loss.mean()
                if not torch.isnan(ut).any().item():
                    loss_sum=loss_sum+loss
                # if i%self.learn_steps==0:
                #     # if torch.isnan(loss).any().item()
                #     if i==10:
                #         print(f"loss_sum :{loss_sum},loss :{loss}")
                #     loss_sum=0
            loss_sum.backward()
            # print('grad')
            self.optimizer.step()
            self.optimizer.zero_grad()
            loss_list[j]=loss_sum.clone().detach()
            if j%(self.save_iters-1)==0:
                print(f"training {j} iters,loss_sum: {loss_sum}")
                torch.save(self.controller.model.state_dict(), self.exp_path+f'/model_weights-{j}_{loss_sum}.pth')
                self.eval()
        numpy_data = loss_list.to("cpu").detach().numpy()
        plt.figure()
        x=np.linspace(0,len(numpy_data),len(numpy_data))
        plt.plot(x, numpy_data)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('loss_list')
        plt.grid(True)
        plt.savefig(self.exp_path+"/loss_list.png")
        # plt.show()
        np.save(self.exp_path+'/loss_lis.npy', numpy_data)
        
    @torch.no_grad()
    def eval(self,weigh_path=None):
        if weigh_path is not None:
            self.controller.model.load_state_dict(torch.load(weigh_path))
        self.controller.model.eval()
        loss_sum=0
        print('begin eval')
        mean_energy_model_based = 0
        mean_energy = 0
        for j in tqdm.trange(self.max_eval_iters):
            # print(f"eval {j} iters")
            self.controller.error_sum=0
            self.controller.last_error=0
            self.env.reset()
            
            self.Ud = torch.rand((1,self.env.num_observation)).to(self.device)
            controls = []
            control_model_based, *_ = env.calculate_model_based_control(self.Ud.squeeze(0))
            energy_model_based = torch.norm(control_model_based.squeeze(), p=2)
            trajectory=[]
            trajectory.append
            for i in range(self.max_iter_steps):
                if i==0:
                    ut=torch.zeros((1,self.env.num_observation)).to(self.device)
                    control=self.controller(ut-self.Ud)
                    control=control.reshape(control.shape[0],-1)
                    trajectory.append(ut.squeeze().unsqueeze(0))
                else:
                    # u0=ut.clone().detach()
                    control=self.controller(ut-self.Ud)
                    control=control.reshape(control.shape[0],-1)
                next_state,_,_=self.env.step(control.squeeze(0))
                trajectory.append(next_state.squeeze().unsqueeze(0))
                ut=next_state
                loss=self.controller.loss_fn(self.Ud,ut)
                controls.append(control)
                
                # loss=loss.mean()
            controls = torch.stack(controls,dim=0).squeeze()
            control_energy = torch.norm(controls.squeeze(), p=2)
            trajectory = torch.stack(trajectory,dim=0)
            end_loss = mse_loss(self.Ud,ut)
            # print(end_loss)
            # if not torch.isnan(ut).any().item():
            loss_sum=loss_sum+end_loss
            
            mean_energy_model_based+=energy_model_based
            mean_energy += control_energy
        
            # print("controls",controls.squeeze())
            # print("model based controls",control_model_based.squeeze())
            # print("target",self.Ud.squeeze())
            # print('traj:',trajectory)
        mean_loss=loss_sum/self.max_eval_iters
        # mean_energy_model_based/=self.max_eval_iters
        # mean_energy/= self.max_eval_iters
        print(f"mean_loss:{mean_loss}, mean_energy:{mean_energy}, mean_energy_model_based:{mean_energy_model_based}, energy_relative:{mean_energy/mean_energy_model_based}")
        # pdb.set_trace()

        print('end eval')
        # write at the end of the yaml file
        with open(self.exp_path+f'/results_{args.ID}.yaml', 'a') as file:
            file.write(f'mean_loss:{mean_loss}, mean_energy:{mean_energy}, mean_energy_model_based:{mean_energy_model_based}, energy_relative:{mean_energy/mean_energy_model_based}\n')
        self.controller.model.train()
    
trainer = train(
    controller=Controller(NeuralPID(input_dim=p, output_dim=m),loss_type='l2'),
    max_training_iters=args.n_train_steps,
    Ud=None,
    U0=None,
    max_eval_iters=200,
    lr=args.lr,
    save_iters=40,
    exp_path=args.train_savepath,
    dataset=train_data,
    device=device,
    env=env
)
os.makedirs(args.train_savepath, exist_ok=True)
# if yaml exists, raise error
if os.path.exists(args.train_savepath+f'/results_{args.ID}.yaml'):
    raise ValueError('yaml file already exists')
with open(args.train_savepath+f'/results_{args.ID}.yaml', 'w') as file:
    file.write(f'args:{args}\n')
trainer.train()