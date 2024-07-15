import numpy as np
import torch
import pickle
import networkx as nx
from utils.dataset import TrainData, TrainData2, TrainData_norm, TrainData_norm2
from utils.trainer import Trainer
from torch.utils.data import DataLoader
from env.env import LinearEnv
from model.diffusion import GaussianDiffusion, GaussianInvDynDiffusion, GaussianDiffusionClassifierGuided
from model.temporal import TemporalUnet

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
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--train_savepath', type=str, default='./results/toy_noguide')
parser.add_argument('--n_train_steps', type=int, default=int(1e4))
parser.add_argument('--n_steps_per_epoch', type=int, default=int(2e3))
parser.add_argument('--sw_dir', type=str, default='./runs/')
parser.add_argument('--sw_name', type=str, default='debug')
parser.add_argument('--resample', type=int, default=0)
parser.add_argument('--horizon', type=int, default=8)
parser.add_argument('--sample_use_test', type=int, default=1)
parser.add_argument('--test_ratio', type=float, default=0.2)
parser.add_argument('--no_cond', type=int, default=0)
parser.add_argument('--data_name', type=str, default='simple_2_1_2_7_1000_sigma_1')
parser.add_argument('--use_invdyn', type=int, default=0)
parser.add_argument('--normalized', type=int, default=1)
parser.add_argument('--pred_eps', type=int, default=0)
parser.add_argument('--sigma', type=float, default=1)
parser.add_argument('--apply_guide', type=int, default=1)
parser.add_argument('--guide_clean', type=int, default=1)
parser.add_argument('--scale', type=float, default=1)


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
env = LinearEnv(sys_A, sys_B, sys_C, adj, T)

if args.apply_guide:
    assert not args.use_invdyn
    model = TemporalUnet(transition_dim=m+p, cond_dim=0, dim=32, dim_mults=(1, 4, 8), attention=False)
    if not args.resample:
        diffusion = GaussianDiffusionClassifierGuided(model, horizon=env.max_T+1, observation_dim=p, action_dim=m, n_timesteps=64, loss_type='l2', clip_denoised=False, predict_epsilon=args.pred_eps, action_weight=1.0, loss_discount=1.0, loss_weights=None, scale=args.scale)
    else:
        raise NotImplementedError
elif args.use_invdyn:
    model = TemporalUnet(transition_dim=p, cond_dim=0, dim=32, dim_mults=(1, 4, 8), attention=False)
    if args.resample:
        diffusion = GaussianInvDynDiffusion(model, horizon=args.horizon, observation_dim=p, action_dim=m, n_timesteps=64, loss_type='l2', clip_denoised=False, predict_epsilon=args.pred_eps, action_weight=1.0, loss_discount=1.0, loss_weights=None)
    else:
        diffusion = GaussianInvDynDiffusion(model, horizon=env.max_T+1, observation_dim=p, action_dim=m, n_timesteps=64, loss_type='l2', clip_denoised=False, predict_epsilon=args.pred_eps, action_weight=1.0, loss_discount=1.0, loss_weights=None)
else:
    model = TemporalUnet(transition_dim=m+p, cond_dim=0, dim=32, dim_mults=(1, 4, 8), attention=False)
    if args.resample:
        diffusion = GaussianDiffusion(model, horizon=args.horizon, observation_dim=p, action_dim=m, n_timesteps=64, loss_type='l2', clip_denoised=False, predict_epsilon=args.pred_eps, action_weight=1.0, loss_discount=1.0, loss_weights=None)
    else:
        diffusion = GaussianDiffusion(model, horizon=env.max_T+1, observation_dim=p, action_dim=m, n_timesteps=64, loss_type='l2', clip_denoised=False, predict_epsilon=args.pred_eps, action_weight=1.0, loss_discount=1.0, loss_weights=None)
diffusion = diffusion.to(device)
# split training and testing

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

# dataloader

print('batch_size: ', args.batch_size)
# train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
trainer = Trainer(diffusion, 
                  train_data, 
                  env=env, 
                  device=device, 
                  train_batch_size=args.batch_size, 
                  train_lr=args.lr, 
                  results_folder=args.train_savepath, 
                  summary_writer_name=args.sw_dir + args.sw_name,
                  resample=args.resample, 
                  use_invdyn=args.use_invdyn, 
                  normalized=args.normalized, 
                  valid_data=val_data, 
                  sigma=args.sigma, 
                  apply_guidance=args.apply_guide, 
                  guide_clean=args.guide_clean)


if args.sample_use_test:
    trainer.train(n_train_steps=args.n_train_steps, test_data=test_data, no_cond=args.no_cond)
else:
    trainer.train(n_train_steps=args.n_train_steps, no_cond=args.no_cond)
