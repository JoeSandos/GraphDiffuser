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
parser.add_argument('--apply_guide', type=int, default=0)
parser.add_argument('--guide_clean', type=int, default=0)
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
                  summary_writer_name='./runs/'+args.sw_name,
                  resample=args.resample, 
                  use_invdyn=args.use_invdyn, 
                  normalized=args.normalized, 
                  valid_data=val_data, 
                  sigma=args.sigma, 
                  apply_guidance=args.apply_guide, 
                  guide_clean=args.guide_clean)
# print("length of train_loader: ", len(trainer.dataloader)) 


# sampling
# def sample(diffusion, env:LinearEnv, device, resample=False):
#     diffusion.eval()
#     eval_epoch = 16
#     mse_total = 0
#     mse_final_total = 0
#     mse_final_target_total = 0
#     mse_final_target2_total = 0
#     mse_u_total = 0
#     mse_u_target_total = 0
#     for epoch in range(eval_epoch):
#         num_sample = 1
#         with torch.no_grad():
#             if not resample:
#                 y_f = torch.randn(num_sample, 1, env.num_observation).to(device)
#                 y_0 = torch.zeros(num_sample, 1, env.num_observation).to(device)
#                 trajectories, *_ = diffusion(cond={0: y_0, env.max_T: y_f})
#                 trajectories = trajectories.cpu()
                
#                 actions = trajectories[:, :-1, :diffusion.action_dim].squeeze(0) # 1, T, m
#                 action = actions[0, 0]
#                 observations = trajectories[:, 1:, diffusion.action_dim:].squeeze(0) # 1, T, p
            
#             # evaluate the correspondense of actions and observations
#                 env.reset()
#                 obs_from_act = env.from_actions_to_obs(actions)
#                 assert obs_from_act.shape == observations.shape
#                 mse = l1_loss(obs_from_act,observations).item()
#                 print("MSE between obs produced from sampled actions and sampled obs: ", mse)
#                 mse_final = l1_loss(obs_from_act[-1],observations[-1]).item()
#                 print("MSE between final obs produced from sampled actions and sampled obs: ", mse_final)
#                 mse_final_target = l1_loss(obs_from_act[-1], y_f.cpu().squeeze(0)).item()
#                 print("MSE between final obs produced from sampled actions and target final obs: ", mse_final_target)
#                 # mse_final_target2 = mse_loss(observations[-1], y_f.cpu().squeeze(0)).item()
#                 # print("MSE between final sampled obs and target final obs: ", mse_final_target2)
                
#             # evaluate the difference between actions and model-based minumum energy control
#                 u_min, y_f_hat = env.calculate_model_based_control(y_f.cpu().squeeze())
#                 assert u_min.shape == actions.shape
#                 mse_u = l1_loss(u_min, actions).item()
#                 print("MSE between model-based minumum energy control and sampled actions: ", mse_u)
#                 mse_u_target = l1_loss(y_f_hat, y_f.cpu().squeeze()).item()
#                 print("MSE between final obs produced from model-based minumum energy control and target final obs: ", mse_u_target)
#                 print("====================================")
                
#                 mse_total += mse
#                 mse_final_total += mse_final
#                 mse_final_target_total += mse_final_target
#                 # mse_final_target2_total += mse_final_target2
#                 mse_u_total += mse_u
#                 mse_u_target_total += mse_u_target
                
#             else:
#                 raise NotImplementedError
        
    # mse_total /= eval_epoch
    # mse_final_total /= eval_epoch
    # mse_final_target_total /= eval_epoch
    # # mse_final_target2_total /= eval_epoch
    # mse_u_total /= eval_epoch
    # mse_u_target_total /= eval_epoch
    # print("==================Average======================")
    # print("Average MSE between obs produced from sampled actions and sampled obs: ", mse_total)
    # print("Average MSE between final obs produced from sampled actions and sampled obs: ", mse_final_total)
    # print("Average MSE between final obs produced from sampled actions and target final obs: ", mse_final_target_total)
    # print("Average MSE between final sampled obs and target final obs: ", mse_final_target2_total)
    # print("Average MSE between model-based minumum energy control and sampled actions: ", mse_u_total)
    # print("Average MSE between final obs produced from model-based minumum energy control and target final obs: ", mse_u_target_total)
    
        
    # return mse_total, mse_final_total, mse_final_target_total, mse_u_total, mse_u_target_total

# n_epochs = int(n_train_steps // n_steps_per_epoch)

# for i in range(n_epochs):
#     print(f'Epoch {i} / {n_epochs} | {train_savepath}')
if args.sample_use_test:
    trainer.train(n_train_steps=args.n_train_steps, test_data=test_data, no_cond=args.no_cond)
else:
    trainer.train(n_train_steps=args.n_train_steps, no_cond=args.no_cond)
