import numpy as np
import torch
from utils.dataset import *
from utils.trainer import Trainer
from utils.utils import load_data_and_env
from model.diffusion import *
from model.temporal import *
import copy
import os
import pdb
# set thread number
def set_cpu_num(cpu_num):
    if cpu_num <= 0: return
    os.environ ['OMP_NUM_THREADS'] = str(cpu_num)
    os.environ ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
    os.environ ['MKL_NUM_THREADS'] = str(cpu_num)
    os.environ ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
    os.environ ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
    torch.set_num_threads(cpu_num)

def set_seeds(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

import argparse

# 创建一个解析器
parser = argparse.ArgumentParser()

# 添加参数
parser.add_argument('--seed', type=int, default=44) # random seed
parser.add_argument('--train_ratio', type=float, default=0.2) # train data ratio
parser.add_argument('--valid_ratio', type=float, default=0.001) # valid data ratio
parser.add_argument('--test_ratio', type=float, default=0.0005)  # test data ratio
parser.add_argument('--batch_size', type=int, default=16) # batch size
parser.add_argument('--lr', type=float, default=5e-3) # learning rate
parser.add_argument('--train_savepath', type=str, default='./results/train_save/') # path to save training model
parser.add_argument('--n_train_steps', type=int, default=int(100)) # number of training steps
parser.add_argument('--n_steps_per_epoch', type=int, default=int(1e3)) # number of steps per epoch
parser.add_argument('--sw_dir', type=str, default='./runs/kuramoto/') # tensorboard log directory
parser.add_argument('--sw_name', type=str, default='kuramoto') # tensorboard log name
parser.add_argument('--data_name', type=str, default='kuramoto_8_8_15_100_2_sigma=2') # dataset name
parser.add_argument('--normalized', type=int, default=1) # whether to normalize the data, default is True
parser.add_argument('--pred_eps', type=int, default=0) # whether to predict epsilon, default is False
parser.add_argument('--sigma', type=float, default=1) # sigma for kuramoto, ignore
parser.add_argument('--apply_guide', type=int, default=1) # whether to apply guidance, default is True
parser.add_argument('--guide_clean', type=int, default=0) # whether to apply clean guidance, default is False
parser.add_argument('--scale', type=float, default=0.05) # scale for classifier guidance
parser.add_argument('--loops', type=int, default=2) # number of self-fintune loops
parser.add_argument('--concat', type=int, default=1) # whether to concat the resampled data with the original data
parser.add_argument('--concat_ratio', type=float, default=0.5) # ratio of the resampled data to the original data
parser.add_argument('--resample_num', type=int, default=10) # number of resampled data
parser.add_argument('--regen', type=int, default=1) # whether to regenerate the resampled data with true env
parser.add_argument('--mixup', type=int, default=0) # whether to use mixup, ignore
parser.add_argument('--use_attn', type=int, default=0) # ignore
parser.add_argument('--use_invdyn', type=int, default=1) # whether to use invdyn, default is True, ignore
parser.add_argument('--use_end', type=int, default=1) # whether to use linear temporal unet, default is True
parser.add_argument('--use_lambda', type=int, default=0) # ignore
parser.add_argument('--free_guide', type=int, default=0) # whether to use free guidance, default is False
parser.add_argument('--use_end_second', type=int, default=1) # whether to use quadratic temporal unet, default is True
parser.add_argument('--repaint', type=int, default=0) # ignore
parser.add_argument('--sample_use_test', type=int, default=1) # whether to use test data for sampling, default is True
parser.add_argument('--use_clustering', type=int, default=0) # ignore
parser.add_argument('--use_multi_cond', type=int, default=0) # ignore
parser.add_argument('--n_timesteps', type=int, default=128) # number of diffusion timesteps
parser.add_argument('--train_inv_first', type=int, default=1) # whether to train invdyn first
parser.add_argument('--no_end', type=int, default=0) # whether use ordinary temporal unet, default is False
# 解析参数

args = parser.parse_args()
print(args)
set_seeds(args.seed)
set_cpu_num(8)
# import mse function
from torch.nn.functional import mse_loss, l1_loss
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

env, train_data, val_data, test_data, U_3d, Y_bar_3d, Y_f_3d = load_data_and_env(args)
p = env.num_observation
m = env.num_driver
n = env.num_nodes
T = env.max_T
dim = max(32, p)
print(dim)
if args.free_guide:
    if args.use_invdyn:
        if args.use_end_second:
            if not args.use_multi_cond:
                model = EndTemporalUnetFreeSecond2(transition_dim=p, cond_dim=p, denoiser_cond_dim=1, dim=dim, dim_mults=(1,2,4), attention=False, use_cond=args.free_guide)
            else:
                model = EndTemporalUnetFreeSecond2(transition_dim=p, cond_dim=p, denoiser_cond_dim=T+1, dim=dim, dim_mults=(1,2,4), attention=False, use_cond=args.free_guide)
                
        elif args.use_end:
            if not args.use_multi_cond:
                model = EndTemporalUnetFreeFirst(transition_dim=p, cond_dim=p, denoiser_cond_dim=1, dim=32, dim_mults=(1, 2,4,4), attention=False, use_cond=args.free_guide)
            else:
                model = EndTemporalUnetFreeFirst(transition_dim=p, cond_dim=p, denoiser_cond_dim=T+1, dim=32, dim_mults=(1, 2,4,4), attention=False, use_cond=args.free_guide)
        elif args.no_end:
            if not args.use_multi_cond:
                model = EndTemporalUnetFreeNone(transition_dim=p, cond_dim=p, denoiser_cond_dim=1, dim=32, dim_mults=(1, 4, 4), attention=False, use_cond=args.free_guide)
            else:
                model = EndTemporalUnetFreeNone(transition_dim=p, cond_dim=p, denoiser_cond_dim=T+1, dim=32, dim_mults=(1, 4, 4), attention=False, use_cond=args.free_guide)           
        else:
            if not args.use_multi_cond:
                model = EndTemporalUnetFreeHigher2(transition_dim=p, cond_dim=p, denoiser_cond_dim=1, dim=32, dim_mults=(1, 2, 4), attention=False, use_cond=args.free_guide)
            else:
                model = EndTemporalUnetFreeHigher2(transition_dim=p, cond_dim=p, denoiser_cond_dim=T+1, dim=32, dim_mults=(1, 2, 4), attention=False, use_cond=args.free_guide)
             
    else:
        if args.use_end_second:
            if not args.use_multi_cond:
                model = EndTemporalUnetFreeSecond2(transition_dim=p+m, cond_dim=p, denoiser_cond_dim=1, dim=dim, dim_mults=(1, 2, 4), attention=False, use_cond=args.free_guide)
            else:
                model = EndTemporalUnetFreeSecond2(transition_dim=p+m, cond_dim=p, denoiser_cond_dim=T+1, dim=dim, dim_mults=(1, 2, 4), attention=False, use_cond=args.free_guide)
        elif args.use_end:
            raise NotImplementedError
        
    # if not args.resample:
    diffusion = GaussianDiffusionClassifierFree(model, horizon=env.max_T+1, observation_dim=p, action_dim=m, n_timesteps=args.n_timesteps, loss_type='l2', clip_denoised=False, predict_epsilon=args.pred_eps, action_weight=1., loss_discount=1.0, loss_weights=None, scale=args.scale, inv_dyn=args.use_invdyn, use_lambda=args.use_lambda, repaint=args.repaint, env=env, )
                                                    # node_dim=n, node_f_dim=((p+m)//n))

elif args.apply_guide:
    if args.use_invdyn:
        if args.use_end_second:
            if not args.use_multi_cond:
                model = EndTemporalUnetGuideSecond2(transition_dim=p, cond_dim=p, denoiser_cond_dim=1, dim=dim, dim_mults=(1, 2), attention=False)
            else:
                model = EndTemporalUnetGuideSecond2(transition_dim=p, cond_dim=p, denoiser_cond_dim=T+1, dim=dim, dim_mults=(1, 2), attention=False)
        
        elif args.use_end:
            if not args.use_multi_cond:
                model = EndTemporalUnetGuideFirst2(transition_dim=p, cond_dim=p, denoiser_cond_dim=1, dim=dim, dim_mults=(1, 2,4), attention=False)
            else:
                model = EndTemporalUnetGuideFirst2(transition_dim=p, cond_dim=p, denoiser_cond_dim=T+1, dim=dim, dim_mults=(1, 2,4), attention=False)
        
        else:
            if not args.use_multi_cond:
                model = EndTemporalUnetGuideNone2(transition_dim=p, cond_dim=p, denoiser_cond_dim=1, dim=dim, dim_mults=(1, 2,4), attention=False)
            else:
                model = EndTemporalUnetGuideNone2(transition_dim=p, cond_dim=p, denoiser_cond_dim=T+1, dim=dim, dim_mults=(1, 2,4), attention=False)

    else:
        raise NotImplementedError
    # if not args.resample:
    diffusion = GaussianDiffusionClassifierGuidance(model, horizon=env.max_T+1, observation_dim=p, action_dim=m, n_timesteps=args.n_timesteps, loss_type='l2', clip_denoised=False, predict_epsilon=args.pred_eps, action_weight=1., loss_discount=1.0, loss_weights=None, scale=args.scale, inv_dyn=args.use_invdyn, use_lambda=args.use_lambda, repaint=args.repaint, env=env, inv_hid=64)


diffusion = diffusion.to(device)


print('batch_size: ', args.batch_size)
trainer = Trainer(diffusion, 
                  train_data, 
                  env=env, 
                  device=device, 
                  train_batch_size=args.batch_size, 
                  train_lr=args.lr, 
                  results_folder=args.train_savepath, 
                  summary_writer_name=args.sw_dir + args.sw_name,
                  use_invdyn=args.use_invdyn, 
                  normalized=args.normalized, 
                  valid_data=val_data, 
                  sigma=args.sigma, 
                  apply_guidance=args.apply_guide, 
                  guide_clean=args.guide_clean,
                  kuramoto=True,
                  mixup=args.mixup,
                  )

# write args to tensorboard
for key, value in vars(args).items():
    trainer.writer.add_text(key, str(value))

if args.train_inv_first:
    print('==============start training invdyn===========')
    step = trainer.step
    if args.sample_use_test:
        trainer.train(n_train_steps=args.n_train_steps, test_data=test_data,  train_inv=True)
    else:
        trainer.train(n_train_steps=args.n_train_steps,  train_inv=True)
    trainer.step = step


print('==============start training diffusion===========')
if args.sample_use_test:
    trainer.train(n_train_steps=args.n_train_steps, test_data=test_data)
else:
    trainer.train(n_train_steps=args.n_train_steps)


for i in range(args.loops):
    print('===================start resampling===================')
    with torch.no_grad():
        if 1:
            
            samples_U, samples_Y_bar, samples_Y_f = trainer.sample_tensors(args=args, sample_num=args.resample_num, test_data=test_data, use_invdyn=args.use_invdyn)
        else:
            samples_U = []
            samples_Y_bar = []
            samples_Y_f = []
            for s in tqdm(range(args.resample_num)):
                sample_U, sample_Y_bar, sample_Y_f = trainer.sample_tensors(args=args, sample_num=args.resample_num, test_data=test_data, use_invdyn=args.use_invdyn)
                samples_U.append(sample_U)
                samples_Y_bar.append(sample_Y_bar)
                samples_Y_f.append(sample_Y_f)
                
            samples_U = torch.cat(samples_U, dim=0)
            samples_Y_bar = torch.cat(samples_Y_bar, dim=0)
            samples_Y_f = torch.cat(samples_Y_f, dim=0)
    # samples_U, samples_Y_bar, samples_Y_f = trainer.sample_tensors(args=args, sample_num=args.resample_num, test_data=None)
    
    print('samples_U shape:', samples_U.shape, 'samples_Y_bar shape:', samples_Y_bar.shape, 'samples_Y_f shape:', samples_Y_f.shape)
    
    if args.regen:
        env.reset()
        sample_Y_start = samples_Y_bar[:,0:1]
        samples_Y_bar_list = []
        samples_Y_f_list = []
        for j,actions in enumerate(samples_U):
            observations = env.from_actions_to_obs_direct(actions, start=samples_Y_bar[j,0])
            samples_Y_bar_list.append(observations[:-1])
            samples_Y_f_list.append(observations[-1])
        samples_Y_bar = torch.stack(samples_Y_bar_list)
        samples_Y_bar = torch.cat([sample_Y_start, samples_Y_bar], dim=1)
        samples_Y_f = torch.stack(samples_Y_f_list)
        print('regenerated samples, shape:', samples_Y_bar.shape, samples_Y_f.shape)
    
    # samples_U = samples_U.flip(1) # u(T-1), u(T-2),...,u(0)
    
    if args.concat:
    # concat half of the samples with the original data
        length = samples_U.shape[0]
        num_new_samples = int(length*args.concat_ratio)
        # length = length//2
        assert length > 0
        length_of_orginal = length-num_new_samples
        
        samples_U = np.concatenate([U_3d[:length_of_orginal], samples_U[:num_new_samples]], axis=0)
        samples_Y_bar = np.concatenate([Y_bar_3d[:length_of_orginal], samples_Y_bar[:num_new_samples]], axis=0)
        samples_Y_f = np.concatenate([Y_f_3d[:length_of_orginal], samples_Y_f[:num_new_samples]], axis=0)
        
    # 随机打乱
    index = np.random.permutation(samples_U.shape[0])
    samples_U = samples_U[index]
    samples_Y_bar = samples_Y_bar[index]
    samples_Y_f = samples_Y_f[index]
    
    if args.normalized:
        if args.free_guide:
            train_data = TrainData_norm_free(samples_U[:], samples_Y_bar[:], samples_Y_f[:], zero_pad=False, train=True, use_clustering=args.use_clustering, use_multi_cond=args.use_multi_cond)
            # val_data = TrainData_norm_free(samples_U[num_train:num_train+num_val], samples_Y_bar[num_train:num_train+num_val], samples_Y_f[num_train:num_train+num_val], kuramoto=True, train=True, use_clustering=args.use_clustering, use_smoothness=args.use_smoothness)
        else:
            train_data = TrainData_norm(samples_U[:], samples_Y_bar[:], samples_Y_f[:], zero_pad=False)
            
    else:
        raise NotImplementedError
    trainer.renew_dataset(train_data)
    if i==0:
        for param_group in trainer.optimizer.param_groups:
            param_group['lr'] = param_group['lr'] / 10
    print('===================finish renewing===================')
    print('==============restart training===========')
    
    if args.train_inv_first:
        try:
            step = trainer.step
            if args.sample_use_test:
                trainer.train(n_train_steps=args.n_train_steps, test_data=test_data,  train_inv=True)
            else:
                trainer.train(n_train_steps=args.n_train_steps,  train_inv=True)
            trainer.step = step
        except:
            print('==========directly train diffusion==========')

    if args.sample_use_test:
        trainer.train(n_train_steps=args.n_train_steps, test_data=test_data)
    else:
        trainer.train(n_train_steps=args.n_train_steps)