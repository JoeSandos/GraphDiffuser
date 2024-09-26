import os
import copy
import numpy as np
import torch
import einops
import pdb
import time
from .arrays import batch_to_device, to_np, to_device, apply_dict
from torch.nn.functional import mse_loss, mse_loss
# from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter
from utils.distance import MMDLoss
from model.temporal import LossFunction_noparams
from utils.dataset import *
import matplotlib.pyplot as plt
from collections import namedtuple
Batch = namedtuple('Batch', 'trajectories conditions')
Batch2 = namedtuple('Batch2', 'trajectories conditions denoiser_conditions')

test_data_global = []

class Timer:

	def __init__(self):
		self._start = time.time()

	def __call__(self, reset=True):
		now = time.time()
		diff = now - self._start
		if reset:
			self._start = now
		return diff

def cycle(dl):
    while True:
        for data in dl:
            yield data

class EMA():
    '''
        empirical moving average
    '''
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        dataset,
        env,
        device,
        ema_decay=0.995,
        train_batch_size=32,
        train_lr=2e-4,
        gradient_accumulate_every=1,
        step_start_ema=2000,
        update_ema_every=10,
        log_freq=400,
        sample_freq=1000,
        save_freq=1000,
        label_freq=2000,
        results_folder='./results',
        n_reference=8,
        n_samples=16,
        resample = False,
        use_invdyn = False,
        normalized = False,
        summary_writer_name= None,
        valid_data = None, 
        sigma = 1,
        apply_guidance = False,
        guide_clean= False,
        kuramoto = False,
        mixup = False
    ):
        super().__init__()
        self.model = diffusion_model
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every

        self.step_start_ema = step_start_ema
        self.log_freq = log_freq
        self.sample_freq = sample_freq
        self.save_freq = save_freq
        self.label_freq = label_freq
        self.sigma = sigma # sample sigma when test data ==None
        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.kuramoto = kuramoto
        self.dataset = dataset
        self.dataloader = cycle(torch.utils.data.DataLoader(
            self.dataset, batch_size=train_batch_size, num_workers=1, shuffle=True, pin_memory=True
        ))
        self.mixup = mixup
        if mixup:
            self.dataloader2 = cycle(torch.utils.data.DataLoader(
                self.dataset, batch_size=train_batch_size, num_workers=1, shuffle=True, pin_memory=True
            ))
        
        if valid_data:
            self.valid_dataloader = cycle(torch.utils.data.DataLoader(
                valid_data, batch_size=train_batch_size, num_workers=1, shuffle=True, pin_memory=True
            ))
        self.optimizer = torch.optim.Adam(diffusion_model.parameters(), lr=train_lr)
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=10, \
		verbose=False, threshold=1e-4, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-6)

        self.use_invdyn = use_invdyn
        self.normalized = normalized
        self.logdir = results_folder
        self.n_reference = n_reference
        self.n_samples = n_samples
        self.resample = resample
        self.env = env
        self.device = device
        self.apply_guidance = apply_guidance
        self.guide_clean = guide_clean

        self.reset_parameters()
        self.step = 0
        if summary_writer_name is not None:
            self.writer = SummaryWriter(summary_writer_name)
        else:
            self.writer = SummaryWriter()

    def renew_dataset(self, dataset):
        self.dataset = dataset
        self.dataloader = cycle(torch.utils.data.DataLoader(
            self.dataset, batch_size=self.batch_size, num_workers=1, shuffle=True, pin_memory=True
        ))
        if self.mixup:
            self.dataloader2 = cycle(torch.utils.data.DataLoader(
                self.dataset, batch_size=self.batch_size, num_workers=1, shuffle=True, pin_memory=True
            ))
    def renew_optimizer(self, train_lr):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=train_lr)
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=10, \
		verbose=False, threshold=1e-4, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-6)

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    #-----------------------------------------------------------------------------#
    #------------------------------------ api ------------------------------------#
    #-----------------------------------------------------------------------------#

    def train(self, n_train_steps, test_data=None, no_cond=False):
        self.model.train()
        self.ema_model.train()
        
        
        timer = Timer()
        # if self.mixup:
        #     mixup_activate = 0
        for step in range(n_train_steps):
            # sample for eval
            # if self.step == 0 and self.sample_freq:
            #     self.sample(self.n_reference,self.resample, test_data, no_cond)

            if self.sample_freq and self.step % self.sample_freq == 0:
                if self.apply_guidance:
                    self.sample_guided(self.n_samples,self.resample, test_data, no_cond, use_invdyn=self.use_invdyn)
                # elif self.use_invdyn:
                #     self.sample_invdyn(self.n_samples,self.resample, test_data, no_cond)
                else:
                    self.sample(self.n_samples,self.resample, test_data, no_cond)
                
            # train
            self.model.train()
            self.ema_model.train()
            for i in range(self.gradient_accumulate_every):
                batch = next(self.dataloader)
                batch = batch_to_device(batch, device=self.device)
                if self.mixup:
                    # sample mixup_activate from bernoulli distribution
                    mixup_activate = np.random.binomial(1, 0.5)
                    if mixup_activate:
                        batch2 = next(self.dataloader2)
                        batch2 = batch_to_device(batch2, device=self.device)
                        if batch.trajectories.shape[0] != batch2.trajectories.shape[0]:
                            continue
                        alpha = np.random.beta(0.5, 0.5)
                        trajectories = alpha * batch.trajectories + (1 - alpha) * batch2.trajectories
                        cond = {}
                        cond[0]= alpha * batch.conditions[0] + (1 - alpha) * batch2.conditions[0]
                        cond[self.dataset.horizon-1] = alpha * batch.conditions[self.dataset.horizon-1] + (1 - alpha) * batch2.conditions[self.dataset.horizon-1]
                        if isinstance(self.dataset, TrainData_norm_free):
                            denoiser_cond = alpha * batch.denoiser_conditions + (1 - alpha) * batch2.denoiser_conditions
                    
                            batch = Batch2(trajectories, cond, denoiser_cond)
                        else:
                            batch = Batch(trajectories, cond)
                    # mixup_activate += 1
                        
                # pdb.set_trace()
                loss, infos = self.model.loss(*batch)
                loss = loss / self.gradient_accumulate_every
                loss.backward()

            self.optimizer.step()
            self.optimizer.zero_grad()
            self.writer.add_scalar('Train/train_loss', loss, self.step)
            for key, val in infos.items():
                self.writer.add_scalar(f'Train/{key}', val, self.step)
            
            if self.step % self.update_ema_every == 0:
                self.step_ema()

            # save and log
            if self.step % self.save_freq == 0:
                label = self.step // self.label_freq * self.label_freq
                self.save(label)

            if self.step % self.log_freq == 0:
                self.valid()
                infos_str = ' | '.join([f'{key}: {val:8.8f}' for key, val in infos.items()])
                print(f'{self.step}: {loss:8.4f} | {infos_str} | t: {timer():8.4f} | lr: {self.optimizer.param_groups[0]["lr"]:8.8f}', flush=True)
                self.writer.add_scalar('Train/lr', self.optimizer.param_groups[0]["lr"], self.step)
                self.scheduler.step(loss)
                
            self.step += 1
        # if self.apply_guidance:
        #     self.sample_guided(self.n_samples,self.resample, test_data, no_cond, draw_traj=True, use_invdyn=self.use_invdyn)
        # # elif self.use_invdyn:
        # #     self.sample_invdyn(self.n_samples,self.resample, test_data, no_cond, draw_traj=True)
        # else:
        #     pass
            # self.sample(self.n_samples,self.resample, test_data, no_cond, draw_traj=True)
        
    def valid(self):
        self.model.eval()
        self.ema_model.eval()
        # for i in range(10):
        batch = next(self.valid_dataloader)
        batch = batch_to_device(batch, device=self.device)
        loss, infos = self.model.loss(*batch)
        self.writer.add_scalar('Valid/valid_loss', loss, self.step)
        for key, val in infos.items():
            self.writer.add_scalar(f'Valid/{key}', val, self.step)
            print(f'{key}: {val:8.8f}')
            
        self.model.train()
        self.ema_model.train()
    
    def save(self, epoch):
        '''
            saves model and ema to disk;
            syncs to storage bucket if a bucket is specified
        '''
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema_model.state_dict()
        }
        if not os.path.exists(self.logdir):
            os.makedirs(self.logdir)
        savepath = os.path.join(self.logdir, f'state_{epoch}.pt')

            
        torch.save(data, savepath)
        print(f'[ utils/training ] Saved model to {savepath}', flush=True)


    def load(self, epoch):
        '''
            loads model and ema from disk
        '''
        loadpath = os.path.join(self.logdir, f'state_{epoch}.pt')
        data = torch.load(loadpath)

        self.step = data['step']
        self.model.load_state_dict(data['model'])
        self.ema_model.load_state_dict(data['ema'])


    # def create_cond(self, start_time, cond_overall):
    #     cond = {}
    #     for i in range(start_time, start_time+self.env.horizon):
    #         if i in cond_overall.keys():
    #             cond[i] = cond_overall[i]
    #     return cond
    
    def sample_guided(self, sample_num=8, resample=False, test_data=None, no_cond=False, draw_traj=False, fn_choose={'energy':1}, use_invdyn=False):
        global test_data_global
        self.model.eval()
        self.ema_model.eval()
        
        mse_total = 0
        mse_final_total = 0
        mse_final_target_total = 0
        mse_final_target2_total = 0
        mse_u_total = 0
        mse_u_target_total = 0
        mse_u_data_total = 0
        mse_u_data_appr_total = 0
        mse_u_target_data_total = 0
        mse_u_target_data_appr_total = 0
        
        energy_u_total = 0
        energy_u_data_total = 0
        energy_u_data_appr_total = 0
        energy_actions_total = []
        mses = torch.zeros(self.env.max_T)
        dis_to_end = torch.zeros(self.env.max_T)
        dis_to_end_from_act = torch.zeros(self.env.max_T)
        # pdb.set_trace()
        
        traj_tensors = []
        search_cond = False
        for epoch in range(sample_num):
            if test_data:
                random_int = np.random.randint(0, len(test_data))
            with torch.no_grad():
                # inpaint cond from test_data or random
                if test_data==None:
                    
                    if not self.kuramoto:
                        y_f = np.random.randn(1, 1, self.env.num_observation) * self.sigma
                        y_0 = np.zeros((1, 1, self.env.num_observation))
                    else:
                        n = self.env.num_observation
                        theta1 = np.mod(4 * np.pi * np.arange(1,n+1) / n-1e-6, 2 * np.pi)
                        theta2 = np.mod(2 * np.pi * np.arange(1,n+1) / n-1e-6, 2 * np.pi)
                        y_f = theta2[np.newaxis, np.newaxis, :]
                        y_0 = theta1[np.newaxis, np.newaxis, :]
                        
                    y_0 = self.dataset.normalizer['Y'].normalize(y_0).to(self.device)
                    y_f = self.dataset.normalizer['Y'].normalize(y_f).to(self.device)
                    # y_0 = torch.tensor(y_0).to(self.device)
                    # y_f = torch.tensor(y_f).to(self.device)
                else:
                    y_f = test_data[random_int][0][-1, -self.env.num_observation:].unsqueeze(0).unsqueeze(0).to(self.device) # 1, 1, p
                    y_0 = test_data[random_int][0][0, -self.env.num_observation:].unsqueeze(0).unsqueeze(0).to(self.device) # 1, 1, p
                    # y_f = np.zeros((1, 1, self.env.num_observation)).astype(np.float32)
                    # y_f = torch.tensor(y_f).to(self.device)
                    # print(y_0)
                    # print(y_f)
                    
                # planning or one-shot
                if not resample:
                    try:
                        guide = LossFunction_noparams(horizon=self.env.max_T+1, transition_dim=self.ema_model.transition_dim, observation_dim=self.ema_model.observation_dim, fn_choose=fn_choose, end_vector=y_f.squeeze())
                        self.ema_model.set_guide_fn(guide)
                    except:
                        pass
                    if not no_cond:
                        batch_size = 1
                        if isinstance(self.dataset, TrainData_norm_free):
                            trajectories, _, _, guidances = self.ema_model(batch_size=batch_size, cond={0: y_0, self.env.max_T: y_f}, denoiser_cond=torch.ones(batch_size,self.ema_model.model.denoiser_cond_dim).to(self.device),horizon=self.env.max_T+1, apply_guidance=self.apply_guidance, guide_clean=self.guide_clean)
                        else:  
                            trajectories, _, _, guidances = self.ema_model(batch_size=batch_size, cond={0: y_0, self.env.max_T: y_f}, horizon=self.env.max_T+1, apply_guidance=self.apply_guidance, guide_clean=self.guide_clean)
                        
                        
                        trajectories = trajectories[0:1]
                    else:
                        raise NotImplementedError
                        # # better to use no_cond when fn_choose uses specific_end
                        # trajectories, _, _, guidances = self.ema_model(batch_size=4, cond={0: y_0}, horizon=self.env.max_T+1, apply_guidance=self.apply_guidance, guide_clean=self.guide_clean)
                        # trajectories = trajectories[0:1]


                    if not use_invdyn:
                        actions = trajectories[0, :-1, :self.ema_model.action_dim] # T, m
                        # action = actions[0, 0]
                        # observations = trajectories[0, 1:, self.ema_model.action_dim:] #  T, p
                        observations = trajectories[0, :, self.ema_model.action_dim:] # T+1, p
                    else:
                        # observations = trajectories[0, 1:]
                        observations = trajectories[0]
                        actions = []
                        for i in range(self.env.max_T):
                            obs_comb = torch.cat([trajectories[:, i, :], trajectories[:, i+1, :]], dim=-1)
                            # obs_comb = obs_comb.reshape(-1, 2*self.ema_model.observation_dim)
                            action = self.ema_model.inv_model(obs_comb)
                            actions.append(action.squeeze(0))
                        actions = torch.stack(actions)
                        actions_with_end = torch.cat([actions, torch.zeros(1, actions.size(-1)).to(actions.device)], dim=0)
                        trajectories = torch.cat([actions_with_end.unsqueeze(0), trajectories], dim=-1)
                    actions = actions.cpu()
                    observations = observations.cpu()
                    trajectories = trajectories.cpu()
                
                    if search_cond:
                        for i in range(5):
                            denoiser_cond = i/3*torch.ones(batch_size,self.ema_model.model.denoiser_cond_dim).to(self.device)
                            trajectories_search, _, _, guidances_search = self.ema_model(batch_size=batch_size, cond={0: y_0, self.env.max_T: y_f}, denoiser_cond=denoiser_cond, horizon=self.env.max_T+1, apply_guidance=self.apply_guidance, guide_clean=self.guide_clean)
                            observations_search = trajectories_search[0]
                            actions_search = []
                            for j in range(self.env.max_T):
                                obs_comb = torch.cat([trajectories_search[:, j, :], trajectories_search[:, j+1, :]], dim=-1)
                                action = self.ema_model.inv_model(obs_comb)
                                actions_search.append(action.squeeze(0))
                            actions_search = torch.stack(actions_search)
                            actions_with_end_search = torch.cat([actions_search, torch.zeros(1, actions_search.size(-1)).to(actions_search.device)], dim=0)
                            trajectories_search = torch.cat([actions_with_end_search.unsqueeze(0), trajectories_search], dim=-1)
                            diff = torch.mean(torch.norm((trajectories_search[:, 1:] - trajectories_search[:, :-1]) - (trajectories_search[:,-1:]-trajectories_search[:,0:1])/(trajectories_search.shape[1]-1)))
                            actions_search = actions_search.cpu()
                            self.writer.add_scalar(f'energy_search/{i}', torch.norm(actions_search, p=2), self.step)
                            self.writer.add_scalar(f'energy_search/diff_{i}', diff.squeeze().item(), self.step)
                            self.writer.add_scalar(f'energy_search/curve', torch.norm(actions_search, p=2), i/3)
                            self.writer.add_scalar(f'energy_search/curve_diff', diff.squeeze().item(), i/3)

                        # self.writer.add_scalar(f'energy_search/{5}', torch.norm(actions, p=2), self.step)
                        # self.writer.add_scalar(f'energy_search/curve', torch.norm(actions, p=2), 5)
                        search_cond = False
                
                 
                else:
                    raise NotImplementedError
                    # cond = {0: y_0, self.ema_model.horizon-1: y_f}
                    # actions = []
                    # observations = []
                    # ternimal = False
                    # for i in range(self.env.max_T):
                    # # i=0
                    # # while not ternimal:
                    #     if i == 0:
                    #         self.env.reset()
                    #         # pass
                    #     else:
                    #         action = self.dataset.normalizer['U'].unnormalize(action.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
                    #         obs, _, ternimal = self.env.step(action, target=y_f.squeeze().cpu())
                    #         # print(obs.shape)
                    #         # observations.append(obs)
                    #         obs = self.dataset.normalizer['Y'].normalize(obs.unsqueeze(0).unsqueeze(0))
                    #         cond[0] = obs.to(self.device)
                    #     # method 1
                    #     if i>self.env.max_T-self.ema_model.horizon+1:
                    #         shift = self.env.max_T-i
                    #         cond.pop(shift+1)
                            
                    #         cond[shift] = y_f
                    #     # print(cond.keys())
                        
                    #     # method 2
                    #     # none
                        
                    #     if not no_cond:
                    #         trajectories, *_ = self.ema_model(cond=cond, horizon=self.ema_model.horizon)
                    #     else:
                    #         raise NotImplementedError
                    #     trajectories = trajectories.cpu()
                    #     action = trajectories[0, 0, :self.ema_model.action_dim]
                    #     obs = trajectories[0, 1, self.ema_model.action_dim:]
                    #     actions.append(action)
                    #     observations.append(obs)
                    #     # i+=1
                    #     # if i>100:
                    #     #     print("Too many steps")
                    #     #     break
                    # # obs, *_ = self.env.step(action)
                    # # observations.append(obs)
                    # actions = torch.stack(actions).cpu() # T, m
                    # observations = torch.stack(observations).cpu() # T, p
                    # # actions = actions[:self.env.max_T]
                    # # observations = observations[:self.env.max_T]
                if not resample:
                    traj_tensors.append(trajectories.squeeze(0))
                else:
                    traj_tensors.append(trajectories.squeeze(0))
                    
                if self.normalized:
                    observations = self.dataset.normalizer['Y'].unnormalize(observations.unsqueeze(0))
                    actions = self.dataset.normalizer['U'].unnormalize(actions.unsqueeze(0))
                    if test_data:
                        y_f = test_data.normalizer['Y'].unnormalize(y_f.cpu()).squeeze(0).squeeze(0)
                        y_0 = test_data.normalizer['Y'].unnormalize(y_0.cpu()).squeeze(0).squeeze(0)
                    else:
                        y_f = self.dataset.normalizer['Y'].unnormalize(y_f.cpu()).squeeze(0).squeeze(0)
                        y_0 = self.dataset.normalizer['Y'].unnormalize(y_0.cpu()).squeeze(0).squeeze(0)
                        
                    observations = observations.squeeze(0)
                    actions = actions.squeeze(0)
                if (not use_invdyn) and (guidances is not None):
                    for k,v in guidances.items():
                        if epoch<4:
                            # pdb.set_trace()
                            self.writer.add_scalar(f'guidance/{k}_{epoch}', np.nanmean(v.cpu()), self.step)
                            
                    
                
                # evaluate the correspondense of actions and observations
                self.env.reset()
                obs_from_act = self.env.from_actions_to_obs_direct(actions, start = y_0)
                obs_from_act_single = []
                for i in range(self.env.max_T):
                    self.env.reset(observations[i].cpu())
                    obs_from_act_single.append(self.env.step(actions[i].cpu())[0])
                # print("actions norm of each time step: ", torch.sum(actions**2, dim=1))
                if self.kuramoto and (test_data is None):
                    obs_from_act_long_max = self.env.from_actions_to_obs_longer(actions, start = y_0, continues=100)
                    pre_time = obs_from_act.shape[0]
                    obs_from_act_long = obs_from_act_long_max[:5*pre_time]
                    obs_from_act_long = torch.cat([y_0.squeeze().unsqueeze(0).repeat(pre_time,1), obs_from_act_long],dim=0)
                    # plot obs_from_act_long
                    # obs_from_act_long = obs_from_act_long.cpu() # T, p
                    fig, ax = plt.subplots()
                    # 用一个区别明显的colormap
                    colormaps  = plt.get_cmap('tab10').colors[:obs_from_act_long.shape[1]]
                    for node in range(obs_from_act_long.shape[1]):
                        ax.plot(np.arange(obs_from_act_long.shape[0]), obs_from_act_long[:, node], label=f'node_{node}', color=colormaps[node])
                        # ax.legend()
                        ax.plot(np.arange(obs_from_act_long.shape[0]), y_f.cpu().squeeze()[node]*torch.ones(obs_from_act_long.shape[0]), label=f'target_node_{node}', linestyle='--', color=colormaps[node])
                    # 绘制绿色区块，横轴范围为[0,obs_from_act.shape[0]]，纵轴范围为全部，透明度为0.3
                    plt.axvspan(pre_time-1, pre_time+obs_from_act.shape[0]-1, color='green', alpha=0.2)
                    plt.title(f"target loss:{mse_loss(obs_from_act[-1],y_f.squeeze().cpu()).item()}, energy:{torch.norm(actions, p=2)}")
                    # pdb.set_trace()
                    fig.savefig(f'./images/{self.writer.logdir.split("/")[-1]}kuramoto_step{self.step}.png')
                    #close figure
                    plt.clf()
                    plt.close(fig)
                    
                    fig, ax = plt.subplots()
                    # 用一个区别明显的colormap
                    colormaps  = plt.get_cmap('tab10').colors[:obs_from_act_long_max.shape[1]]
                    
                    mses_long_max = [mse_loss(obs_long ,y_f.squeeze().cpu()).item() for obs_long in obs_from_act_long_max]
                    position = next((i for i, val in enumerate(mses_long_max) if val < 1e-4), None)
                    for node in range(obs_from_act_long.shape[1]):
                        ax.plot(np.arange(obs_from_act_long_max.shape[0]), obs_from_act_long_max[:, node], label=f'node_{node}', color=colormaps[node])
                        # ax.legend()
                        ax.plot(np.arange(obs_from_act_long_max.shape[0]), y_f.cpu().squeeze()[node]*torch.ones(obs_from_act_long_max.shape[0]), label=f'target_node_{node}', linestyle='--', color=colormaps[node])
                    if position is not None:
                        ax.axvline(x=position, color='r', linestyle='dotted')
                        ax.text(position + 0.1, 0, f'x={position}', color='r', fontsize=12)
                    plt.axvspan(0, obs_from_act.shape[0]-1, color='green', alpha=0.2)
                    plt.title(f"target loss:{mse_loss(obs_from_act[-1],y_f.squeeze().cpu()).item()}, energy:{torch.norm(actions, p=2)}")
                        # pdb.set_trace()
                    fig.savefig(f'./images/{self.writer.logdir.split("/")[-1]}kuramoto_step{self.step}_long.png')
                        #close figure
                    plt.clf()
                    plt.close(fig)
                    
                    # pdb.set_trace()
                    
                assert obs_from_act.shape == observations[1:].shape
                mse = mse_loss(obs_from_act,observations[1:]).item()
                # print("MSE between obs produced from sampled actions and sampled obs: ", mse)
                # assert torch.all(y_f.cpu().squeeze()==observations.cpu().squeeze()), "Error"
                mse_final = mse_loss(obs_from_act[-1],y_f.squeeze().cpu()).item()
                # print("MSE between final obs produced from sampled actions and sampled obs: ", mse_final)
                # mse_final_target = mse_loss(obs_from_act[-1], y_f.cpu().squeeze(0)).item()
                # print("MSE between final obs produced from sampled actions and target final obs: ", mse_final_target)
                # mse_final_target2 = mse_loss(observations[-1], y_f.cpu().squeeze(0)).item()
                # print("MSE between final sampled obs and target final obs: ", mse_final_target2)
                for i in range(self.env.max_T):
                    mses[i] += mse_loss(obs_from_act_single[i],observations[i+1]).item()
                    dis_to_end[i] += mse_loss(observations[i].cpu(),y_f.cpu()).item()
                    dis_to_end_from_act[i] += mse_loss(obs_from_act[i].cpu(),y_f.cpu()).item()
                    
                
                # evaluate the difference between actions and model-based minumum energy control
                u_min, y_f_hat = self.env.calculate_model_based_control(y_f.cpu().squeeze())
                assert u_min.shape == actions.shape,f"{u_min.shape}, {actions.shape}"
                mse_u = mse_loss(u_min, actions).item()
                # print("MSE between model-based minumum energy control and sampled actions: ", mse_u)
                mse_u_target = mse_loss(y_f_hat, y_f.cpu().squeeze()).item()
                # print("MSE between final obs produced from model-based minumum energy control and target final obs: ", mse_u_target)
                # print("====================================")
                u_min_data, y_f_hat_data, u_min_data_appr, y_f_hat_data_appr = self.env.calculate_data_driven_control(y_f.cpu().squeeze())
                assert u_min_data.shape == actions.shape, f"{u_min_data.shape}, {actions.shape}"
                mse_u_data = mse_loss(u_min_data, actions).item()
                mse_u_data_appr = mse_loss(u_min_data_appr, actions).item()
                mse_u_target_data = mse_loss(y_f_hat_data, y_f.cpu().squeeze()).item()
                mse_u_target_data_appr = mse_loss(y_f_hat_data_appr, y_f.cpu().squeeze()).item()
                
                # calculate energy of controls:
                energy_u = torch.norm(u_min, p=2)
                energy_u_data = torch.norm(u_min_data, p=2)
                energy_u_data_appr = torch.norm(u_min_data_appr, p=2)
                energy_actions = torch.norm(actions, p=2)
                
                
                mse_total += mse
                mse_final_total += mse_final
                # mse_final_target_total += mse_final_target
                # mse_final_target2_total += mse_final_target2
                mse_u_total += mse_u
                mse_u_target_total += mse_u_target
                mse_u_data_total += mse_u_data
                mse_u_data_appr_total += mse_u_data_appr
                mse_u_target_data_total += mse_u_target_data
                mse_u_target_data_appr_total += mse_u_target_data_appr
                
                energy_u_total += energy_u
                energy_u_data_total += energy_u_data
                energy_u_data_appr_total += energy_u_data_appr
                energy_actions_total += [energy_actions]
                
                traj_sampled = obs_from_act
                traj_model_based = self.env.from_actions_to_obs_direct(u_min)
                traj_data_driven = self.env.from_actions_to_obs_direct(u_min_data)
                
                # draw observations and obs_from_act
                if self.step % 1000 == 0:
                    obs_from_act_draw = torch.cat([y_0.unsqueeze(0), obs_from_act], dim=0)
                    for dim in range(observations.shape[-1]):
                        fig, ax = plt.subplots()
                        ax.plot(np.arange(observations.shape[0]), observations[:, dim], label='obs')
                        ax.plot(np.arange(observations.shape[0]), obs_from_act_draw[:, dim], label='obs_from_act')
                        ax.legend()
                        ax.grid()
                        self.writer.add_figure(f'val/obs_dim_{dim}', fig, self.step)
                        plt.close()
                
                if traj_sampled.shape[-1]>=2:
                    if draw_traj:
                        # draw plot to writer
                        fig, ax = plt.subplots()
                        ax.plot(y_f.cpu().squeeze()[0], y_f.cpu().squeeze()[1], 'ro', label='target')
                        ax.plot(traj_sampled[:, 0], traj_sampled[:, 1], label='sampled')
                        ax.plot(traj_model_based[:, 0], traj_model_based[:, 1], label='model_based')
                        ax.plot(traj_data_driven[:, 0], traj_data_driven[:, 1], label='data_driven')
                        ax.legend()
                        self.writer.add_figure('val/trajectory', fig, self.step)
                        plt.close()
                        draw_traj = False

        traj_tensors = torch.stack(traj_tensors, dim=0)
        num_samples = len(traj_tensors)
        # randomly select 8 samples in dataloader
        for i in range(num_samples):
            
            batch = next(self.dataloader).trajectories
            random_int1 = np.random.randint(0, len(batch))
            random_int2 = np.random.randint(0, len(batch))
            batch1 = batch[random_int1]
            batch2 = batch[random_int2]
            
            if i==0:
                batch1_list = [batch1]
                batch2_list = [batch2]
            else:
                batch1_list.append(batch1)
                batch2_list.append(batch2)
        batch1_list = torch.stack(batch1_list,dim=0)
        batch2_list = torch.stack(batch2_list,dim=0)
        print(batch1_list.shape, traj_tensors.shape)
        
        mmd = MMDLoss()
        mmd1 = mmd(batch1_list.view(num_samples,-1), traj_tensors.view(num_samples,-1)).item()
        mmd2 = mmd(batch1_list.view(num_samples,-1), batch2_list.view(num_samples,-1)).item()
        print('mmd between gt and sample:', mmd1)
        print('mmd between gt and gt\':', mmd2)
        
        # print(traj_tensors.shape)
        mse_total /= sample_num
        mse_final_total /= sample_num
        # mse_final_target_total /= sample_num
        # mse_final_target2_total /= sample_num
        mse_u_total /= sample_num
        mse_u_target_total /= sample_num
        mse_u_data_total /= sample_num
        mse_u_data_appr_total /= sample_num
        mse_u_target_data_total /= sample_num
        mse_u_target_data_appr_total /= sample_num
        
        mses /= sample_num
        dis_to_end /= sample_num
        dis_to_end_from_act /= sample_num
        
        energy_actions_total, energy_actions_std = np.mean(np.array(energy_actions_total)), np.std(np.array(energy_actions_total))
        print("==================Average======================")
        print("Average MSE between obs produced from sampled actions and sampled obs: ", mse_total)
        print("Average MSE between final obs produced from sampled actions and sampled obs: ", mse_final_total)
        # print("Average MSE between final obs produced from sampled actions and target final obs: ", mse_final_target_total)
        # print("Average MSE between final sampled obs and target final obs: ", mse_final_target2_total)
        
        print("Average MSE between model-based minumum energy control and sampled actions: ", mse_u_total)
        print(f"Average MSE between final obs produced from model-based minumum energy control and target final obs: ", mse_u_target_total)
        
        print(f"Average MSE between data-driven minumum energy control and sampled actions: ", mse_u_data_total)
        print(f"Average MSE between data-driven minumum energy control approximation and sampled actions: ", mse_u_data_appr_total)
        print(f"Average MSE between final obs produced from data-driven minumum energy control and target final obs: ", mse_u_target_data_total)
        print(f"Average MSE between final obs produced from data-driven minumum energy control approximation and target final obs: ", mse_u_target_data_appr_total)
        
        
        print(f"Energies of model-based minumum energy control: {energy_u_total}, data-driven minumum energy control: {energy_u_data_total}, data-driven minumum energy control approximation: {energy_u_data_appr_total}, sampled actions: {energy_actions_total}")
        
        self.writer.add_scalar('val/obs', mse_total, self.step)
        self.writer.add_scalar('val/final_obs', mse_final_total, self.step)
        self.writer.add_scalar('val/energy', energy_actions_total, self.step)
        self.writer.add_scalar('val/energy_std', energy_actions_std, self.step)
        self.writer.add_scalar('val/energy_dd', energy_u_data_total, self.step)
        
        self.writer.add_scalar('val/mmd1', mmd1, self.step)
        self.writer.add_scalar('val/mmd2', mmd2, self.step)
        
        for i in range(self.env.max_T):
            self.writer.add_scalar(f'val2/obs_{i}', mses[i], self.step)
            self.writer.add_scalar(f'val3/distoend', dis_to_end[i], i)
            self.writer.add_scalar(f'val4/distoendfromact', dis_to_end_from_act[i], i)
            
        self.writer.flush()
        self.model.train()
        self.ema_model.train()
        return mse_total, mse_final_total, mse_final_target_total, mse_u_total, mse_u_target_total


    def sample_invdyn(self, sample_num=8, resample=False, test_data=None, no_cond=False, draw_traj=False):
        self.model.eval()
        self.ema_model.eval()
        
        mse_total = 0
        mse_final_total = 0
        mse_final_target_total = 0
        mse_final_target2_total = 0
        mse_u_total = 0
        mse_u_target_total = 0
        mse_u_data_total = 0
        mse_u_data_appr_total = 0
        mse_u_target_data_total = 0
        mse_u_target_data_appr_total = 0
        
        energy_u_total = 0
        energy_u_data_total = 0
        energy_u_data_appr_total = 0
        energy_actions_total = 0
        mses = torch.zeros(self.env.max_T)
        dis_to_end = torch.zeros(self.env.max_T)
        dis_to_end_from_act = torch.zeros(self.env.max_T)
        for epoch in range(sample_num):
            if test_data:
                random_int = np.random.randint(0, len(test_data))
            with torch.no_grad():
                # inpaint cond from test_data or random
                if test_data==None:
                    y_f = np.random.randn(1, 1, self.env.num_observation) * self.sigma
                    y_0 = np.zeros((1, 1, self.env.num_observation))
                    y_0 = self.dataset.normalizer['Y'].normalize(y_0)
                    y_f = self.dataset.normalizer['Y'].normalize(y_f)
                    y_0 = torch.tensor(y_0).to(self.device)
                    y_f = torch.tensor(y_f).to(self.device)
                else:
                    y_f = test_data[random_int][0][-1, -self.env.num_observation:].unsqueeze(0).unsqueeze(0).to(self.device) # 1, 1, p
                    y_0 = test_data[random_int][0][0, -self.env.num_observation:].unsqueeze(0).unsqueeze(0).to(self.device) # 1, 1, p
                    
                # planning or one-shot
                if not resample:
                    if not no_cond:
                        trajectories = self.ema_model(cond={0: y_0, self.env.max_T: y_f}, horizon=self.env.max_T+1)
                    else:
                        trajectories = self.ema_model(cond = {}, horizon=self.env.max_T+1)
                        y_0 = trajectories[0, 0]
                        y_f = trajectories[0, -1]
                    # print(trajectories.shape)
                    
                    # trajectories = trajectories.cpu()
                    # print(trajectories.shape)
                    # actions = trajectories[0, :-1, :self.ema_model.action_dim] # T, m
                    # # action = actions[0, 0]
                    observations = trajectories[0, 1:].cpu() #  T, p
                    actions = []
                    for i in range(self.env.max_T):
                        obs_comb = torch.cat([trajectories[:, i, :], trajectories[:, i+1, :]], dim=-1)
                        obs_comb = obs_comb.reshape(-1, 2*self.ema_model.observation_dim)
                        action = self.ema_model.inv_model(obs_comb)
                        actions.append(action.squeeze(0))
                    actions = torch.stack(actions).cpu()
                    # print(actions.shape)
                else:
                    # raise NotImplementedError
                    cond = {0: y_0, self.ema_model.horizon-1: y_f}
                    actions = []
                    observations = []
                    for i in range(self.env.max_T):
                        if i == 0:
                            self.env.reset()
                            # pass
                        else:
                            action = self.dataset.normalizer['U'].unnormalize(action.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
                            obs, _, ternimal = self.env.step(action, target=y_f.squeeze().cpu())
                            # print(obs.shape)
                            # observations.append(obs)
                            obs = self.dataset.normalizer['Y'].normalize(obs.unsqueeze(0).unsqueeze(0))
                            cond[0] = obs.to(self.device)
                        # method 1
                        if i>self.env.max_T-self.ema_model.horizon+1:
                            shift = self.env.max_T-i
                            cond.pop(shift+1)
                            
                            cond[shift] = y_f
                        # print(cond.keys())
                        
                        # method 2
                        # none
                        
                        if not no_cond:
                            trajectories = self.ema_model(cond=cond, horizon=self.ema_model.horizon)
                        else:
                            raise NotImplementedError
                        # action = trajectories[0, 0, :self.ema_model.action_dim]
                        # print(trajectories.shape)
                        obs_pre = trajectories[:, 0, :]
                        obs = trajectories[:, 1, :]
                        obs_comb = torch.cat([obs_pre, obs], dim=-1)
                        obs_comb = obs_comb.reshape(-1, 2*self.ema_model.observation_dim)
                        action = self.ema_model.inv_model(obs_comb).squeeze(0).cpu()
                        actions.append(action)
                        obs = obs.squeeze(0)
                        observations.append(obs)
                    obs, *_ = self.env.step(action)
                    # observations.append(obs)
                    actions = torch.stack(actions).cpu() # T, m
                    observations = torch.stack(observations).cpu() # T, p
                if self.normalized:
                    observations = self.dataset.normalizer['Y'].unnormalize(observations.unsqueeze(0))
                    actions = self.dataset.normalizer['U'].unnormalize(actions.unsqueeze(0))
                    if test_data:
                        y_f = test_data.normalizer['Y'].unnormalize(y_f.cpu()).squeeze(0).squeeze(0)
                        y_0 = test_data.normalizer['Y'].unnormalize(y_0.cpu()).squeeze(0).squeeze(0)
                    else:
                        y_f = self.dataset.normalizer['Y'].unnormalize(y_f.cpu()).squeeze(0).squeeze(0)
                        y_0 = self.dataset.normalizer['Y'].unnormalize(y_0.cpu()).squeeze(0).squeeze(0)
                            
                    observations = observations.squeeze(0)
                    actions = actions.squeeze(0)
                    
            # evaluate the correspondense of actions and observations
                self.env.reset()
                obs_from_act = self.env.from_actions_to_obs_direct(actions, start=y_0)
                assert obs_from_act.shape == observations.shape, f'{obs_from_act.shape} and {observations.shape}'
                mse = mse_loss(obs_from_act,observations).item()
                # print("MSE between obs produced from sampled actions and sampled obs: ", mse)
                # assert torch.all(y_f.cpu().squeeze()==observations.cpu().squeeze()), "Error"
                mse_final = mse_loss(obs_from_act[-1],y_f.squeeze().cpu()).item()
                # print("MSE between final obs produced from sampled actions and sampled obs: ", mse_final)
                # mse_final_target = mse_loss(obs_from_act[-1], y_f.cpu().squeeze(0)).item()
                # print("MSE between final obs produced from sampled actions and target final obs: ", mse_final_target)
                # mse_final_target2 = mse_loss(observations[-1], y_f.cpu().squeeze(0)).item()
                # print("MSE between final sampled obs and target final obs: ", mse_final_target2)
                for i in range(self.env.max_T):
                    mses[i] += mse_loss(obs_from_act[i],observations[i]).item()
                    dis_to_end[i] += mse_loss(observations[i].cpu(),y_f.cpu()).item()
                    dis_to_end_from_act[i] += mse_loss(obs_from_act[i].cpu(),y_f.cpu()).item()
                    
            # evaluate the difference between actions and model-based minumum energy control
                u_min, y_f_hat = self.env.calculate_model_based_control(y_f.cpu().squeeze())
                assert u_min.shape == actions.shape, f'{u_min.shape} & {actions.shape}'
                mse_u = mse_loss(u_min, actions).item()
                # print("MSE between model-based minumum energy control and sampled actions: ", mse_u)
                mse_u_target = mse_loss(y_f_hat, y_f.cpu().squeeze()).item()
                # print("MSE between final obs produced from model-based minumum energy control and target final obs: ", mse_u_target)
                # print("====================================")
                u_min_data, y_f_hat_data, u_min_data_appr, y_f_hat_data_appr = self.env.calculate_data_driven_control(y_f.cpu().squeeze())
                assert u_min_data.shape == actions.shape, f"{u_min_data.shape}, {actions.shape}"
                mse_u_data = mse_loss(u_min_data, actions).item()
                mse_u_data_appr = mse_loss(u_min_data_appr, actions).item()
                mse_u_target_data = mse_loss(y_f_hat_data, y_f.cpu().squeeze()).item()
                mse_u_target_data_appr = mse_loss(y_f_hat_data_appr, y_f.cpu().squeeze()).item()
                
                # calculate energy of controls:
                energy_u = torch.sum(u_min**2)
                energy_u_data = torch.sum(u_min_data**2)
                energy_u_data_appr = torch.sum(u_min_data_appr**2)
                energy_actions = torch.sum(actions**2)
                
                
                mse_total += mse
                mse_final_total += mse_final
                # mse_final_target_total += mse_final_target
                # mse_final_target2_total += mse_final_target2
                mse_u_total += mse_u
                mse_u_target_total += mse_u_target
                mse_u_data_total += mse_u_data
                mse_u_data_appr_total += mse_u_data_appr
                mse_u_target_data_total += mse_u_target_data
                mse_u_target_data_appr_total += mse_u_target_data_appr
                
                energy_u_total += energy_u
                energy_u_data_total += energy_u_data
                energy_u_data_appr_total += energy_u_data_appr
                energy_actions_total += energy_actions
                
                traj_sampled = obs_from_act
                traj_model_based = self.env.from_actions_to_obs_direct(u_min)
                traj_data_driven = self.env.from_actions_to_obs_direct(u_min_data)
                if traj_sampled.shape[-1]==2:
                    if draw_traj:
                        import matplotlib.pyplot as plt
                        # draw plot to writer
                        fig, ax = plt.subplots()
                        ax.plot(y_f.cpu().squeeze()[0], y_f.cpu().squeeze()[1], 'ro', label='target')
                        ax.plot(traj_sampled[:, 0], traj_sampled[:, 1], label='sampled')
                        ax.plot(traj_model_based[:, 0], traj_model_based[:, 1], label='model_based')
                        ax.plot(traj_data_driven[:, 0], traj_data_driven[:, 1], label='data_driven')
                        ax.legend()
                        self.writer.add_figure('val/trajectory', fig, self.step)
                        plt.close()
                        draw_traj = False
            
        mse_total /= sample_num
        mse_final_total /= sample_num
        # mse_final_target_total /= sample_num
        # mse_final_target2_total /= sample_num
        mse_u_total /= sample_num
        mse_u_target_total /= sample_num
        mse_u_data_total /= sample_num
        mse_u_data_appr_total /= sample_num
        mse_u_target_data_total /= sample_num
        mse_u_target_data_appr_total /= sample_num
        
        mses /= sample_num
        dis_to_end /= sample_num
        dis_to_end_from_act /= sample_num
        print("==================Average======================")
        print("Average MSE between obs produced from sampled actions and sampled obs: ", mse_total)
        print("Average MSE between final obs produced from sampled actions and sampled obs: ", mse_final_total)
        # print("Average MSE between final obs produced from sampled actions and target final obs: ", mse_final_target_total)
        # print("Average MSE between final sampled obs and target final obs: ", mse_final_target2_total)
        
        print("Average MSE between model-based minumum energy control and sampled actions: ", mse_u_total)
        print(f"Average MSE between final obs produced from model-based minumum energy control and target final obs: ", mse_u_target_total)
        
        print(f"Average MSE between data-driven minumum energy control and sampled actions: ", mse_u_data_total)
        print(f"Average MSE between data-driven minumum energy control approximation and sampled actions: ", mse_u_data_appr_total)
        print(f"Average MSE between final obs produced from data-driven minumum energy control and target final obs: ", mse_u_target_data_total)
        print(f"Average MSE between final obs produced from data-driven minumum energy control approximation and target final obs: ", mse_u_target_data_appr_total)
        
        
        print(f"Energies of model-based minumum energy control: {energy_u_total}, data-driven minumum energy control: {energy_u_data_total}, data-driven minumum energy control approximation: {energy_u_data_appr_total}, sampled actions: {energy_actions_total}")
        
        self.writer.add_scalar('val/obs', mse_total, self.step)
        self.writer.add_scalar('val/final_obs', mse_final_total, self.step)
        self.writer.add_scalar('val/energy', energy_actions_total, self.step)
        self.writer.add_scalar('val/energy_dd', energy_u_data_total, self.step)
        for i in range(self.env.max_T):
            self.writer.add_scalar(f'val2/obs_{i}', mses[i], self.step)
            self.writer.add_scalar(f'val3/distoend', dis_to_end[i], i)
            self.writer.add_scalar(f'val4/distoendfromact', dis_to_end_from_act[i], i)
        self.writer.flush()
        self.model.train()
        self.ema_model.train()
        return mse_total, mse_final_total, mse_final_target_total, mse_u_total, mse_u_target_total
    
    def sample(self, sample_num=8, resample=False, test_data=None, no_cond=False, draw_traj=False):
        self.model.eval()
        self.ema_model.eval()
        
        mse_total = 0
        mse_final_total = 0
        mse_final_target_total = 0
        mse_final_target2_total = 0
        mse_u_total = 0
        mse_u_target_total = 0
        mse_u_data_total = 0
        mse_u_data_appr_total = 0
        mse_u_target_data_total = 0
        mse_u_target_data_appr_total = 0
        
        energy_u_total = 0
        energy_u_data_total = 0
        energy_u_data_appr_total = 0
        energy_actions_total = 0
        mses = torch.zeros(self.env.max_T)
        dis_to_end = torch.zeros(self.env.max_T)
        dis_to_end_from_act = torch.zeros(self.env.max_T)
        
        traj_tensors = []
        for epoch in range(sample_num):
            if test_data:
                random_int = np.random.randint(0, len(test_data))
            with torch.no_grad():
                # inpaint cond from test_data or random
                if test_data==None:
                    if not self.kuramoto:
                        y_f = np.random.randn(1, 1, self.env.num_observation) * self.sigma
                        y_0 = np.zeros((1, 1, self.env.num_observation))
                    else:
                        n = self.env.num_observation
                        theta1 = np.mod(0 * np.pi * np.arange(n) / n, 2 * np.pi)
                        theta2 = np.mod(4 * np.pi * np.arange(n) / n, 2 * np.pi)
                        y_f = theta2[np.newaxis, np.newaxis, :]
                        y_0 = theta1[np.newaxis, np.newaxis, :]
                        
                    y_0 = self.dataset.normalizer['Y'].normalize(y_0)
                    y_f = self.dataset.normalizer['Y'].normalize(y_f)
                    y_0 = torch.tensor(y_0).to(self.device)
                    y_f = torch.tensor(y_f).to(self.device)
                else:
                    y_f = test_data[random_int][0][-1, -self.env.num_observation:].unsqueeze(0).unsqueeze(0).to(self.device) # 1, 1, p
                    y_0 = test_data[random_int][0][0, -self.env.num_observation:].unsqueeze(0).unsqueeze(0).to(self.device) # 1, 1, p
                    
                # planning or one-shot
                if not resample:
                    if not no_cond:
                        trajectories, *_ = self.ema_model(cond={0: y_0, self.env.max_T: y_f}, horizon=self.env.max_T+1)
                    else:
                        trajectories, *_ = self.ema_model(cond = {}, horizon=self.env.max_T+1)
                        y_0 = trajectories[0, 0, self.ema_model.action_dim:]
                        y_f = trajectories[0, -1, self.ema_model.action_dim:]
                    trajectories = trajectories.cpu()
                    
                    actions = trajectories[0, :-1, :self.ema_model.action_dim] # T, m
                    # action = actions[0, 0]
                    observations = trajectories[0, 1:, self.ema_model.action_dim:] #  T, p
               
                else:
                    cond = {0: y_0, self.ema_model.horizon-1: y_f}
                    actions = []
                    observations = []
                    ternimal = False
                    for i in range(self.env.max_T):
                    # i=0
                    # while not ternimal:
                        if i == 0:
                            self.env.reset()
                            # pass
                        else:
                            action = self.dataset.normalizer['U'].unnormalize(action.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
                            obs, _, ternimal = self.env.step(action, target=y_f.squeeze().cpu())
                            # print(obs.shape)
                            # observations.append(obs)
                            obs = self.dataset.normalizer['Y'].normalize(obs.unsqueeze(0).unsqueeze(0))
                            cond[0] = obs.to(self.device)
                        # method 1
                        if i>self.env.max_T-self.ema_model.horizon+1:
                            shift = self.env.max_T-i
                            cond.pop(shift+1)
                            
                            cond[shift] = y_f
                        # print(cond.keys())
                        
                        # method 2
                        # none
                        
                        if not no_cond:
                            trajectories, *_ = self.ema_model(cond=cond, horizon=self.ema_model.horizon)
                        else:
                            raise NotImplementedError
                        trajectories = trajectories.cpu()
                        action = trajectories[0, 0, :self.ema_model.action_dim]
                        obs = trajectories[0, 1, self.ema_model.action_dim:]
                        actions.append(action)
                        observations.append(obs)
                        # i+=1
                        # if i>100:
                        #     print("Too many steps")
                        #     break
                    # obs, *_ = self.env.step(action)
                    # observations.append(obs)
                    actions = torch.stack(actions).cpu() # T, m
                    observations = torch.stack(observations).cpu() # T, p
                    # actions = actions[:self.env.max_T]
                    # observations = observations[:self.env.max_T]
                if not resample:
                    traj_tensors.append(trajectories.squeeze(0))
                else:
                    traj_tensors.append(trajectories.squeeze(0))
                    
                if self.normalized:
                    observations = self.dataset.normalizer['Y'].unnormalize(observations.unsqueeze(0))
                    actions = self.dataset.normalizer['U'].unnormalize(actions.unsqueeze(0))
                    if test_data:
                        y_f = test_data.normalizer['Y'].unnormalize(y_f.cpu()).squeeze(0).squeeze(0)
                        y_0 = test_data.normalizer['Y'].unnormalize(y_0.cpu()).squeeze(0).squeeze(0)
                    else:
                        y_f = self.dataset.normalizer['Y'].unnormalize(y_f.cpu()).squeeze(0).squeeze(0)
                        y_0 = self.dataset.normalizer['Y'].unnormalize(y_0.cpu()).squeeze(0).squeeze(0)
                        
                    observations = observations.squeeze(0)
                    actions = actions.squeeze(0)
                
                
                    
                
            # evaluate the correspondense of actions and observations
                self.env.reset()
                obs_from_act = self.env.from_actions_to_obs_direct(actions)
                assert obs_from_act.shape == observations.shape
                mse = mse_loss(obs_from_act,observations).item()
                # print("MSE between obs produced from sampled actions and sampled obs: ", mse)
                # assert torch.all(y_f.cpu().squeeze()==observations.cpu().squeeze()), "Error"
                mse_final = mse_loss(obs_from_act[-1],y_f.squeeze().cpu()).item()
                # print("MSE between final obs produced from sampled actions and sampled obs: ", mse_final)
                # mse_final_target = mse_loss(obs_from_act[-1], y_f.cpu().squeeze(0)).item()
                # print("MSE between final obs produced from sampled actions and target final obs: ", mse_final_target)
                # mse_final_target2 = mse_loss(observations[-1], y_f.cpu().squeeze(0)).item()
                # print("MSE between final sampled obs and target final obs: ", mse_final_target2)
                for i in range(self.env.max_T):
                    mses[i] += mse_loss(obs_from_act[i],observations[i]).item()
                    dis_to_end[i] += mse_loss(observations[i].cpu(),y_f.cpu()).item()
                    dis_to_end_from_act[i] += mse_loss(obs_from_act[i].cpu(),y_f.cpu()).item()
                    
                
            # evaluate the difference between actions and model-based minumum energy control
                u_min, y_f_hat = self.env.calculate_model_based_control(y_f.cpu().squeeze())
                assert u_min.shape == actions.shape,f"{u_min.shape}, {actions.shape}"
                mse_u = mse_loss(u_min, actions).item()
                # print("MSE between model-based minumum energy control and sampled actions: ", mse_u)
                mse_u_target = mse_loss(y_f_hat, y_f.cpu().squeeze()).item()
                # print("MSE between final obs produced from model-based minumum energy control and target final obs: ", mse_u_target)
                # print("====================================")
                u_min_data, y_f_hat_data, u_min_data_appr, y_f_hat_data_appr = self.env.calculate_data_driven_control(y_f.cpu().squeeze())
                assert u_min_data.shape == actions.shape, f"{u_min_data.shape}, {actions.shape}"
                mse_u_data = mse_loss(u_min_data, actions).item()
                mse_u_data_appr = mse_loss(u_min_data_appr, actions).item()
                mse_u_target_data = mse_loss(y_f_hat_data, y_f.cpu().squeeze()).item()
                mse_u_target_data_appr = mse_loss(y_f_hat_data_appr, y_f.cpu().squeeze()).item()
                
                # calculate energy of controls:
                energy_u = torch.norm(u_min, p=2)
                energy_u_data = torch.norm(u_min_data, p=2)
                energy_u_data_appr = torch.norm(u_min_data_appr, p=2)
                energy_actions = torch.norm(actions, p=2)
                
                
                mse_total += mse
                mse_final_total += mse_final
                # mse_final_target_total += mse_final_target
                # mse_final_target2_total += mse_final_target2
                mse_u_total += mse_u
                mse_u_target_total += mse_u_target
                mse_u_data_total += mse_u_data
                mse_u_data_appr_total += mse_u_data_appr
                mse_u_target_data_total += mse_u_target_data
                mse_u_target_data_appr_total += mse_u_target_data_appr
                
                energy_u_total += energy_u
                energy_u_data_total += energy_u_data
                energy_u_data_appr_total += energy_u_data_appr
                energy_actions_total += energy_actions
                
                traj_sampled = obs_from_act
                traj_model_based = self.env.from_actions_to_obs_direct(u_min)
                traj_data_driven = self.env.from_actions_to_obs_direct(u_min_data)
                if traj_sampled.shape[-1]>=2:
                    if draw_traj:
                        import matplotlib.pyplot as plt
                        # draw plot to writer
                        fig, ax = plt.subplots()
                        ax.plot(y_f.cpu().squeeze()[0], y_f.cpu().squeeze()[1], 'ro', label='target')
                        ax.plot(traj_sampled[:, 0], traj_sampled[:, 1], label='sampled')
                        ax.plot(traj_model_based[:, 0], traj_model_based[:, 1], label='model_based')
                        ax.plot(traj_data_driven[:, 0], traj_data_driven[:, 1], label='data_driven')
                        ax.legend()
                        self.writer.add_figure('val/trajectory', fig, self.step)
                        plt.close()
                        draw_traj = False

        traj_tensors = torch.stack(traj_tensors, dim=0)
        num_samples = len(traj_tensors)
        # randomly select 8 samples in dataloader
        for i in range(num_samples):
            
            batch = next(self.dataloader).trajectories
            random_int1 = np.random.randint(0, len(batch))
            random_int2 = np.random.randint(0, len(batch))
            batch1 = batch[random_int1]
            batch2 = batch[random_int2]
            
            if i==0:
                batch1_list = [batch1]
                batch2_list = [batch2]
            else:
                batch1_list.append(batch1)
                batch2_list.append(batch2)
        batch1_list = torch.stack(batch1_list,dim=0)
        batch2_list = torch.stack(batch2_list,dim=0)
        print(batch1_list.shape, traj_tensors.shape)
        
        mmd = MMDLoss()
        mmd1 = mmd(batch1_list.view(num_samples,-1), traj_tensors.view(num_samples,-1)).item()
        mmd2 = mmd(batch1_list.view(num_samples,-1), batch2_list.view(num_samples,-1)).item()
        print('mmd between gt and sample:', mmd1)
        print('mmd between gt and gt\':', mmd2)
        
        # print(traj_tensors.shape)
        mse_total /= sample_num
        mse_final_total /= sample_num
        # mse_final_target_total /= sample_num
        # mse_final_target2_total /= sample_num
        mse_u_total /= sample_num
        mse_u_target_total /= sample_num
        mse_u_data_total /= sample_num
        mse_u_data_appr_total /= sample_num
        mse_u_target_data_total /= sample_num
        mse_u_target_data_appr_total /= sample_num
        
        mses /= sample_num
        dis_to_end /= sample_num
        dis_to_end_from_act /= sample_num
        print("==================Average======================")
        print("Average MSE between obs produced from sampled actions and sampled obs: ", mse_total)
        print("Average MSE between final obs produced from sampled actions and sampled obs: ", mse_final_total)
        # print("Average MSE between final obs produced from sampled actions and target final obs: ", mse_final_target_total)
        # print("Average MSE between final sampled obs and target final obs: ", mse_final_target2_total)
        
        print("Average MSE between model-based minumum energy control and sampled actions: ", mse_u_total)
        print(f"Average MSE between final obs produced from model-based minumum energy control and target final obs: ", mse_u_target_total)
        
        print(f"Average MSE between data-driven minumum energy control and sampled actions: ", mse_u_data_total)
        print(f"Average MSE between data-driven minumum energy control approximation and sampled actions: ", mse_u_data_appr_total)
        print(f"Average MSE between final obs produced from data-driven minumum energy control and target final obs: ", mse_u_target_data_total)
        print(f"Average MSE between final obs produced from data-driven minumum energy control approximation and target final obs: ", mse_u_target_data_appr_total)
        
        
        print(f"Energies of model-based minumum energy control: {energy_u_total}, data-driven minumum energy control: {energy_u_data_total}, data-driven minumum energy control approximation: {energy_u_data_appr_total}, sampled actions: {energy_actions_total}")
        
        self.writer.add_scalar('val/obs', mse_total, self.step)
        self.writer.add_scalar('val/final_obs', mse_final_total, self.step)
        self.writer.add_scalar('val/energy', energy_actions_total, self.step)
        self.writer.add_scalar('val/energy_dd', energy_u_data_total, self.step)
        
        self.writer.add_scalar('val/mmd1', mmd1, self.step)
        self.writer.add_scalar('val/mmd2', mmd2, self.step)
        
        for i in range(self.env.max_T):
            self.writer.add_scalar(f'val2/obs_{i}', mses[i], self.step)
            self.writer.add_scalar(f'val3/distoend', dis_to_end[i], i)
            self.writer.add_scalar(f'val4/distoendfromact', dis_to_end_from_act[i], i)
            
        self.writer.flush()
        self.model.train()
        self.ema_model.train()
        return mse_total, mse_final_total, mse_final_target_total, mse_u_total, mse_u_target_total
    
    def sample_tensors(self, args, test_data=None, sample_num=10, fn_choose={'energy':1}, use_invdyn=False):
        self.model.eval()
        self.ema_model.eval()
        U_s = []
        Y_s = []
        num = 0
            # if test_data:
            #     random_int = np.random.randint(0, len(test_data))
            # with torch.no_grad():
            #     if test_data==None:
            #         if not self.kuramoto:
            #             y_f = np.random.randn(1, 1, self.env.num_observation) * self.sigma
            #             y_0 = np.zeros((1, 1, self.env.num_observation))
            #         else:
            #             n = self.env.num_observation
            #             theta1 = np.mod(0 * np.pi * np.arange(n) / n, 2 * np.pi)
            #             theta2 = np.mod(4 * np.pi * np.arange(n) / n, 2 * np.pi)
            #             y_f = theta2[np.newaxis, np.newaxis, :]
            #             y_0 = theta1[np.newaxis, np.newaxis, :]
                        
            #         y_0 = self.dataset.normalizer['Y'].normalize(y_0).to(self.device)
            #         y_f = self.dataset.normalizer['Y'].normalize(y_f).to(self.device)
            #         # y_0 = torch.tensor(y_0).to(self.device)
            #         # y_f = torch.tensor(y_f).to(self.device)
            #     else:
        assert test_data
        datal = cycle(torch.utils.data.DataLoader(
            test_data, batch_size=sample_num, num_workers=1, shuffle=True, pin_memory=True
        ))
        batch = next(datal)
        batch = batch_to_device(batch, device=self.device)
        # traj = batch.trajectories
        cond = batch.conditions
        for key,values in cond.items():
            cond[key] = values.unsqueeze(1)
        # denoiser_cond = batch.denoiser_conditions
        batch_size =  sample_num         
    
        # planning or one-shot
        if self.apply_guidance:
            try:
                guide = LossFunction_noparams(horizon=self.env.max_T+1, transition_dim=self.ema_model.transition_dim, observation_dim=self.ema_model.observation_dim, fn_choose=fn_choose, end_vector=y_f.squeeze())
                self.ema_model.set_guide_fn(guide)
            except:
                pass
            if isinstance(self.dataset, TrainData_norm_free):
                trajectories, _, _, guidances = self.ema_model(batch_size=batch_size, cond = cond,denoiser_cond=torch.ones(batch_size,self.ema_model.model.denoiser_cond_dim).to(self.device),horizon=self.env.max_T+1, apply_guidance=self.apply_guidance, guide_clean=self.guide_clean)
            else:  
                trajectories, _, _, guidances = self.ema_model(batch_size=batch_size, cond = cond, denoiser_cond=torch.ones(batch_size,self.ema_model.model.denoiser_cond_dim).to(self.device),horizon=self.env.max_T+1, apply_guidance=self.apply_guidance, guide_clean=self.guide_clean)
            # trajectories = trajectories[0:1]
    


        # trajectories = trajectories.cpu()
        
        if not use_invdyn:
            actions = trajectories[:, :-1, :self.ema_model.action_dim] # T, m
            # action = actions[0, 0]
            # observations = trajectories[0, 1:, self.ema_model.action_dim:] #  T, p
            observations = trajectories[:, :, self.ema_model.action_dim:] # T+1, p
        else:
            # observations = trajectories[0, 1:]
            observations = trajectories
            actions = []
            for i in range(self.env.max_T):
                obs_comb = torch.cat([trajectories[:, i, :], trajectories[:, i+1, :]], dim=-1)
                # obs_comb = obs_comb.reshape(-1, 2*self.ema_model.observation_dim)
                action = self.ema_model.inv_model(obs_comb)
                actions.append(action)
            actions = torch.stack(actions)
            actions_with_end = torch.cat([actions, torch.zeros((1, *actions.shape[1:])).to(actions.device)], dim=0)
            actions = actions.transpose(1,0) 
            actions_with_end = actions_with_end.transpose(1,0)
            trajectories = torch.cat([actions_with_end, trajectories], dim=-1)
        actions = actions.cpu()
        observations = observations.cpu()
        trajectories = trajectories.cpu()
        if self.normalized:
            actions = self.dataset.normalizer['U'].unnormalize(actions)
            observations = self.dataset.normalizer['Y'].unnormalize(observations)
            # if test_data:
            #     y_f = test_data.normalizer['Y'].unnormalize(y_f.cpu()).squeeze(0).squeeze(0)
            #     y_0 = test_data.normalizer['Y'].unnormalize(y_0.cpu()).squeeze(0).squeeze(0)
            # else:
            #     y_f = self.dataset.normalizer['Y'].unnormalize(y_f.cpu()).squeeze(0).squeeze(0)
            #     y_0 = self.dataset.normalizer['Y'].unnormalize(y_0.cpu()).squeeze(0).squeeze(0)
                
            #     obs_from_act = self.env.from_actions_to_obs_direct(actions, start=y_0)
                # if mse_loss(obs_from_act[-1].squeeze(),y_f.squeeze()) >0.5:
                #     continue
        U_s=actions
        Y_s=observations
        print(U_s.shape,Y_s.shape)
        
        return U_s, Y_s[:,:-1], Y_s[:,-1]