import torch
import torch.nn as nn
import einops
from einops.layers.torch import Rearrange
import pdb
from .attention import SpatialTransformer1D
from .helpers import (
    SinusoidalPosEmb,
    Downsample1d,
    Upsample1d,
    Conv1dBlock,
    Residual,
    PreNorm,
    LinearAttention,
)


class ResidualTemporalBlock(nn.Module):

    def __init__(self, inp_channels, out_channels, embed_dim, kernel_size=5):
        super().__init__()

        self.blocks = nn.ModuleList([
            Conv1dBlock(inp_channels, out_channels, kernel_size),
            Conv1dBlock(out_channels, out_channels, kernel_size),
        ])

        self.time_mlp = nn.Sequential(
            nn.Mish(),
            nn.Linear(embed_dim, out_channels),
            Rearrange('batch t -> batch t 1'),
        )

        self.residual_conv = nn.Conv1d(inp_channels, out_channels, 1) \
            if inp_channels != out_channels else nn.Identity()

    def forward(self, x, t):
        '''
            x : [ batch_size x inp_channels x horizon ]
            t : [ batch_size x embed_dim ]
            returns:
            out : [ batch_size x out_channels x horizon ]
        '''
        out = self.blocks[0](x) + self.time_mlp(t)
        out = self.blocks[1](out)
        return out + self.residual_conv(x)


class TemporalUnet(nn.Module):

    def __init__(
        self,
        transition_dim,
        cond_dim,
        dim=32,
        dim_mults=(1, 2, 4, 8),
        attention=False,
    ):
        super().__init__()

        dims = [transition_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        print(f'[ models/temporal ] Channel dimensions: {in_out}')

        time_dim = dim
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, dim * 4),
            nn.Mish(),
            nn.Linear(dim * 4, dim),
        )

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        print(in_out)
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                ResidualTemporalBlock(dim_in, dim_out, embed_dim=time_dim),
                ResidualTemporalBlock(dim_out, dim_out, embed_dim=time_dim),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))) if attention else nn.Identity(),
                Downsample1d(dim_out) if not is_last else nn.Identity()
            ]))


        mid_dim = dims[-1]
        self.mid_block1 = ResidualTemporalBlock(mid_dim, mid_dim, embed_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, LinearAttention(mid_dim))) if attention else nn.Identity()
        self.mid_block2 = ResidualTemporalBlock(mid_dim, mid_dim, embed_dim=time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(nn.ModuleList([
                ResidualTemporalBlock(dim_out * 2, dim_in, embed_dim=time_dim),
                ResidualTemporalBlock(dim_in, dim_in, embed_dim=time_dim),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))) if attention else nn.Identity(),
                Upsample1d(dim_in) if not is_last else nn.Identity()
            ]))


        self.final_conv = nn.Sequential(
            Conv1dBlock(dim, dim, kernel_size=5),
            nn.Conv1d(dim, transition_dim, 1),
        )

    def forward(self, x, cond, time, returns=None):
        '''
            x : [ batch x horizon x transition ]
        '''

        x = einops.rearrange(x, 'b h t -> b t h')

        t = self.time_mlp(time)
        h = []

        for resnet, resnet2, attn, downsample in self.downs:
            x = resnet(x, t)
            x = resnet2(x, t)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for resnet, resnet2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, t)
            x = resnet2(x, t)
            x = attn(x)
            x = upsample(x)

        x = self.final_conv(x)

        x = einops.rearrange(x, 'b t h -> b h t')
        return x



class CondTemporalUnet(nn.Module):

    def __init__(
        self,
        transition_dim,
        cond_dim,
        dim=32,
        dim_mults=(1, 2, 4, 8),
        attention=False,
    ):
        super().__init__()

        dims = [transition_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        print(f'[ models/temporal ] Channel dimensions: {in_out}')

        time_dim = dim
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, dim * 4),
            nn.Mish(),
            nn.Linear(dim * 4, dim),
        )

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)
        self.d_cond = cond_dim
        print(in_out)
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                ResidualTemporalBlock(dim_in, dim_out, embed_dim=time_dim),
                ResidualTemporalBlock(dim_out, dim_out, embed_dim=time_dim),
                # Residual(PreNorm(dim_out, LinearAttention(dim_out))) if attention else nn.Identity(),
                SpatialTransformer1D(channels=dim_out, n_heads=1, n_layers=1, d_cond=self.d_cond),
                Downsample1d(dim_out) if not is_last else nn.Identity()
            ]))


        mid_dim = dims[-1]
        self.mid_block1 = ResidualTemporalBlock(mid_dim, mid_dim, embed_dim=time_dim)
        # self.mid_attn = Residual(PreNorm(mid_dim, LinearAttention(mid_dim))) if attention else nn.Identity()
        self.mid_attn = SpatialTransformer1D(channels=mid_dim, n_heads=1, n_layers=1, d_cond=self.d_cond)
        self.mid_block2 = ResidualTemporalBlock(mid_dim, mid_dim, embed_dim=time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(nn.ModuleList([
                ResidualTemporalBlock(dim_out * 2, dim_in, embed_dim=time_dim),
                ResidualTemporalBlock(dim_in, dim_in, embed_dim=time_dim),
                # Residual(PreNorm(dim_in, LinearAttention(dim_in))) if attention else nn.Identity(),
                SpatialTransformer1D(channels=dim_in, n_heads=1, n_layers=1, d_cond=self.d_cond),
                Upsample1d(dim_in) if not is_last else nn.Identity()
            ]))


        self.final_conv = nn.Sequential(
            Conv1dBlock(dim, dim, kernel_size=5),
            nn.Conv1d(dim, transition_dim, 1),
        )

    def forward(self, x, cond, time, returns=None):
        '''
            x : [ batch x horizon x transition ]
        '''

        x = einops.rearrange(x, 'b h tr -> b tr h')

        t = self.time_mlp(time)
        h = []

        for resnet, resnet2, attn, downsample in self.downs:
            x = resnet(x, t)
            x = resnet2(x, t)
            x = attn(x, cond)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x,cond)
        x = self.mid_block2(x, t)

        for resnet, resnet2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, t)
            x = resnet2(x, t)
            x = attn(x, cond)
            x = upsample(x)

        x = self.final_conv(x)

        x = einops.rearrange(x, 'b t h -> b h t')
        return x


class ValueFunction(nn.Module):

    def __init__(
        self,
        horizon,
        transition_dim,
        cond_dim,
        dim=32,
        dim_mults=(1, 2, 4, 8),
        out_dim=1,
    ):
        super().__init__()

        dims = [transition_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        time_dim = dim
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, dim * 4),
            nn.Mish(),
            nn.Linear(dim * 4, dim),
        )

        self.blocks = nn.ModuleList([])
        num_resolutions = len(in_out)

        print(in_out)
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.blocks.append(nn.ModuleList([
                ResidualTemporalBlock(dim_in, dim_out, kernel_size=5, embed_dim=time_dim, horizon=horizon),
                ResidualTemporalBlock(dim_out, dim_out, kernel_size=5, embed_dim=time_dim, horizon=horizon),
                Downsample1d(dim_out)
            ]))

            if not is_last:
                horizon = horizon // 2

        mid_dim = dims[-1]
        mid_dim_2 = mid_dim // 2
        mid_dim_3 = mid_dim // 4
        ##
        self.mid_block1 = ResidualTemporalBlock(mid_dim, mid_dim_2, kernel_size=5, embed_dim=time_dim, horizon=horizon)
        self.mid_down1 = Downsample1d(mid_dim_2)
        horizon = horizon // 2
        ##
        self.mid_block2 = ResidualTemporalBlock(mid_dim_2, mid_dim_3, kernel_size=5, embed_dim=time_dim, horizon=horizon)
        self.mid_down2 = Downsample1d(mid_dim_3)
        horizon = horizon // 2
        ##
        fc_dim = mid_dim_3 * max(horizon, 1)

        self.final_block = nn.Sequential(
            nn.Linear(fc_dim + time_dim, fc_dim // 2),
            nn.Mish(),
            nn.Linear(fc_dim // 2, out_dim),
        )

    def forward(self, x, cond, time, *args):
        '''
            x : [ batch x horizon x transition ]
        '''

        x = einops.rearrange(x, 'b h t -> b t h') # COMM: (BS, horizon, state+action dim) -> (BS, state+action dim, horizon) 
        # COMM：in 1-D conv, the last dim is the time dim, the second last dim is the channel dim
        ## mask out first conditioning timestep, since this is not sampled by the model
        # x[:, :, 0] = 0

        t = self.time_mlp(time) # COMM: (BS, 1) -> (BS, dim=32)

        for resnet, resnet2, downsample in self.blocks:
            x = resnet(x, t)
            x = resnet2(x, t)
            x = downsample(x)

        ##
        x = self.mid_block1(x, t)
        x = self.mid_down1(x)
        ##
        x = self.mid_block2(x, t)
        x = self.mid_down2(x)
        ##
        x = x.view(len(x), -1)
        out = self.final_block(torch.cat([x, t], dim=-1))
        return out


# class ValueFunction_noparams(nn.Module):
#     def __init__(
#         self,
#         horizon,
#         transition_dim,
#         observation_dim,
#         out_dim=1,
#         fn_choose={'specific_end':1, 'any_end':1, 'energy':1},
#         end_vector = None
#     ):
#         self.horizon = horizon
#         self.transition_dim = transition_dim
#         self.action_dim = transition_dim - observation_dim
#         self.observation_dim = observation_dim
#         self.out_dim = out_dim
#         assert len(end_vector) == self.observation_dim
#         self.end_vector = end_vector
#         self.fn_choose = fn_choose
#         super().__init__()
        
#     def forward(self, x, cond=None, time=None, *args):
#         rewards = []
#         reward_dict = {}
#         for fn,weight in self.fn_choose.items():
#             if fn == 'specific_end':
#                 reward = self.any_end(x) * weight
#             elif fn == 'any_end':
#                 reward = self.any_end(x) * weight
#             elif fn == 'energy':
#                 reward = self.energy(x) * weight
#             else:
#                 raise NotImplementedError
#             rewards.append(reward)
#             reward_dict[fn] = reward
#         # normalize rewards using l1 norm
#         rewards = torch.stack(rewards, dim=-1)
#         reward_by_sample = rewards.mean(dim=-1).detach().squeeze()
#         rewards = rewards.mean()
#         # the range of reward is [0,1]
#         return rewards, reward_dict, reward_by_sample
            
        
#     def any_end(self, x):
#         assert x.shape[-1]==self.transition_dim
#         # a focal weight is applied
#         power = list(range(self.horizon))
#         power = torch.tensor(power, dtype=torch.float32).to(x.device)
#         # inverse
#         power = power.flip(0)

#         power = torch.exp(-power)
#         power = power / power.sum()
#         # apply focal weight
#         l2_dist = torch.norm(x[:, :, self.action_dim:] - self.end_vector, dim=-1)
#         reward = l2_dist * power.unsqueeze(0)
#         reward = torch.exp(-reward.sum(dim=-1, keepdim=True))
#         # the range of reward is [0,1], the larger the distance, the smaller the reward
#         return reward
    
#     def specific_end(self, x):
#         assert x.shape[-1]==self.transition_dim
#         # only the final state is considered
#         l2_dist = torch.norm(x[:, -1, self.action_dim:] - self.end_vector, dim=-1).unsqueeze(-1)
#         reward = torch.exp(-l2_dist)
#         return reward
    
        
#     def energy(self, x):
#         assert x.shape[-1]==self.transition_dim
#         # the energy is calculated based on the action
#         energy = torch.norm(x[:, :, :self.action_dim], dim=-1)
#         reward = torch.exp(-energy.sum(dim=-1, keepdim=True)/self.horizon/self.action_dim)
#         # the range of reward is [0,1], the larger the energy, the smaller the reward
#         return reward
    
class LossFunction_noparams(nn.Module):
    def __init__(
        self,
        horizon,
        transition_dim,
        observation_dim,
        out_dim=1,
        fn_choose={'specific_end':1, 'any_end':1, 'energy':1},
        end_vector = None
    ):
        self.horizon = horizon
        self.transition_dim = transition_dim
        self.action_dim = transition_dim - observation_dim
        self.observation_dim = observation_dim
        self.out_dim = out_dim
        assert len(end_vector) == self.observation_dim
        self.end_vector = end_vector
        self.fn_choose = fn_choose
        super().__init__()
        
    def forward(self, x, cond=None, time=None, *args):
        losses = []
        loss_dict = {}
        for fn,weight in self.fn_choose.items():
            if fn == 'specific_end':
                loss = self.specific_end(x) * weight
            elif fn == 'any_end':
                loss = self.any_end(x) * weight
            elif fn == 'energy':
                loss = self.energy(x) * weight
            else:
                raise NotImplementedError
            losses.append(loss)
            loss_dict[fn] = loss
        # normalize rewards using l1 norm
        losses = torch.stack(losses, dim=-1)
        loss_by_sample = losses.mean(dim=-1).detach().squeeze()
        losses = losses.mean()
        # the range of reward is [0,1]
        return losses, loss_dict, loss_by_sample
            
        
    def any_end(self, x):
        assert x.shape[-1]==self.transition_dim
        # a focal weight is applied
        power = list(range(self.horizon)) # power = [0,1,2,...,horizon-1]
        power = torch.tensor(power, dtype=torch.float32).to(x.device)
        # inverse
        power = power.flip(0) # power = [horizon-1,horizon-2,...,0]

        power = torch.exp(-power) # power = [exp(-horizon+1),exp(-horizon+2),...,exp(-1),exp(0)]
        power = power / power.sum() # power = [exp(-horizon+1)/sum,exp(-horizon+2)/sum,...,exp(-1)/sum,exp(0)/sum]
        # apply focal weight
        l2_dist = torch.norm(x[:, :, self.action_dim:] - self.end_vector, dim=-1)
        loss = l2_dist * power.unsqueeze(0)
        loss = loss.sum(dim=-1, keepdim=True)
        # the larger the distance, the larger the loss
        return loss
    
    def specific_end(self, x):
        assert x.shape[-1]==self.transition_dim
        # only the final state is considered
        l2_dist = torch.norm(x[:, -1, self.action_dim:] - self.end_vector, dim=-1).unsqueeze(-1)
        # reward = torch.exp(-l2_dist)
        return l2_dist
    
        
    def energy(self, x):
        assert x.shape[-1]==self.transition_dim
        # the energy is calculated based on the action
        energy = (x[:, :, :self.action_dim]**2).sum(dim=-1)
        energy = energy.sum(dim=-1, keepdim=True)
        # reward = torch.exp(-energy.sum(dim=-1, keepdim=True)/self.horizon/self.action_dim)
        # the range of reward is [0,1], the larger the energy, the smaller the reward
        return energy