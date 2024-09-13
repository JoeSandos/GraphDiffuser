from collections import namedtuple
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import pdb
from tqdm import tqdm
# import diffuser.utils as utils
from .helpers import (
    cosine_beta_schedule,
    extract,
    apply_conditioning,
    Losses,
)
from .temporal import TemporalUnetInvdyn

Sample = namedtuple('Sample', 'trajectories values chains guidances')


@torch.no_grad()
def default_sample_fn(model, x, cond, t):
    model_mean, _, model_log_variance = model.p_mean_variance(x=x, cond=cond, t=t)
    model_std = torch.exp(0.5 * model_log_variance)

    # no noise when t == 0
    noise = torch.randn_like(x)
    noise[t == 0] = 0

    values = torch.zeros(len(x), device=x.device)
    return model_mean + model_std * noise, values


@torch.no_grad()
def n_step_guided_p_sample(
    model, x, cond, t, guide, scale=0.001, t_stopgrad=0, n_guide_steps=1, scale_grad_by_std=True,
):
    """
    COMM:
    Sample from the guided diffusion model for n steps at a given time step t.
    first guide the noisy input x_t, then update x_t with the gradients of the guide
    next sample from the model distribution given guided x_t
    produce the mean and log of x_t-1
    
    """
    model_log_variance = extract(model.posterior_log_variance_clipped, t, x.shape)
    model_std = torch.exp(0.5 * model_log_variance)
    model_var = torch.exp(model_log_variance)

    for _ in range(n_guide_steps):
        with torch.enable_grad():
            y, grad = guide.gradients(x, cond, t)

        if scale_grad_by_std: # COMM : usually True
            grad = model_var * grad

        grad[t < t_stopgrad] = 0 # COMM : stop grad after t_stopgrad

        x = x + scale * grad
        x = apply_conditioning(x, cond, model.action_dim)

    model_mean, _, model_log_variance = model.p_mean_variance(x=x, cond=cond, t=t)

    # no noise when t == 0
    noise = torch.randn_like(x)
    noise[t == 0] = 0

    return model_mean + model_std * noise, y

def sort_by_values(x, values, decending = True): #COMM: sort x by values, biggest value first
    inds = torch.argsort(values, descending=decending)
    x = x[inds]
    values = values[inds]
    return x, values


def make_timesteps(batch_size, i, device):
    t = torch.full((batch_size,), i, device=device, dtype=torch.long)
    return t


class GaussianDiffusion(nn.Module):
    def __init__(self, model, horizon, observation_dim, action_dim, n_timesteps=1000,
        loss_type='l2', clip_denoised=False, predict_epsilon=True,
        action_weight=1.0, loss_discount=1.0, loss_weights=None,
    ):
        super().__init__()
        self.horizon = horizon
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.transition_dim = observation_dim + action_dim
        self.model = model

        betas = cosine_beta_schedule(n_timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.n_timesteps = int(n_timesteps)
        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)

        ## log calculation clipped because the posterior variance
        ## is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped',
            torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer('posterior_mean_coef1',
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))

        ## get loss coefficients and initialize objective
        loss_weights = self.get_loss_weights(action_weight, loss_discount, loss_weights)
        # print(loss_weights)
        self.loss_fn = Losses[loss_type](loss_weights, self.action_dim)

    def get_loss_weights(self, action_weight, discount, weights_dict):
        '''
            sets loss coefficients for trajectory

            action_weight   : float
                coefficient on first action loss
            discount   : float
                multiplies t^th timestep of trajectory loss by discount**t
            weights_dict    : dict
                { i: c } multiplies dimension i of observation loss by c
        '''
        self.action_weight = action_weight

        dim_weights = torch.ones(self.transition_dim, dtype=torch.float32)

        ## set loss coefficients for dimensions of observation
        if weights_dict is None: weights_dict = {}
        for ind, w in weights_dict.items():
            dim_weights[self.action_dim + ind] *= w

        ## decay loss with trajectory timestep: discount**t
        discounts = discount ** torch.arange(self.horizon, dtype=torch.float)
        discounts = discounts / discounts.mean()
        loss_weights = torch.einsum('h,t->ht', discounts, dim_weights)

        ## manually set a0 weight
        loss_weights[0, :self.action_dim] = action_weight
        return loss_weights

    #------------------------------------------ sampling ------------------------------------------#

    def predict_start_from_noise(self, x_t, t, noise):
        '''
            if self.predict_epsilon, model output is (scaled) noise;
            otherwise, model predicts x0 directly
        '''
        if self.predict_epsilon:
            return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )
        else:
            return noise

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, cond, t):
        x_recon = self.predict_start_from_noise(x, t=t, noise=self.model(x, cond, t))

        if self.clip_denoised:
            x_recon.clamp_(-1., 1.)
        else:
            pass

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
                x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample_loop(self, shape, cond, verbose=True, return_chain=False, sample_fn=default_sample_fn, **sample_kwargs):
        device = self.betas.device

        batch_size = shape[0]
        x = torch.randn(shape, device=device)
        x = apply_conditioning(x, cond, self.action_dim)

        chain = [x] if return_chain else None

        for i in reversed(range(0, self.n_timesteps)):
            t = make_timesteps(batch_size, i, device)
            x, values = sample_fn(self, x, cond, t, **sample_kwargs) #COMM; sample x_{t-1} from x_t
            x = apply_conditioning(x, cond, self.action_dim) # COMM: remenber to apply conditioning on x_t-1 (start and endpoint, etc)

            if return_chain: chain.append(x)


        x, values = sort_by_values(x, values)
        if return_chain: chain = torch.stack(chain, dim=1)
        return Sample(x, values, chain, None)

    @torch.no_grad()
    def conditional_sample(self, cond, batch_size, horizon=None, **sample_kwargs):
        '''
            conditions : [ (time, state), ... ]
        '''
        device = self.betas.device
        # batch_size = batch_size
        horizon = horizon or self.horizon
        shape = (batch_size, horizon, self.transition_dim)
        # print(shape)
        return self.p_sample_loop(shape, cond, **sample_kwargs)

    #------------------------------------------ training ------------------------------------------#

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sample = (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

        return sample

    def p_losses(self, x_start, cond, t):
        noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_noisy = apply_conditioning(x_noisy, cond, self.action_dim)

        x_recon = self.model(x_noisy, cond, t)
        if not self.predict_epsilon:
            x_recon = apply_conditioning(x_recon, cond, self.action_dim)

        assert noise.shape == x_recon.shape

        if self.predict_epsilon:
            loss, info = self.loss_fn(x_recon, noise)
        else:
            loss, info = self.loss_fn(x_recon, x_start)

        return loss, info

    def loss(self, x, *args):
        batch_size = len(x)
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()
        return self.p_losses(x, *args, t)

    def forward(self, cond, batch_size=1, *args, **kwargs):
        return self.conditional_sample(cond, batch_size, *args, **kwargs)

class GaussianInvDynDiffusion(nn.Module):
    def __init__(self, model, horizon, observation_dim, action_dim, n_timesteps=1000,
        loss_type='l2', clip_denoised=False, predict_epsilon=False, hidden_dim=256,
        action_weight=1.0, loss_discount=1.0, loss_weights=None, returns_condition=False,
        condition_guidance_w=0.1, ar_inv=False, train_only_inv=False, diffusion_weight=10.0):
        super().__init__()
        self.horizon = horizon
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.transition_dim = observation_dim + action_dim
        self.model = model
        self.ar_inv = ar_inv
        self.train_only_inv = train_only_inv
        self.diffusion_weight = diffusion_weight
        if self.ar_inv:
            self.inv_model = ARInvModel(hidden_dim=hidden_dim, observation_dim=observation_dim, action_dim=action_dim)
        else:
            self.inv_model = nn.Sequential(
                nn.Linear(2 * self.observation_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, self.action_dim),
            )
        self.returns_condition = returns_condition
        self.condition_guidance_w = condition_guidance_w

        betas = cosine_beta_schedule(n_timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.n_timesteps = int(n_timesteps)
        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)

        ## log calculation clipped because the posterior variance
        ## is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped',
            torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer('posterior_mean_coef1',
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))

        ## get loss coefficients and initialize objective
        loss_weights = self.get_loss_weights(loss_discount)
        self.loss_fn = Losses['l2'](loss_weights, 0)

    def get_loss_weights(self, discount):
        '''
            sets loss coefficients for trajectory

            action_weight   : float
                coefficient on first action loss
            discount   : float
                multiplies t^th timestep of trajectory loss by discount**t
            weights_dict    : dict
                { i: c } multiplies dimension i of observation loss by c
        '''
        self.action_weight = 1
        dim_weights = torch.ones(self.observation_dim, dtype=torch.float32)

        ## decay loss with trajectory timestep: discount**t
        discounts = discount ** torch.arange(self.horizon, dtype=torch.float)
        discounts = discounts / discounts.mean()
        loss_weights = torch.einsum('h,t->ht', discounts, dim_weights)
        # Cause things are conditioned on t=0
        if self.predict_epsilon:
            loss_weights[0, :] = 0

        return loss_weights

    #------------------------------------------ sampling ------------------------------------------#

    def predict_start_from_noise(self, x_t, t, noise):
        '''
            if self.predict_epsilon, model output is (scaled) noise;
            otherwise, model predicts x0 directly
        '''
        if self.predict_epsilon:
            return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )
        else:
            return noise

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, cond, t, returns=None):
        if self.returns_condition:
            # epsilon could be epsilon or x0 itself
            epsilon_cond = self.model(x, cond, t, returns, use_dropout=False)
            epsilon_uncond = self.model(x, cond, t, returns, force_dropout=True)
            epsilon = epsilon_uncond + self.condition_guidance_w*(epsilon_cond - epsilon_uncond)
        else:
            epsilon = self.model(x, cond, t)

        t = t.detach().to(torch.int64)
        x_recon = self.predict_start_from_noise(x, t=t, noise=epsilon)

        if self.clip_denoised:
            x_recon.clamp_(-1., 1.)
        else:
            pass

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
                x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, cond, t, returns=None):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, cond=cond, t=t, returns=returns)
        noise = 0.5*torch.randn_like(x)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, shape, cond, returns=None, verbose=True, return_diffusion=False):
        device = self.betas.device

        batch_size = shape[0]
        x = torch.randn(shape, device=device)
        x = apply_conditioning(x, cond, 0)

        if return_diffusion: diffusion = [x]

        for i in reversed(range(0, self.n_timesteps)):
            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x = self.p_sample(x, cond, timesteps, returns)
            x = apply_conditioning(x, cond, 0)
    

            if return_diffusion: diffusion.append(x)


        if return_diffusion:
            return x, torch.stack(diffusion, dim=1)
        else:
            return x

    @torch.no_grad()
    def conditional_sample(self, cond, batch_size, returns=None, horizon=None, *args, **kwargs):
        '''
            conditions : [ (time, state), ... ]
        '''
        device = self.betas.device
        # batch_size = len(cond[0])
        horizon = horizon or self.horizon
        shape = (batch_size, horizon, self.observation_dim)

        return self.p_sample_loop(shape, cond, returns, *args, **kwargs)
    #------------------------------------------ training ------------------------------------------#

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sample = (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

        return sample

    def p_losses(self, x_start, cond, t, returns=None):
        noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_noisy = apply_conditioning(x_noisy, cond, 0)
        x_recon = self.model(x_noisy, cond, t, returns)

        if not self.predict_epsilon:
            x_recon = apply_conditioning(x_recon, cond, 0)

        assert noise.shape == x_recon.shape

        if self.predict_epsilon:
            loss, info = self.loss_fn(x_recon, noise)
        else:
            loss, info = self.loss_fn(x_recon, x_start)

        return loss, info

    def loss(self, x, cond, returns=None):
        if self.train_only_inv:
            # Calculating inv loss
            x_t = x[:, :-1, self.action_dim:]
            a_t = x[:, :-1, :self.action_dim]
            x_t_1 = x[:, 1:, self.action_dim:]
            x_comb_t = torch.cat([x_t, x_t_1], dim=-1)
            x_comb_t = x_comb_t.reshape(-1, 2 * self.observation_dim)
            a_t = a_t.reshape(-1, self.action_dim)
            if self.ar_inv:
                loss = self.inv_model.calc_loss(x_comb_t, a_t)
                info = {'a0_loss':loss}
            else:
                pred_a_t = self.inv_model(x_comb_t)
                loss = F.mse_loss(pred_a_t, a_t)
                info = {'a0_loss': loss}
        else:
            batch_size = len(x)
            t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()
            diffuse_loss, info = self.p_losses(x[:, :, self.action_dim:], cond, t, returns)
            # Calculating inv loss
            x_t = x[:, :-1, self.action_dim:]
            a_t = x[:, :-1, :self.action_dim]
            x_t_1 = x[:, 1:, self.action_dim:]
            x_comb_t = torch.cat([x_t, x_t_1], dim=-1)
            x_comb_t = x_comb_t.reshape(-1, 2 * self.observation_dim)
            a_t = a_t.reshape(-1, self.action_dim)
            if self.ar_inv:
                inv_loss = self.inv_model.calc_loss(x_comb_t, a_t)
            else:
                pred_a_t = self.inv_model(x_comb_t)
                inv_loss = F.mse_loss(pred_a_t, a_t)

            info['diffuse_loss'] = diffuse_loss.item()
            info['inv_loss'] = inv_loss.item()
            loss = (1 / 2) * (self.diffusion_weight * diffuse_loss + inv_loss)

        return loss, info

    def forward(self, cond, batch_size=1, returns=None, *args, **kwargs):
        return self.conditional_sample(cond=cond, batch_size=batch_size, returns=returns, *args, **kwargs)


class GaussianDiffusionClassifierGuided(nn.Module):
    def __init__(self, model, horizon, observation_dim, action_dim, n_timesteps=1000,
        loss_type='l2', clip_denoised=False, predict_epsilon=True,
        action_weight=1.0, loss_discount=1.0, loss_weights=None, scale=0.01, inv_dyn=False, 
        train_conditioning=True, use_lambda = False
    ):
        super().__init__()
        self.horizon = horizon
        self.observation_dim = observation_dim
        if not inv_dyn:
            self.real_action_dim = action_dim
            self.action_dim = action_dim
            self.transition_dim = observation_dim + action_dim
        else:
            self.real_action_dim = action_dim
            self.action_dim = 0
            self.transition_dim = observation_dim
        self.model = model
        self.train_conditioning = train_conditioning
        self.use_lambda = use_lambda
        
        if inv_dyn:
            self.inv_model = ARInvModel(hidden_dim=64, observation_dim=observation_dim, action_dim=action_dim)
        
        betas = cosine_beta_schedule(n_timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.n_timesteps = int(n_timesteps)
        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon
        self.use_inv_dyn = inv_dyn
        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)

        ## log calculation clipped because the posterior variance
        ## is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped',
            torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer('posterior_mean_coef1',
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))

        ## get loss coefficients and initialize objective
        loss_weights = self.get_loss_weights(action_weight, loss_discount, loss_weights)
        if self.use_lambda:
            loss_weights = loss_weights.unsqueeze(0).repeat(16,1,1)
        # print(loss_weights)
        self.loss_fn = Losses[loss_type](loss_weights, self.action_dim)
        self.scale = scale

    def get_loss_weights(self, action_weight, discount, weights_dict):
        '''
            sets loss coefficients for trajectory

            action_weight   : float
                coefficient on first action loss
            discount   : float
                multiplies t^th timestep of trajectory loss by discount**t
            weights_dict    : dict
                { i: c } multiplies dimension i of observation loss by c
        '''
        self.action_weight = action_weight

        dim_weights = torch.ones(self.transition_dim, dtype=torch.float32)

        ## set loss coefficients for dimensions of observation
        if weights_dict is None: weights_dict = {}
        for ind, w in weights_dict.items():
            dim_weights[self.action_dim + ind] *= w

        ## decay loss with trajectory timestep: discount**t
        discounts = discount ** torch.arange(self.horizon, dtype=torch.float)
        discounts = discounts / discounts.mean()
        loss_weights = torch.einsum('h,t->ht', discounts, dim_weights)

        ## manually set a0 weight
        # loss_weights[0, :self.action_dim] = action_weight
        loss_weights[:, :self.action_dim] = action_weight #COMM: set all action weights to action_weight
        
        return loss_weights

    #------------------------------------------ sampling ------------------------------------------#

    def predict_start_from_noise(self, x_t, t, noise):
        '''
            if self.predict_epsilon, model output is (scaled) noise;
            otherwise, model predicts x0 directly
        '''
        if self.predict_epsilon:
            return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )
        else:
            return noise

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, cond, t):
        x_tmp = x.detach()
        t_inp = t
        x_model_in = x
        x_recon = self.predict_start_from_noise(x_tmp, t=t, noise=self.model(x_model_in, cond, t_inp))
        # if t.max() == 63:
        #     pdb.set_trace()

        if self.clip_denoised:
            x_recon.clamp_(-1., 1.)
        else:
            pass

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
                x_start=x_recon, x_t=x_tmp, t=t)
        return model_mean, posterior_variance, posterior_log_variance, (x_recon, x_tmp, t)

    def set_guide_fn(self, guide):
        self.current_guidance = guide
    
    def guidance(self, x, num_samp=1, return_grad_of=None):
        '''
        estimate the gradient of rule reward w.r.t. the input trajectory
        Input:
            x: [num_samp, time_steps, feature_dim].  scaled input trajectory.
            return_grad_of: which variable to take gradient of guidance loss wrt, if not given,
                            takes wrt the input x.
        '''
        assert self.current_guidance is not None, 'Must instantiate guidance object before calling'
        with torch.enable_grad():

            # compute losses and gradient
            tol_loss, loss_dict, loss_by_sample = self.current_guidance(x)
            # print(tot_loss)
            tol_loss.backward()
            guide_grad = x.grad if return_grad_of is None else return_grad_of.grad

            return guide_grad, loss_dict, loss_by_sample
    
    @torch.no_grad()
    def p_sample(self, x, cond, t, returns=None, apply_guidance=True, guide_clean=False, eval_final_guide_loss=True):
        b, *_, device = *x.shape, x.device
        with_func = torch.no_grad
        if apply_guidance and guide_clean:
            # will need to take grad wrt noisy input
            x = x.detach()
            x.requires_grad_()
            with_func = torch.enable_grad

        with with_func():
            # get prior mean and variance for next step
            # model_mean, _, model_log_variance, q_posterior_in = self.p_mean_variance(x=x, t=t, aux_info=aux_info,
                                                                                    # class_free_guide_w=class_free_guide_w)
            model_mean, _, model_log_variance, q_posterior_in = self.p_mean_variance(x=x, cond=cond, t=t)
        # no noise or guidance when t == 0
        #       i.e. use the mean of the distribution predicted at the final step rather than sampling.
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))

        noise = torch.randn_like(model_mean)
        sigma = (0.5 * model_log_variance).exp()

        # compute guidance
        guide_losses = None
        guide_grad = torch.zeros_like(model_mean)
        if apply_guidance:
            if guide_clean:
                # want to guide the predicted clean traj from model, not the noisy one
                model_clean_pred = q_posterior_in[0]
                x_guidance = model_clean_pred
                return_grad_of = x
            else:
                x_guidance = model_mean.clone().detach()
                return_grad_of = x_guidance
                x_guidance.requires_grad_()

            guide_grad, guide_rewards, guide_sample= self.guidance(x_guidance, num_samp=1, return_grad_of=return_grad_of)
            guide_grad = nonzero_mask * guide_grad #* sigma
            
        noise = nonzero_mask * sigma * noise

        if guide_clean:
            # perturb clean trajectory
            guided_clean = q_posterior_in[0] - guide_grad*self.scale
            # use the same noisy input again
            guided_x_t = q_posterior_in[1]
            # re-compute next step distribution with guided clean & noisy trajectories
            model_mean, _, _ = self.q_posterior(x_start=guided_clean,
                                                x_t=guided_x_t,
                                                t=q_posterior_in[2])
            # NOTE: variance is not dependent on x_start, so it won't change. Therefore, fine to use same noise.
            x_out = model_mean + noise
        else:
            x_out = model_mean - guide_grad*self.scale + noise
        if eval_final_guide_loss:
            # eval guidance loss one last time for filtering if desired
            #       (even if not applied during sampling)
            _, guide_rewards, guide_sample = self.guidance(x_out.clone().detach().requires_grad_(), num_samp=1)
        if apply_guidance:
            return x_out, guide_rewards, guide_sample
        else:
            return x_out, None, None

    @torch.no_grad()
    def p_sample_loop(self, shape, cond, returns=None, verbose=True, return_diffusion=False, apply_guidance=True,
                    guide_clean=False):
        device = self.betas.device

        batch_size = shape[0]
        x = torch.randn(shape, device=device)
        x = apply_conditioning(x, cond, self.action_dim)

        diffusion = [x]
        guide_sample = 0
        steps = [i for i in reversed(range(0, self.n_timesteps))]
        # steps = tqdm(steps)
        conds = [i for i in cond.values()]
        conds = torch.cat(conds, dim=1)
        conds = conds.repeat(batch_size, 1, 1)
        for i in steps:
            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)
            # TODO :fix invdyn_bug
            if self.use_inv_dyn:
                apply_guidance = 0
            x, guide_rewards, guide_sample = self.p_sample(x, conds, timesteps, returns,
                                      apply_guidance=apply_guidance,
                                            guide_clean=guide_clean,
                                            eval_final_guide_loss=(i == steps[-1]))
            x = apply_conditioning(x, cond, self.action_dim)
    

            if return_diffusion: diffusion.append(x)
        # if guide_rewards is not None:
        #     print('===== GUIDANCE LOSSES ======')
        #     for k,v in guide_rewards.items():
        #         print('%s: %.012f' % (k, np.nanmean(v.cpu())))
        if not self.use_inv_dyn:
            x, guide_sample = sort_by_values(x, guide_sample, decending=False)

        if return_diffusion:
            diffusion = torch.stack(diffusion, dim=1)
        
        return Sample(x, guide_sample, diffusion, guide_rewards)
        
    # def p_sample_loop(self, shape, cond, verbose=True, return_chain=False, sample_fn=default_sample_fn, **sample_kwargs):
    #     device = self.betas.device

    #     batch_size = shape[0]
    #     x = torch.randn(shape, device=device)
    #     x = apply_conditioning(x, cond, self.action_dim)

    #     chain = [x] if return_chain else None

    #     for i in reversed(range(0, self.n_timesteps)):
    #         t = make_timesteps(batch_size, i, device)
    #         x, values = sample_fn(self, x, cond, t, **sample_kwargs) #COMM; sample x_{t-1} from x_t
    #         x = apply_conditioning(x, cond, self.action_dim) # COMM: remenber to apply conditioning on x_t-1 (start and endpoint, etc)

    #         if return_chain: chain.append(x)


    #     x, values = sort_by_values(x, values)
    #     if return_chain: chain = torch.stack(chain, dim=1)
    #     return Sample(x, values, chain)

    @torch.no_grad()
    def conditional_sample(self, cond, batch_size, horizon=None, **sample_kwargs):
        '''
            conditions : [ (time, state), ... ]
        '''
        device = self.betas.device
        # batch_size = batch_size
        horizon = horizon or self.horizon
        shape = (batch_size, horizon, self.transition_dim)
        # print(shape)
        return self.p_sample_loop(shape, cond, **sample_kwargs)

    #------------------------------------------ training ------------------------------------------#

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sample = (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

        return sample

    def p_losses(self, x_start, cond, t, mask=None):
        noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        if self.train_conditioning:
            x_noisy = apply_conditioning(x_noisy, cond, self.action_dim)

        conds = [i for i in cond.values()]
        conds = torch.stack(conds, dim=1)
        x_recon = self.model(x_noisy, conds, t)
        # if t.max()==40:
        #     pdb.set_trace()
        if self.train_conditioning:
            if not self.predict_epsilon:
                x_recon = apply_conditioning(x_recon, cond, self.action_dim)

        assert noise.shape == x_recon.shape

        if mask is not None:
            self.loss_fn.weights = mask
        if self.predict_epsilon:
            loss, info = self.loss_fn(x_recon, noise)
        else:
            loss, info = self.loss_fn(x_recon, x_start)

        return loss, info

    def loss(self, x, *args):
        batch_size = len(x)
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()
        if self.use_inv_dyn:
            x_t = x[:, :-1, self.real_action_dim:]
            a_t = x[:, :-1, :self.real_action_dim]
            x_t_1 = x[:, 1:, self.real_action_dim:]
            x_comb_t = torch.cat([x_t, x_t_1], dim=-1)
            x_comb_t = x_comb_t.reshape(-1, 2 * self.observation_dim)
            a_t = a_t.reshape(-1, self.real_action_dim)
            inv_loss = torch.functional.F.mse_loss(self.inv_model(x_comb_t), a_t)
            diffuse_loss, info = self.p_losses(x[:, :, self.real_action_dim:], *args, t)
            info['diffuse_loss'] = diffuse_loss.item()
            info['inv_loss'] = inv_loss.item()
            return (1 / 2) * (diffuse_loss + inv_loss), info
        else:
            mask=None
            # mask = None
            if self.use_lambda:
                # the type of model should be TemporalUnetInvdyn
                assert isinstance(self.model, TemporalUnetInvdyn)
                mask = torch.ones_like(x)
            # mask last action
                mask[:, -1, :self.real_action_dim] = 0
                lamb=0.99
                lamb_power_t = lamb**t.float() # size: [batch_size]
                # apply to mask action:
                mask[:, :, :self.real_action_dim] = mask[:, :, :self.real_action_dim] * lamb_power_t.unsqueeze(1).unsqueeze(2)
            return self.p_losses(x, *args, t, mask=mask)

    def forward(self, cond, batch_size=1, *args, **kwargs):
        return self.conditional_sample(cond, batch_size, *args, **kwargs)


class ARInvModel(nn.Module):
    def __init__(self, hidden_dim, observation_dim, action_dim, low_act=-1.0, up_act=1.0):
        super(ARInvModel, self).__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim

        self.action_embed_hid = 64
        self.out_lin = 64
        # self.num_bins = 80

        # self.up_act = up_act
        # self.low_act = low_act
        # self.bin_size = (self.up_act - self.low_act) / self.num_bins
        # self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()

        self.state_embed = nn.Sequential(
            nn.Linear(3 * self.observation_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.lin_mod = nn.ModuleList([nn.Linear(i, self.out_lin) for i in range(1,self.action_dim)])
        self.act_mod = nn.ModuleList()
        self.act_mod.append(nn.Sequential(nn.Linear(hidden_dim, self.action_embed_hid), nn.ReLU(),
                              nn.Linear(self.action_embed_hid, 1)))

        for _ in range( self.action_dim-1):
            self.act_mod.append(
                nn.Sequential(nn.Linear(hidden_dim + self.out_lin, self.action_embed_hid), nn.ReLU(),
                              nn.Linear(self.action_embed_hid, 1)))

    def forward(self, comb_state):
        # state_inp = comb_state
        state1 = comb_state[..., :self.observation_dim]
        state2 = comb_state[..., self.observation_dim:]
        state_diff = state2 - state1
        state_inp = torch.cat([state1, state_diff, state2], dim=-1)
        state_d = self.state_embed(state_inp)
        
        lp_0 = self.act_mod[0](state_d)
        a = [lp_0]
        # l_0 = torch.distributions.Categorical(logits=lp_0).sample()

        # if deterministic:
        #     a_0 = self.low_act + (l_0 + 0.5) * self.bin_size
        # else:
        #     a_0 = torch.distributions.Uniform(self.low_act + l_0 * self.bin_size,
        #                                       self.low_act + (l_0 + 1) * self.bin_size).sample()


        for i in range(1, self.action_dim):
            lp_i = self.act_mod[i](torch.cat([state_d, self.lin_mod[i-1](torch.cat(a, dim=-1))], dim=1))
            # l_i = torch.distributions.Categorical(logits=lp_i).sample()

            # if deterministic:
            #     a_i = self.low_act + (l_i + 0.5) * self.bin_size
            # else:
            #     a_i = torch.distributions.Uniform(self.low_act + l_i * self.bin_size,
            #                                       self.low_act + (l_i + 1) * self.bin_size).sample()
            a.append(lp_i)


        out = torch.cat(a, dim=-1)
        return out 

    # def calc_loss(self, comb_state, action):
    #     eps = 1e-8
    #     action = torch.clamp(action, min=self.low_act + eps, max=self.up_act - eps)
    #     l_action = torch.div((action - self.low_act), self.bin_size, rounding_mode='floor').long()
    #     state_inp = comb_state

    #     state_d = self.state_embed(state_inp)
    #     loss = self.ce_loss(self.act_mod[0](state_d), l_action[:, 0])

    #     for i in range(1, self.action_dim):
    #         loss += self.ce_loss(self.act_mod[i](torch.cat([state_d, self.lin_mod[i - 1](action[:, :i])], dim=1)),
    #                                  l_action[:, i])

    #     return loss/self.action_dim