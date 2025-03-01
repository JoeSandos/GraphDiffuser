import numpy as np
import torch
import pickle
import sys
sys.path.append('..')
from env.env import Kuramoto, Power, IP, Burger
from utils.dataset import *

def load_data_and_env(args):
    with open('data/synthetic_data/'+args.data_name+'.pkl', 'rb') as f:
        pickle_data = pickle.load(f)
    if 'kuramoto' in args.data_name:
        sys_A, sys_B, sys_C, sys_k = pickle_data['sys']['A'], pickle_data['sys']['B'], pickle_data['sys']['C'], pickle_data['sys']['k']
        n, m, T, N = pickle_data['meta_data']['num_nodes'], pickle_data['meta_data']['input_dim'], pickle_data['meta_data']['control_horizon'], pickle_data['meta_data']['num_samples']
        p=n
        env = Kuramoto(sys_A, sys_B, sys_C, sys_k, T)
        
    elif 'power' in args.data_name:
        sys_C = pickle_data['sys']['C']
        n, m, T, N = pickle_data['meta_data']['num_nodes'], pickle_data['meta_data']['input_dim'], pickle_data['meta_data']['control_horizon'], pickle_data['meta_data']['num_samples']
        p=n
        env = Power(C=sys_C,m=m,n=n,T=T)
    elif 'inverted' in args.data_name:
        sys_C = pickle_data['sys']['C']
        n, m, T, N = pickle_data['meta_data']['num_nodes'], pickle_data['meta_data']['input_dim'], pickle_data['meta_data']['control_horizon'], pickle_data['meta_data']['num_samples']
        p=n
        env = IP(C=sys_C,m=m,n=n,T=T)
    elif 'burger' in args.data_name:
        sys_C = pickle_data['sys']['C']
        n, m, T, N = pickle_data['meta_data']['num_nodes'], pickle_data['meta_data']['input_dim'], pickle_data['meta_data']['control_horizon'], pickle_data['meta_data']['num_samples']
        m=n
        p=n
        env = Burger(C=sys_C,m=n,n=n,T=T)
    else:
        raise NotImplementedError
    U_3d, Y_bar_3d, Y_f_3d = pickle_data['data']['U'], pickle_data['data']['Y_bar'], pickle_data['data']['Y_f'] # U_3d-control signal: (N, T-1, m), Y_bar_3d-observate signal from 0~T-1: (N, T-1, p), Y_f_3d-observate final at T: (N, p)
    num_train = int(N*args.train_ratio)
    num_val = int(N*args.valid_ratio)
    num_test = int(N*args.test_ratio)
    print('num_train:', num_train, 'num_val:', num_val, 'num_test:', num_test)
    if args.resample and args.normalized:
        raise NotImplementedError
    elif args.resample and (not args.normalized):
        raise NotImplementedError
    elif args.normalized:
        if args.free_guide:
            train_data = TrainData_norm_free(U_3d[:num_train], Y_bar_3d[:num_train], Y_f_3d[:num_train], zero_pad=False, train=True, use_clustering=args.use_clustering, use_smoothness=args.use_smoothness)
            val_data = TrainData_norm_free(U_3d[num_train:num_train+num_val], Y_bar_3d[num_train:num_train+num_val], Y_f_3d[num_train:num_train+num_val], zero_pad=False, train=True, use_clustering=args.use_clustering, use_smoothness=args.use_smoothness)
        else:
            train_data = TrainData_norm(U_3d[:num_train], Y_bar_3d[:num_train], Y_f_3d[:num_train], zero_pad=False)
            val_data = TrainData_norm(U_3d[num_train:num_train+num_val], Y_bar_3d[num_train:num_train+num_val], Y_f_3d[num_train:num_train+num_val], zero_pad=False)
    else:
        raise NotImplementedError
    if args.sample_use_test:
        if args.normalized:
            if args.free_guide:
                test_data = TrainData_norm_free(U_3d[-num_test:], Y_bar_3d[-num_test:], Y_f_3d[-num_test:], zero_pad=False, use_clustering=args.use_clustering)
            else:
                test_data = TrainData_norm(U_3d[-num_test:], Y_bar_3d[-num_test:], Y_f_3d[-num_test:], zero_pad=False)
        else:
            raise NotImplementedError
    else:
        test_data = None
        
    return env, train_data, val_data, test_data, U_3d, Y_bar_3d, Y_f_3d