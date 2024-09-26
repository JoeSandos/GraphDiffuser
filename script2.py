import os
import itertools
import time
# 定义数据名称列表
data_names=[
    # 'erdos_renyi_5_5_5_15_1000',
    # 'kuramoto_8_8_15_1000_2_sigma=1'
    'power_18_9_31_100000_sigma=0.01',
    # 'kuramoto_8_8_15_100000_2_sigma=1',
    # 'kuramoto_8_8_15_100000_2_sigma=2',
    # 'kuramoto_8_8_15_10000_2_sigma=1_data_driven',
    # 'kuramoto_8_8_15_10000_2_sigma=1_combined',
    
    # 'kuramoto_8_8_15_10000_2_sigma=2_data_driven',
    
    
    # 'erdos_renyi_20_5_20_15_1000',
    # 'simple_2_1_2_7_1000_sigma_1'
    # 'erdos_renyi_100_5_100_15_1000',
    
]
num_data_total = 100000
# data_names = [
#     # 'erdos_renyi_5_5_5_15_1000',
#     # 'erdos_renyi_10_5_5_15_1000',
#     # 'erdos_renyi_15_5_5_15_1000',
#     # 'erdos_renyi_20_5_5_15_1000',
#     # 'erdos_renyi_5_5_5_15_1000',
#     # 'erdos_renyi_5_5_5_15_1000_data_driven',
    
#     # 'erdos_renyi_10_5_5_15_1000_data_driven',
#     # 'erdos_renyi_15_5_15_15_1000',
#     # 'erdos_renyi_10_5_5_15_1000_model_based',
#     # 'erdos_renyi_10_5_5_15_1000',
    
#     # 'erdos_renyi_10_5_10_15_1000',
#     # 'erdos_renyi_10_5_10_15_1000_model_based',
#     # 'erdos_renyi_10_5_10_15_1000_data_driven',
#     # 'erdos_renyi_20_5_5_15_1000',
#     # 'erdos_renyi_20_5_5_15_1000_data_driven',
#     'simple_2_1_2_7_1000_combined_0.3',
#     'simple_2_1_2_7_1000_combined_0.5',
#     'simple_2_1_2_7_1000_combined_0.7',
#     'simple_2_1_2_7_1000_model_based',
#     'simple_2_1_2_7_1000_sigma_1'

# ]
# data_names = [
    # 'erdos_renyi_25_5_5_15_1000',
    # 'erdos_renyi_30_5_5_15_1000',
    # 'erdos_renyi_25_5_25_15_1000',
    # 'erdos_renyi_30_5_30_15_1000',
    # 'erdos_renyi_10_5_5_15_1000',
    # 'erdos_renyi_5_5_5_15_1000',
    # 'erdos_renyi_10_5_5_15_1000',
    # 'erdos_renyi_15_5_5_15_1000',
    # 'erdos_renyi_20_5_5_15_1000',  
    # 'scale_free_20_5_5_15_1000_sigma_1',
    # 'scale_free_15_5_5_15_1000_sigma_1',
    # 'scale_free_10_5_5_15_1000_sigma_1',
    # 'scale_free_5_5_5_15_1000_sigma_1',
    
    # 'small_world_5_5_5_15_1000_sigma_1',
    # 'small_world_10_5_10_15_1000_sigma_1',
    # 'small_world_20_5_20_15_1000_sigma_1',
    # 'small_world_40_5_40_15_1000_sigma_1',
    
    
# ]
# data_names = [
#     # 'erdos_renyi_5_5_5_15_1000_sigma_1',
#     'erdos_renyi_5_5_5_15_1000',
#     'erdos_renyi_10_5_5_15_1000',
    
#     'erdos_renyi_20_5_5_15_1000',
#     'erdos_renyi_30_5_5_15_1000',     
# ]
params = {}
# params['pred_eps']=[0]
# params['no_cond'] = [1]
# params['normalized'] = [1]
# params['use_invdyn'] = [0,1]
# params['sigma'] = [5]
interact_ratio = [1] #interact num = interact_ratio * train_ratio*1000+ 200*interact_ratio*concat_ratio*loops)
# params['mixup'] = [0]
# params['use_attn'] = [0]
# params['has_invdyn'] = [1]
params['use_lambda'] = [0]
# params['use_end'] = [1]
# params['use_end_second'] = [1]
# params['free_guide'] = [1]
params['scale'] = [2,1]

# params['use_cond'] = [1,0]

params['n_timesteps'] = [64,128]
# params['valid_ratio'] =[2e-2]
# params['test_ratio'] =[5e-2]
# params['use_clustering'] =[1]
params['use_smoothness'] =['uni_first'] # use_smoothness in ['uni_first', 'uni_second', 'first', 'second', None]
# params['use_posenc'] = [1]

# params['use_invdyn'] = [1]

params['lr'] = [2e-3]
params['sample_use_test']=[1]
params['data_name'] = data_names
# params['retrain_data_name'] = retrain_data_names
# params['train_conditioning'] = [0]
# params['guide_clean'] = [1]
# params['scale'] = [10, 1, 0.1]
params['n_train_steps'] = [int(2e4)]
# params['seed'] = [44]
# params['concat'] = [0]
# params['concat'] = [1]
# params['regen'] = [1]
params['concat_ratio']=[1]
params['loops'] = [1]
params['train_ratio'] = [round(0.2*ratio,4) for ratio in interact_ratio]
# params['n_train_steps'] = [int(1e4)]
sw_dir = './runs/train_num_new_power/'
exp_name = 'power'

os.makedirs(sw_dir, exist_ok=True)
lists = list(params.values())
key_names = list(params.keys())
length = len(key_names)
print('=================start commands=================')
for combination in itertools.product(*lists):
    resample_num = int(num_data_total*combination[-1])
    interact_num = int(combination[-1]*num_data_total+resample_num*combination[-2]*combination[-3])
    command = f'python toy_guide_multi_power.py'
    sw_name = ''+time.strftime(f'%a_%b_%d_%H:%M:%S_', time.localtime())
    for i in range(length):
        command += f' --{key_names[i]} {combination[i]}'
        sw_name += f'{key_names[i]}_{combination[i]}_'
    sw_name += exp_name
    sw_name += f'_inter{interact_num}'
    command += f' --sw_dir {sw_dir} --sw_name {sw_name} --resample_num {resample_num} --train_savepath ./results/{sw_name}/'
    print(command)
    
    os.system(command)
    time.sleep(2)
    print("==================================================")
    # 运行命令行命令
