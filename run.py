# main_script.py
import os
import itertools
import time
import yaml
import argparse

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Generate commands for experiments.')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to configuration file.')
    return parser.parse_args()

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def generate_commands(config):
    """Generate commands based on configuration."""
    # Extract parameters
    params = config['parameters']
    exp_settings = config['experiment_settings']
    
    # Setup basic variables
    num_data_total = exp_settings['num_data_total']
    sw_dir = exp_settings['sw_dir']
    exp_name = exp_settings['exp_name']
    
    os.makedirs(sw_dir, exist_ok=True)
    
    # Get parameter lists and keys
    lists = list(params.values())
    key_names = list(params.keys())
    length = len(key_names)
    
    print('=================start commands=================')
    for combination in itertools.product(*lists):
        # Calculate specific numbers based on combination

        
        # Generate command
        command = f'python main.py'
        sw_name = '' + time.strftime(f'%a_%b_%d_%H:%M:%S_', time.localtime())
        
        # Add parameters to command
        for i in range(length):
            command += f' --{key_names[i]} {combination[i]}'
            sw_name += f'{key_names[i]}_{combination[i]}_'
        
        # Add experiment name and interaction number
        sw_name += exp_name
        
        # Add additional command parameters
        command += (f' --sw_dir {sw_dir} '
                   f'--sw_name {sw_name} '
                   f'--train_savepath ./results/{sw_name}/')
        
        print(command)
        os.system(command)
        time.sleep(2)
        print("==================================================")

if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config)
    generate_commands(config)