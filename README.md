# Introduction

Implementation of the work "Sample-efficient diffusion-based control of complex nonlinear systems"

# Installation

Configure the environment by running the following command:

```bash
pip install -r requirements.txt
```

# Usage

## Data Generation

```bash
python data/generate_[model].py 
```

## Configuration and Running
Config files .yaml are located in the `configs` folder. Check the notes in the main.py file to understand the parameters. Change ''Parameter'' in the yaml to the desired values.

To run the code, use the following command:

```bash
python run.py --config configs/[config_file].yaml
```


