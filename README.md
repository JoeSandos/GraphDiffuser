# Introduction

Implementation of the work "A Diffusion-based Generative Approach for Model-free Finite-time Control of Complex Systems"

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
Config files .yaml are located in the `configs` folder. To run the code, use the following command:

```bash
python run.py --config configs/[config_file].yaml
```
