# SEDC: Sample-Efficient Diffusion-based Control

## Introduction

This repository contains the implementation of **SEDC**, a novel diffusion-based control framework for complex nonlinear physical systems. 

![Alt text](asset\results.png)

## Installation

Configure the environment by running the following command:

```bash
pip install -r requirements.txt
```

## Usage

### Steps to Run the Demo
**① Data Generation**

You can generate the data with the following command:
```bash
python data/generate_[model].py 
```

Currently available `model` are `burgers`, `kuramoto`. 


Finally, store the data in the `data\synthetic_data` folder.

**② Configuration**
Config files .yaml are located in the `configs` folder. Check the notes in the main.py file to understand the parameters. Change ''Parameter'' in the yaml to the desired values.

Config files .yaml are located in the `configs` folder. Check the notes in the `main.py` file to understand the parameters. Change `Parameter` in the yaml to the desired values. Change `data_name` to the name of the dataset you generated or downloaded.


**③ Running the Code**
To run the code, use the following command:

```bash
python run.py --config configs/[config_file].yaml
```


## Repository Structure


```

 ┣ 📂asset
 ┃ ┗ 📜results.png
 ┣ 📂configs
 ┃ ┣ 📜burgers.yaml
 ┃ ┗ 📜kuramoto.yaml
 ┣ 📂data
 ┃ ┣ 📂synthetic_data
 ┃ ┃ ┗ 📜readme.md
 ┃ ┣ 📜generate_burgers.py
 ┃ ┗ 📜generate_kuramoto.py
 ┣ 📂env
 ┃ ┗ 📜env.py
 ┣ 📂model
 ┃ ┣ 📜attention.py
 ┃ ┣ 📜diffusion.py
 ┃ ┣ 📜diffusion_inv.py
 ┃ ┣ 📜dynamic.py
 ┃ ┣ 📜guide.py
 ┃ ┣ 📜helpers.py
 ┃ ┗ 📜temporal.py
 ┣ 📂utils
 ┃ ┣ 📜arrays.py
 ┃ ┣ 📜dataset.py
 ┃ ┣ 📜distance.py
 ┃ ┣ 📜er_syn_data copy 2.ipynb
 ┃ ┣ 📜er_syn_data copy.ipynb
 ┃ ┣ 📜er_syn_data.ipynb
 ┃ ┣ 📜normalization.py
 ┃ ┣ 📜simple_data.py
 ┃ ┣ 📜syn_data.py
 ┃ ┣ 📜trainer.py
 ┃ ┗ 📜utils.py
 ┣ 📜main.py
 ┣ 📜README.md
 ┣ 📜requirements.txt
 ┗ 📜run.py
```

## Notes

- The complete version, including additional environments and datasets, will be made available upon manuscript acceptance.
