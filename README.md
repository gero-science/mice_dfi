# Mice dFI

## AI for unsupervised learning of aging principles from longitudinal data

This repository contains the proposed model in the paper
[https://www.biorxiv.org/content/10.1101/2020.01.23.917286v1](https://www.biorxiv.org/content/10.1101/2020.01.23.917286v1).
It also contains notebooks and data to reproduce the results from the paper.


## Abstract
We proposed and characterized a novel biomarker of aging and frailty in mice trained 
from the large set of the most conventional, easily measured blood parameters such as 
Complete Blood Counts (CBC) from the open-access Mouse Phenome Database (MPD).
Instead of postulating the existence of an aging clock associated with any particular 
subsystem of an aging organism, we assumed that aging arises cooperatively from positive
feedback loops spanning across physiological compartments and leading to an organism-level
instability of the underlying regulatory network. To analyze the data, we employed a 
deep artificial neural network including auto-encoder (AE) and auto-regression (AR) 
components. The AE was used for dimensionality reduction and denoising the data.
The AR was used to describe the dynamics of an individual mouse’s health state by means
of stochastic evolution of a single organism state variable, the “dynamic frailty index”
(dFI), that is the linear combination of the latent AE features and has the meaning of 
the total number of regulatory abnormalities developed up to the point of the measurement
or, more formally, the order parameter associated with the instability. 
We used neither the chronological age nor the remaining lifespan of the animals while 
training the model. Nevertheless, dFI fully described aging on the organism level, 
that is it increased exponentially with age and predicted remaining lifespan. 
Notably, dFI correlated strongly with multiple hallmarks of aging such as physiological 
frailty index, indications of physical decline, molecular markers of inflammation and
accumulation of senescent cells. The dynamic nature of dFI was demonstrated in mice 
subjected to aging acceleration by placement on a high-fat diet and aging deceleration 
by treatment with rapamycin.

## Requirements
1. Python >= 3.8, pip => 20.0
2. Required python packages are listed in [setup.py](setup.py). Will be installed automatically.
3. To run jupyter notebooks, you should be able to connect to a running [jupyter server](https://jupyter-notebook.readthedocs.io/en/stable/public_server.html)


## Installation
Use `pip` install to install a package. 
```bash
git clone https://github.com/gero-science/mice_dfi
cd mice_dfi
pip install .
```

## Obtaining MPD datasets
This study is mostly based on data from the [Mouse Phenome Database](https://phenome.jax.org/). 
To download the dataset used for training a model simply run the following command.
```bash
python -m mice_dfi.dataset.download
```
The other datasets used in this study are stored in [this repository](notebooks/generated).

## Training model

Start a model training with the command. Note, that datasets should be downloaded in prior
```bash
python -m mice_dfi.model.train -o dump -c ./src/mice_dfi/model/config/model_resnet.yaml --tb
```
or display command-line argument help.
```bash
python -m mice_dfi.model.train --help
```
File `model_resnet.yaml` could be modified for tuning neural network parameters, 
such as depth, activation and dropouts. 

## Notebooks

Notebooks are stored in the [notebooks](notebooks/) folder. Note, you have to install and run jupyter
server by yourself. 

## TODOs
 - Write missing documentation
 - Add learning rate scheduler and loss weights scheduler 

## License
The `mice_dfi` package is licensed under the [GNU GENERAL PUBLIC LICENSE Version 3 (GPLv3)](LICENSE)

