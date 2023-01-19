# FedGSL

The code for the paper "Subgraph Federated Learning with Global Graph Reconstruction"



## Acknowledgment

The Implementation of our code is based on FederatedScope V2.0 :https://github.com/alibaba/FederatedScope



## Quick Start

We provide an end-to-end example for users to start running a standard FL course with FederatedScope.

### Step 1. Install FederatedScope 

First of all, users need to clone the source code and install the required packages (we suggest python version >= 3.9). 

```bash
git clone https://github.com/poipoipoi233/FedGSL.git
cd FederatedScope
```

#### Use Conda to Install backend (PyTorch)

We recommend using a new virtual environment to install FederatedScope:

```bash
conda create -n fs python=3.9
conda activate fs
```

```bash
conda install -y pytorch=1.10.1 torchvision=0.11.2 torchaudio=0.10.1 torchtext=0.11.1 cudatoolkit=11.3 -c pytorch -c conda-forge
```

For users with Apple M1 chips:

```bash
conda install pytorch torchvision torchaudio -c pytorch
# Downgrade torchvision to avoid segmentation fault
python -m pip install torchvision==0.11.3
```

Finally, after the backend is installed, you can install FederatedScope from `source`:

##### From source

```bash
pip install .

# Or (for dev mode)
pip install -e .[dev]
pre-commit install
```

##### Install graph-related dependencies

```bash
conda install -y pyg==2.0.4 -c pyg
conda install -y rdkit=2021.09.4=py39hccf6a74_0 -c conda-forge
conda install -y nltk
```



### Step 2. Run FedGSL

```bash
cd federatedscope/FedGSL
```

##### For windows user:

```bash
# /bin/sh run_fedgsl.sh {cuda_id} {dataset}
# example:
/bin/sh run_fedgsl.sh 0 citeseer
```

##### For Linux user:

```bash
sh run_fedgsl.sh 0 citeseer
```

