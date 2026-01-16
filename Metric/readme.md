### Dependencies and Installation
```
# CLIPIQA Requirement
# Create a conda environment and activate it
conda create -n clipiqa python=3.8 -y
conda activate clipiqa
# Install PyTorch following official instructions, e.g.
pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
# Install pre-built MMCV using MIM.
pip3 install openmim
mim install mmcv-full==1.5.0
# Install CLIP-IQA from the source code.
git clone git@github.com:IceClear/CLIP-IQA.git
cd CLIP-IQA
pip install -r requirements.txt
pip install -e .
```

```
# TReS Requirement （you need download pretrain weight first）
cd TReS
conda create -n TReS python=3.8 -y
conda activate TReS
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install scipy
pip install openpyxl
pip install tqdm
```