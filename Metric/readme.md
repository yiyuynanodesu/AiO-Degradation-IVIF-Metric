### Dependencies and Installation
```
# Create a conda environment and activate it
conda create -n clipiqa python=3.8 -y
conda activate clipiqa
# Install PyTorch following official instructions, e.g.
conda install pytorch=1.10 torchvision cudatoolkit=11.3 -c pytorch
# Install pre-built MMCV using MIM.
pip3 install openmim
mim install mmcv-full==1.5.0
# Install CLIP-IQA from the source code.
git clone git@github.com:IceClear/CLIP-IQA.git
cd CLIP-IQA
pip install -r requirements.txt
pip install -e .
```