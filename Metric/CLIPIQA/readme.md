### Dependencies and Installation
```
# Create a conda environment and activate it
conda create -n metric python=3.8 -y
conda activate metric
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