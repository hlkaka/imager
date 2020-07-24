apt update
apt upgrade

apt install wget htop vim screen python3-pip unzip libglib2.0-0 jupyterlab

pip install --upgrade pip
pip install virtualenv

mkdir env
python -m venv env

source env/bin/activate

# nightly pytorch
pip install --pre torch torchvision -f https://download.pytorch.org/whl/nightly/cu102/torch_nightly.html

pip install -U tqdm segmentation-models-pytorch pytorch-lightning albumentations SimpleITK