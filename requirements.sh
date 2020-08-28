apt update
apt -y upgrade

apt -y install wget htop vim unzip libglib2.0-0 libsm6 libxext6 libxrender-dev

pip install --upgrade pip
pip install virtualenv

mkdir env
python -m venv env

source env/bin/activate

# nightly pytorch
pip install --pre torch torchvision -f https://download.pytorch.org/whl/nightly/cu102/torch_nightly.html

pip install -U tqdm segmentation-models-pytorch pytorch-lightning albumentations SimpleITK jupyterlab torchsummary

pip install -U --no-deps Kornia

wget https://hnscc-ds.nyc3.cdn.digitaloceanspaces.com/organized_dataset_2.zip