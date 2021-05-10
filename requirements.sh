apt update
apt -y upgrade

apt -y install wget htop vim unzip libglib2.0-0 libsm6 libxext6 libxrender-dev

pip install --upgrade pip
pip install virtualenv

mkdir env
python -m venv env

source env/bin/activate

pip install --upgrade pip

# 1.6 CUDA 10.2+
pip install torch torchvision

pip install -U tqdm segmentation-models-pytorch pytorch-lightning albumentations SimpleITK jupyterlab torchsummary unidecode

pip install -U --no-deps Kornia

#wget https://hnscc-ds.nyc3.cdn.digitaloceanspaces.com/organized_dataset_2.zip

#echo "Unzipping downloaded dataset"

#unzip -q organized_dataset_2.zip