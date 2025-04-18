#apt update
#apt -y upgrade

#apt -y install wget htop vim unzip libglib2.0-0 libsm6 libxext6 libxrender-dev

#sudo apt install python3-pip
pip3 install --upgrade pip
pip3 install virtualenv

#mkdir env
#python -m venv env

#source env/bin/activate

pip3 install --upgrade pip

# 1.6 CUDA 10.2+
#pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

pip3 install -U tqdm tensorboard segmentation-models-pytorch pytorch-lightning==1.8.4 albumentations SimpleITK jupyterlab torchsummary unidecode torchsort segmentation-models Kornia

wget https://hnscc-ds.nyc3.cdn.digitaloceanspaces.com/organized_dataset_2.zip

echo "Unzipping downloaded dataset"

unzip -q organized_dataset_2.zip
