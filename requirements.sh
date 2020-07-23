apt update
apt install wget htop vim screen python3-pip unzip libglib2.0-0

pip3 install --upgrade pip
pip3 install virtualenv

mkdir env
python3 -m venv env

source env/bin/activate

pip3 install -U tqdm segmentation-models-pytorch pytorch-lightning albumentations SimpleITK