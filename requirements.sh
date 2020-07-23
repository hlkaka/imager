apt update
apt install wget htop vim screen python3-pip unzip

python3 -m pip install --user --upgrade pip
python3 -m pip install --user virtualenv

mkdir env
python3 -m venv env

source env/bin/activate

pip3 install -U tqdm segmentation-models-pytorch pytorch-lightning albumentations