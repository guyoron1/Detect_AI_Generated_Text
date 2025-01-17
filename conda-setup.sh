# create your workshop directory
WORKDIR=/home/tkoren/mldaw2425/imryziv
mkdir $WORKDIR
cd $WORKDIR

# download and install Miniconda
wget <
https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh>
bash $WORKDIR/Miniconda3-latest-Linux-x86_64.sh -b -p $WORKDIR/miniconda3

# comment the following lines if you do not want to install the packages
source ~/.bashrc # this is necessary in order to use the newly installed conda command
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install notebook

rm Miniconda3-latest-Linux-x86_64.sh