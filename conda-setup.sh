# Create your workshop directory
WORKDIR=/home/tkoren/mldaw2425/imryziv
mkdir -p $WORKDIR
cd $WORKDIR

# Download and install Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash $WORKDIR/Miniconda3-latest-Linux-x86_64.sh -b -p $WORKDIR/miniconda3

# Comment the following lines if you do not want to install the packages
source ~/.bashrc # This is necessary to use the newly installed conda command
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install notebook

# Download requirements.txt for Detect_AI_Generated_Text
PROJECT_DIR=$WORKDIR/Detect_AI_Generated_Text
mkdir -p $PROJECT_DIR
wget -O $PROJECT_DIR/requirements.txt https://example.com/path/to/Detect_AI_Generated_Text/requirements.txt

# Install dependencies from requirements.txt
pip install -r $PROJECT_DIR/requirements.txt

# Clean up the Miniconda installer
rm $WORKDIR/Miniconda3-latest-Linux-x86_64.sh
