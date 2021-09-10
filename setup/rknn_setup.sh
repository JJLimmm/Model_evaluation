
envName=rknn
hasDownloaded=false
condaName=miniconda3

while getopts n:d:a flag
do
    case "${flag}" in
        n) envName=${OPTARG};;
        f) hasDownloaded=${OPTARG};;
        a) condaName=${OPTARG};;
    esac
done

echo "Creating environment named $envName"

source ~/${condaName}/etc/profile.d/conda.sh

conda create --name $envName -y python=3.6
conda activate $envName

if [ "$hasDownloaded" = false ]; then
    echo "Downloading rknn toolkit package for Ubuntu and Python 3.6"
    rknnPackage=https://github.com/rockchip-linux/rknn-toolkit/releases/download/v1.6.1/rknn-toolkit-v1.6.1-packages.tar.gz
    packageName=rknn_package
    wget -O ${packageName}.tar.gz ${rknnPackage}
    tar -xzvf ${packageName}.tar.gz
    echo "Downloaded and unzipped/tar package from $rknnPackage"
fi

echo "Registering environment via nb_conda"
conda install -y nb_conda

pip install tensorflow==1.11.0
pip install torch==1.5.1 torchvision
pip install mxnet==1.5.0

if [ "$hasDownloaded" = false ]; then
    pip install packages/rknn_toolkit-1.6.1-cp36-cp36m-linux_x86_64.whl
else
    echo "Please run pip install on your exiting download of the rknn_toolkit"
fi

echo "Setup complete. Avoid using conda install to prevent breaking the rknn dependencies"

echo "You now can access your (mini)conda environment: ${envName}"