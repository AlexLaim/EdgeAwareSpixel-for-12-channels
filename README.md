# Edge-aware Superpixel Segmentation with Unsupervised Convolutional Neural Networks

The open source code is taken from the repository:
`https://github.com/yueyu-stu/EdgeAwareSpixel`

## Environment
Main requirement: Pytorch 
If you are having problems installing PyTorch visit the website:
`https://pytorch.org/`

## Usage

1. The libraries must be installed before you can start working:
   `pip install -r requirements.txt`

2. Adjust parameters in `utils/config.py`  
   * `IMG_PATH` and `LBL_PATH` should be customized to the dataset folders.  
   * New folders should be created to store generated images and NPY data. If the desired number of superpixels is N, the format is `img/to/path/N/`, where `img/to/path` is a user-defined folder name.
   * `NUM_SPIXELS` inidicates the number of superpixels.

3. Generate superpixels  
   Run `python inference.py`


