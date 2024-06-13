# fundamental configuration
IMG_PATH = "./imgs/"
OUT_IMG = "./results/bdry/"
OUT_NPY = "./results/npy/"

LBL_PATH = "../../ssn_spixel/data/BSR/BSDS500/data/groundTruth/test/"

# parameters for SpixelCNN initialization
NUM_SPIXELS = 100
IN_CHANNELS = 5
NUM_FEAT = 32
NUM_LAYERS = 4

# parameters for SpixelCNN optimization
NUM_ITER = 1000
LR = 1e-2
LOSS_WEIGHTS = [1, 2, 10]
SC_WEIGHTS = [1, .75]
THRESH = 0
COEF_CARD = 1.5
SIGMA = 2
MARGIN = 1

# List of image file extensions
image_extensions = ['.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp', '.gif']