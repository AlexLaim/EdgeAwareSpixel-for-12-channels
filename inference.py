import os

import numpy as np
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt

import torch
import rioxarray as rxr

from utils.config import *
from utils.data_process import preprocess, calc_spixel, apply_PCA, handle_nan_values
from models.spixel_model import SpixelCNN

# initialization
device = "cuda" if torch.cuda.is_available() else "cpu"

# get dataset
img_list = os.listdir(IMG_PATH)

# get superpixels in inference process
for img_name in img_list:
    # File extension
    extension = os.path.splitext(img_name)[1]
    if extension.lower() in image_extensions:
        print(img_name)

        # Read GeoTiff
        img_rxr = rxr.open_rasterio(os.path.join(IMG_PATH, img_name), masked=True)

        # Convert to ndarray
        img_arr = img_rxr.to_numpy()

        # PCA 12 -> 3 channels
        if np.isnan(img_arr).any():
            img_arr = handle_nan_values(img_arr)

        # Applying PCA to an image
        img_reshaped = apply_PCA(img_arr)

        # image preprocessing
        if img_arr.shape[0] == 12:
            channels, height, width = img_arr.shape
            img, img_in = preprocess(img_reshaped, height, width)
        else:
            img, img_in = preprocess(None, 0, 0, os.path.join(IMG_PATH, img_name))

        # get model
        model = SpixelCNN(num_spixels=NUM_SPIXELS,
                          img_shape=img.shape[:2],
                          in_c=IN_CHANNELS,
                          num_feat=NUM_FEAT,
                          num_layers=NUM_LAYERS,
                          device=device).to(device)

        # model weights initialization
        model.weight_init()

        # optimization
        model.optimize(img_in=img_in,
                       img=img,
                       num_iter=NUM_ITER,
                       lr=LR,
                       loss_weights=LOSS_WEIGHTS,
                       sc_weights=SC_WEIGHTS,
                       thresh=THRESH,
                       coef_card=COEF_CARD,
                       sigma=SIGMA,
                       margin=MARGIN,
                       device=device)

        # generate sparse assignment matrix spxiel_prob
        spixel_prob, _ = model.forward(img_in)

        # obtain superpixel labels, save segmented image and .npy file for measurement
        label = calc_spixel(spixel_prob.argmax(1).long(), NUM_SPIXELS)
        plt.imsave(
            os.path.join(OUT_IMG, str(NUM_SPIXELS), img_name[:-4] + "_bdry.png"),
            mark_boundaries(img, label))
        np.save(os.path.join(OUT_NPY, str(NUM_SPIXELS), img_name[:-4] + "_spixel"),
                label)
    else:
        print(f"The file is not an image: {img_name}")
