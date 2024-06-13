import torch
import torch.nn.functional as F

import numpy as np

from skimage import io
from skimage.util import img_as_float
from skimage.exposure import equalize_adapthist
from skimage.segmentation._slic import _enforce_label_connectivity_cython

from sklearn.decomposition import PCA as sklearnPCA
from sklearn.impute import SimpleImputer

from utils.swf import SideWindowBoxFilter


def handle_nan_values(img):
    # Get the number of channels, height and width of the image
    channels, height, width = img.shape

    # Convert image to shape (height * width, channels)
    reshaped_img = img.reshape(channels, height * width).T

    # Replacing NaN with average values
    imputer = SimpleImputer(strategy='mean')

    # Apply the imputer to our image
    imputed_img = imputer.fit_transform(reshaped_img)

    # Convert the image back to its original form (channels, height, width)
    imputed_img = imputed_img.T.reshape(channels, height, width)

    return imputed_img


def apply_PCA(img, n_components=3):
    # Get the number of channels, height and width of the image
    channels, height, width = img.shape

    # Convert the image to a shape (height * width, channels) to apply PCA
    reshaped_img = img.reshape(channels, height * width).T

    # Data normalization
    reshaped_img = (reshaped_img - np.mean(reshaped_img, axis=0)) / np.std(reshaped_img, axis=0)

    # Create a PCA object with the specified number of components
    pca = sklearnPCA(n_components=n_components)

    # Apply PCA to the image, reduce the number of channels
    transformed_data = pca.fit_transform(reshaped_img)

    # Convert the result back to 2D array
    transformed_img = transformed_data.T.reshape(n_components, height, width)

    return transformed_img


def preprocess(img,height, width, img_name = None, device="cuda"):
    if img_name != None:
        # If the image is 3-channel
        img = io.imread(img_name)
    else:
        # Convert back to 3D array (n_components, height, width)
        img = img.reshape(-1, height, width).transpose(1, 2, 0)

    # 0, 1, 2 channel - original image
    image = img_as_float(img)
    image = torch.from_numpy(image).permute(2, 0, 1).float()[None]
    # 3, 4, 5 channel - filtered and clahed image
    blur = SideWindowBoxFilter(device=device).to(device)
    img_bc = blur.forward(img)
    img_bc = img_as_float(equalize_adapthist(img_bc))
    img_bc = torch.from_numpy(img_bc).permute(2, 0, 1).float()[None]
    # 6, 7 channels - coordinates
    h, w = image.shape[-2:]
    coord = torch.stack(torch.meshgrid(torch.arange(h),
                                       torch.arange(w))).float()[None]

    # instance normalization
    img_in = torch.cat([image, img_bc, coord], 1).to(device)
    img_in = (img_in - img_in.mean((2, 3), keepdim=True)) / \
        img_in.std((2, 3), keepdim=True)

    # Normalizing img values to save an image
    img = image[0].permute(1, 2, 0).cpu().numpy()
    img = (img - img.min()) / (img.max() - img.min())

    return img, img_in


def calc_spixel(spixel_labels, num_spixel, device="cuda"):
    spix = spixel_labels.squeeze().to("cpu").detach().numpy()

    segment_size = spix.size / num_spixel
    min_size = int(0.06 * segment_size)
    max_size = int(3.0 * segment_size)
    spix = _enforce_label_connectivity_cython(spix[None], min_size,
                                              max_size)[0]

    return spix
