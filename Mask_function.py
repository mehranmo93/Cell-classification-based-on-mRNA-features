import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_cdt
from mpl_toolkits.axes_grid1 import ImageGrid
from skimage.transform import resize
import imageio
import cv2
import os
from PIL import Image
from glob import glob

# Define the fixed image size
image_size = (250, 250)

# Input directory containing .npz files
input_directory = "/cluster/simulations/"

# Output directory to save .ppm images
output_directory = "/cluster/simulations_mask_complete_func/"

# List all the .npz files in the input directory
npz_files = glob(os.path.join(input_directory, "*.npz"))

npz_files = sorted(npz_files, key=lambda x: os.path.basename(x).split('-')[0])

def custom_mapping(x, max):
    if x == 0:
        return 0
    else:
        return max - (x - 1)


def custom_mapping2(x, max):
    if x == 0:
        return 0
    else:
        return x
def mask_creator(npz_file, i):
    # Load the .npz file
    data = np.load(npz_file)
    mask_nuc_data = data['nuc_mask']
    mask_cyt_data = data['cell_mask']

    mask_cyt_data = mask_cyt_data.astype(int)
    mask_nuc_data = mask_nuc_data.astype(int)

    # Calculate the subtracted image
    Subtract = mask_cyt_data - mask_nuc_data

    # Scale the image to a fixed size of 250x250
    scaled_image_sub = resize(Subtract, (250, 250), anti_aliasing=True)
    scaled_image_nuc = resize(mask_nuc_data, (250, 250), anti_aliasing=True)
    scaled_image_cyt = resize(mask_cyt_data, (250, 250), anti_aliasing=True)

    fig = plt.figure(figsize=(5, 5))
    grid = ImageGrid(fig, 111, nrows_ncols=(1, 1), axes_pad=0)

    for ax in grid:
        ax.axis('off')
    ax = grid[0]

    distance_taxicab = distance_transform_cdt(scaled_image_sub, metric="taxicab")
    taxicab_transform = ax.imshow(distance_taxicab, cmap='gray')

    norm = taxicab_transform.norm

    clim_min, clim_max = norm.vmin, norm.vmax

    # Apply the transformation element-wise
    transformed_image = np.vectorize(custom_mapping)(distance_taxicab, clim_max)
    distance_taxicab_int = (transformed_image * 255 / clim_max).astype(np.uint8)

    # Save the image as .ppm in the output directory
    png_filename = os.path.join(output_directory, f"{name}_sub_mask_{i}.png")

    imageio.imwrite(png_filename, distance_taxicab_int)

    fig = plt.figure(figsize=(5, 5))
    grid = ImageGrid(fig, 111, nrows_ncols=(1, 1), axes_pad=0)

    for ax in grid:
        ax.axis('off')
    ax = grid[0]
    distance_taxicab_nuc = distance_transform_cdt(scaled_image_nuc, metric="taxicab")
    taxicab_transform_nuc = ax.imshow(distance_taxicab_nuc, cmap='gray')

    norm_nuc = taxicab_transform_nuc.norm

    clim_min_nuc, clim_max_nuc = norm_nuc.vmin, norm_nuc.vmax

    # Apply the transformation element-wise
    transformed_image_nuc = np.vectorize(custom_mapping)(distance_taxicab_nuc, clim_max_nuc)
    distance_taxicab_int_nuc = (transformed_image_nuc * 255 / clim_max_nuc).astype(np.uint8)

    # Save the image as .ppm in the output directory
    png_filename_nuc = os.path.join(output_directory, f"{name}_nuc_mask_{i}.png")
    imageio.imwrite(png_filename_nuc, distance_taxicab_int_nuc)

    fig = plt.figure(figsize=(5, 5))
    grid = ImageGrid(fig, 111, nrows_ncols=(1, 1), axes_pad=0)

    for ax in grid:
        ax.axis('off')
    ax = grid[0]
    distance_taxicab_cyt = distance_transform_cdt(scaled_image_cyt, metric="taxicab")
    taxicab_transform_cyt = ax.imshow(distance_taxicab_cyt, cmap='gray')

    norm_cyt = taxicab_transform_cyt.norm

    clim_min_cyt, clim_max_cyt = norm_cyt.vmin, norm_cyt.vmax

    # Apply the transformation element-wise
    transformed_image_cyt_normal = np.vectorize(custom_mapping2)(distance_taxicab_cyt, clim_max_cyt)
    distance_taxicab_int_cyt_normal = (transformed_image_cyt_normal * 255 / clim_max_cyt).astype(np.uint8)
    transformed_image_cyt = np.vectorize(custom_mapping)(distance_taxicab_cyt, clim_max_cyt)
    distance_taxicab_int_cyt = (transformed_image_cyt * 255 / clim_max_cyt).astype(np.uint8)

    # Save the image as .ppm in the output directory
    png_filename_cyt_normal = os.path.join(output_directory, f"{name}_cyt_mask_{i}.png")
    imageio.imwrite(png_filename_cyt_normal, distance_taxicab_int_cyt_normal)

    # Save the image as .ppm in the output directory
    png_filename_cyt = os.path.join(output_directory, f"{name}_cyt_reverse_mask_{i}.png")
    imageio.imwrite(png_filename_cyt, distance_taxicab_int_cyt)


i , j, k, l, n, m, b, v = 0, 0, 0, 0, 0, 0, 0, 0
# Loop through each .npz file
for npz_file in npz_files:

    filename = os.path.basename(npz_file).split('.')[0]
    # Extract information from the filename
    name, _, _, _, _ = filename.split('-')

    if 'cell_edge'== name:
        i += 1
        mask_creator(npz_file, i)
    elif 'extranuclear'== name:
        j += 1
        mask_creator(npz_file, j)
    elif 'foci' == name:
        k += 1
        mask_creator(npz_file, k)
    elif 'intranuclear' == name:
        l += 1
        mask_creator(npz_file, l)
    elif 'nuclear_edge' == name:
        n += 1
        mask_creator(npz_file, n)
    elif 'pericellular' == name:
        m += 1
        mask_creator(npz_file, m)
    elif 'perinuclear' == name:
        b += 1
        mask_creator(npz_file, b)
    elif 'random' == name:
        v += 1
        mask_creator(npz_file, v)
    else:
        print("there is an unexpected file")
        break



