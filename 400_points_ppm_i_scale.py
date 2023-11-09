import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import numpy as np
from glob import glob
from PIL import Image

# Define the fixed image size
image_size = (250, 250)

# Input directory containing .npz files
input_directory = "/cluster/simulations/"

# Output directory to save .ppm images
output_directory = "/cluster/simulations_400_RGB/"

# List all the .npz files in the input directory
npz_files = glob(os.path.join(input_directory, "*.npz"))

npz_files = sorted(npz_files, key=lambda x: os.path.basename(x).split('-')[0])

i = 0
prev_filename = ""

# Loop through each .npz file
for npz_file in npz_files:

    # Extract the filename without extension
    filename = os.path.basename(npz_file).split('.')[0]

    # Extract information from the filename
    name, _, _, num_points, _ = filename.split('-')
    num_points = int(num_points)

    # Check if the file has 300 points
    if num_points == 400:

        if prev_filename != name:
            i = 0
            prev_filename = name
        else:
            i += 1
        # Load the npz file
        data = np.load(npz_file)

        # Calculate the maximum extents of the data
        rna_coords = data['rna_coord']  # Extract the 'rna_coord' data
        cell_coords = data['cell_coord']

        # Extract the z and y coordinates
        z_coords = rna_coords[:, 2]
        y_coords = rna_coords[:, 1]
        z_coords_cell = cell_coords[:, 1]
        y_coords_cell = cell_coords[:, 0]

        # Calculate the scaling factors for z and y coordinates
        scale_z = image_size[0] / (np.max(z_coords_cell) - np.min(z_coords_cell))
        scale_y = image_size[1] / (np.max(y_coords_cell) - np.min(y_coords_cell))

        # Create an image for the current .npz file
        image = np.zeros(image_size)

        for z, y in zip(z_coords, y_coords):
            # Scale the z and y coordinates
            scaled_z = int(scale_z * (z - np.min(z_coords_cell)))
            scaled_y = int(scale_y * (y - np.min(y_coords_cell)))

            if 0 <= scaled_z < image_size[0] and 0 <= scaled_y < image_size[1]:
                image[scaled_y, scaled_z] = 255  # Set pixel value to indicate a point

        # Convert the numpy array to a PIL image
        image = np.stack((image,) * 3, axis=-1)
        #print(image.shape)
        pil_image = Image.fromarray(image.astype(np.uint8))

        # Save the image as .ppm in the output directory
        ppm_filename = os.path.join(output_directory, f"{name}_{i}_400.ppm")
        pil_image.save(ppm_filename, format='PPM')

        print(f"Saved {ppm_filename}")

