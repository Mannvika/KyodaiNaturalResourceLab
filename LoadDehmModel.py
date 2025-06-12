import numpy as np
import matplotlib.pyplot as plt
import os

# --- 1. Define File Parameters ---
# The name of your data file
file_path = 'Kyoto-Osaka_gsi.dehm_be' 

# Image dimensions (as per your information)
# EW: East-West, corresponds to number of columns (width)
# NS: North-South, corresponds to number of rows (height)
ew_pixels = 11702
ns_pixels = 12602

# Data type: 4-byte float, big-endian
# In numpy, '>' denotes big-endian, and 'f4' denotes a 4-byte float.
data_type = np.dtype('>f4')

# --- 2. Check if the File Exists ---
if not os.path.exists(file_path):
    print(f"Error: The file was not found at '{file_path}'")
    print("Please make sure the script is in the same directory as the data file.")
else:
    print(f"Loading data from '{file_path}'...")
    
    try:
        # --- 3. Read the Binary File ---
        # Use np.fromfile to read the raw binary data into a 1D numpy array.
        raw_data = np.fromfile(file_path, dtype=data_type)
        
        print(f"Successfully read {raw_data.size} values from the file.")

        # --- 4. Reshape the Array into a 2D Grid ---
        # The expected number of pixels is ns_pixels * ew_pixels.
        expected_size = ns_pixels * ew_pixels
        if raw_data.size != expected_size:
            print(f"Warning: The file size ({raw_data.size} values) does not match the expected dimensions ({expected_size} values).")
            # The script will attempt to reshape anyway, but it may fail or produce a distorted image.

        # Reshape the 1D array into a 2D grid.
        # The standard convention for images/arrays is (rows, columns), which is (NS, EW).
        topography_data = raw_data.reshape((ns_pixels, ew_pixels))
        
        print(f"Data successfully reshaped into a ({ns_pixels}, {ew_pixels}) grid.")

        # --- 5. Visualize the Data ---
        print("Generating plot...")
        
        plt.figure(figsize=(10, 10))
        # Use imshow to display the data as an image.
        # 'cmap="terrain"' is a good colormap for elevation data.
        # 'origin="upper"' places the (0,0) index at the top-left corner.
        im = plt.imshow(topography_data, cmap='terrain', origin='upper')
        
        plt.title("Topography Map of Kyoto-Osaka Region")
        plt.xlabel("East-West Pixels")
        plt.ylabel("North-South Pixels")
        
        # Add a color bar to show the elevation scale (in meters).
        cbar = plt.colorbar(im)
        cbar.set_label("Elevation (meters)")
        
        # Display the plot.
        plt.show()

    except Exception as e:
        print(f"An error occurred during processing: {e}")