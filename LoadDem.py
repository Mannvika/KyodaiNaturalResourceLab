import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter


    
if __name__ == "__main__":
    dem_file = 'Kyoto-Osaka.dehm'
    dem_data = LoadDem(dem_file)
    
    if dem_data is not None:
        plt.imshow(dem_data, cmap='terrain')
        plt.colorbar(label='Elevation (m)')
        plt.title('Digital Elevation Model')
        plt.show()
    else:
        print("Failed to load DEM data.")