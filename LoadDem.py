import matplotlib.pyplot as plt
import numpy as np

def LoadDem(dem_path):
    data_type = np.dtype('>f4')
    raw_data = np.fromfile(dem_path, dtype=data_type)
    dem_data = raw_data.reshape((12602, 11702))
    return dem_data
    
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