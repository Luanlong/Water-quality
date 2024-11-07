

import os
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import geometry_mask
from rasterio.mask import mask


tif_file_path = r"pop_2022.tif"
output_folder_path = r"pop\pop_2022.xlsx"

output_dir = os.path.dirname(output_folder_path)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

point_shapefile_path = r"points.shp"
shapefile_directory = r"data_monthly\shp"

# IDW
def idw_influence(x, y, z, target_point, power=2):
    distances = np.sqrt((x - target_point[0])**2 + (y - target_point[1])**2)
    distances = np.where(distances == 0, 1e-10, distances)
    weights = 1 / distances**power

    valid_indices = np.where((z != src.nodata) & (z > 0))
    weights = weights[valid_indices]
    z = z[valid_indices]

    return np.sum(weights * z) / np.sum(weights)

with rasterio.open(tif_file_path) as src:
    tif_data = src.read(1)  
    tif_transform = src.transform

points_gdf = gpd.read_file(point_shapefile_path)

all_results_df = pd.DataFrame(columns=['Index', 'IDW_Value'])
results_list = []

for filename in os.listdir(shapefile_directory):
    if filename.endswith('.shp'):

        polygon_gdf = gpd.read_file(os.path.join(shapefile_directory, filename))
        shapefile_index = int(os.path.splitext(filename)[0])

        for index, polygon in polygon_gdf.iterrows():
            with rasterio.open(tif_file_path) as src:

                clipped_tif_data, clipped_transform = mask(src, [polygon['geometry']], crop=True)

                if clipped_tif_data.ndim != 3 or clipped_tif_data.shape[0] != 1:
                    print("Clipped data is not 2D or does not have a single layer for polygon:", index)
                    continue

                clipped_tif_data = clipped_tif_data[0] 

                matched_points = points_gdf[points_gdf['Index'] == shapefile_index]
                target_point = (matched_points.iloc[0].geometry.x, matched_points.iloc[0].geometry.y)

                x, y = np.meshgrid(np.arange(clipped_tif_data.shape[1]), np.arange(clipped_tif_data.shape[0]))
                x, y = rasterio.transform.xy(clipped_transform, y, x, offset='center')

                idw_value = idw_influence(np.array(x).flatten(), np.array(y).flatten(), clipped_tif_data.flatten(), target_point)
                results_list.append({'Index': shapefile_index, 'IDW_Value': idw_value})
                print(shapefile_index, idw_value)

                del clipped_tif_data, clipped_transform, matched_points, target_point, x, y, idw_value

all_results_df = pd.DataFrame(results_list)
all_results_df.to_excel(output_folder_path, index=False)