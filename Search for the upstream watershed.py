import geopandas as gpd
from shapely.geometry import Point
from tqdm import tqdm
import os

points_shp_path = 'RL_Basins_China_del_Points/GLAKES_Points_by_area2del.shp'
basins_shp_path = 'Hydro_BASINS/hybas_lake_as+si_lev12_v1c/hybas_lake_as+si_lev01-12_v1c.shp'
output_shp = 'Water_Basins/GLAKES_BASINS_shp/'

points_gdf = gpd.read_file(points_shp_path)
basins_gdf = gpd.read_file(basins_shp_path)

basins_sindex = basins_gdf.sindex

def process_point(row):
    point_geometry = Point(row['Lon'], row['Lat'])
    
    possible_matches_index = list(basins_sindex.query(point_geometry))
    possible_matches = basins_gdf.iloc[possible_matches_index]
    containing_basin = possible_matches[possible_matches.geometry.contains(point_geometry)]
     

    if not containing_basin.empty:
        basin_id = containing_basin.iloc[0]['HYBAS_ID']

        searched = {basin_id}  
        to_search = {basin_id}

        while to_search:
            current_search = set()
            for next_down_id in to_search:
                upstream_rows = basins_gdf[basins_gdf['NEXT_DOWN'] == next_down_id]
                current_search.update(upstream_rows['HYBAS_ID'].unique())
            to_search = current_search - searched
            searched.update(current_search)

        geometries = basins_gdf[basins_gdf['HYBAS_ID'].isin(searched)]['geometry']

        if not geometries.is_empty.all():
            merged_geom = geometries.unary_union
            merged_gdf = gpd.GeoDataFrame([{'geometry': merged_geom}], crs=basins_gdf.crs)
            output_path = f'{output_shp}merged_basin_{row["FID_GLAKES"]}.shp'
            merged_gdf.to_file(output_path)
            #print(f'Merged shapefile for point FID_GLAKES {row["FID_GLAKES"]} saved to {output_path}')
        #else:
            #print(f'No geometries found to merge for point FID_GLAKES {point["FID_GLAKES"]}')
    #else:
        #print(f'No containing basin found for point FID_GLAKES {point["FID_GLAKES"]}')

existing_files = set(os.path.basename(f) for f in os.listdir(output_shp) if f.endswith('.shp'))

for index, row in tqdm(points_gdf.iterrows(), total=points_gdf.shape[0]):
    output_filename = f'merged_basin_{row["FID_GLAKES"]}.shp'
    if output_filename not in existing_files:
        process_point(row)
    else:
        print(f"Point {row['FID_GLAKES']} already processed, skipping.")

print("All over.")