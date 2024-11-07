
import pandas as pd
import os

file_path = 'code/input_data/water_quality_example.xlsx'  
output_directory = ''
df = pd.read_excel(file_path, parse_dates=['Monitoring time'], infer_datetime_format=True)

df['Date'] = df['Monitoring time'].dt.strftime('%Y/%m/%d')

# Group data by specified columns and calculate mean for each parameter
grouped_data = df.groupby(['province', 'City', 'river', 'River Basin', 'Section Name', 'Date']).agg({
    'Water temperature': 'mean',
    'pH': 'mean',
    'Dissolved oxygen': 'mean',
    'Permanganate Index': 'mean',
    'Ammonia nitrogen': 'mean',
    'Total Phosphorus': 'mean',
    'Total Nitrogen': 'mean',
    'Conductivity': 'mean',
    'Turbidity': 'mean',
    'Chlorophyll': 'mean',
    'Algae density': 'mean'
}).reset_index()

output_file_path = os.path.join(output_directory, 'daily.csv')
grouped_data.to_csv(output_file_path, header=True, index=False, encoding='utf-8')

# Add latitude and longitude
latlon_path = 'code/input_data/water_quality_example_LatLon.csv' 
latlon_df = pd.read_csv(latlon_path, encoding='GBK')

month_df = pd.read_csv(output_file_path, encoding='utf-8')

# Merge latitude and longitude data based on province, city, and section name
merged_df = pd.merge(
    month_df, 
    latlon_df[['province', 'City', 'Section Name', 'longitude', 'latitude']],
    left_on=['province', 'City', 'Section Name'],
    right_on=['province', 'City', 'Section Name'],
    how='left'
)

merged_df = merged_df.drop(columns=['Section Name'])
merged_df.to_csv(output_file_path, index=False, encoding='utf-8')