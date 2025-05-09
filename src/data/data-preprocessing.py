import pandas as pd
import numpy as np
import os

input_file_path = 'data/raw/Western-Palearctic-greater-white-fronted-geese.csv'
output_dir = 'data/processed'
output_file_path = os.path.join(output_dir, 'data_preprocessed_geese.csv')

if not os.path.exists(input_file_path):
     print(f"Lỗi: Không tìm thấy tệp dữ liệu đầu vào tại '{input_file_path}'")
     exit()

df = pd.read_csv(input_file_path)

columns_to_drop = ['event-id',
 'visible',
 'sensor-type',
 'individual-taxon-canonical-name',
 'tag-local-identifier',
 'individual-local-identifier',
 'study-name']

df = df.drop(columns=columns_to_drop)


numeric_cols = ['ground-speed', 'heading', 'height-above-msl']
for col in numeric_cols:
     if df[col].isnull().any():
           df[col] = df[col].interpolate(method='linear')


df = df[(df['height-above-msl'] >= -1000) & (df['height-above-msl'] <= 12000)]


df['heading'] = df['heading'].apply(lambda x: x if 0 <= x <= 360 else np.nan)


df = df.dropna(subset=['heading', 'height-above-msl', 'location-long', 'location-lat', 'ground-speed', 'timestamp'])


df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%d %H:%M:%S.%f')


df['year'] = df['timestamp'].dt.year
df['month'] = df['timestamp'].dt.month
df['hour'] = df['timestamp'].dt.hour
df['day_of_year'] = df['timestamp'].dt.dayofyear


df = df.rename(columns={'location-long': 'longitude', 'location-lat': 'latitude'})

df = df[['timestamp', 'longitude', 'latitude', 'ground-speed', 'heading', 'height-above-msl', 'year', 'month', 'hour', 'day_of_year']]

os.makedirs(output_dir, exist_ok=True)
df.to_csv(output_file_path, index=False)

print(f"Dữ liệu đã được tiền xử lý và lưu tại: {output_file_path}")
print(df.info())
print(df.head())