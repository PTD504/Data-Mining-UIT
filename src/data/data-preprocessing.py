import pandas as pd
import numpy as np

df = pd.read_csv('path_to_your_file/geese.csv')

columns_to_drop = ['event-id',
 'visible',
#  'timestamp',
#  'location-long',
#  'location-lat',
#  'ground-speed',
#  'heading',
#  'height-above-msl',
 'sensor-type',
 'individual-taxon-canonical-name',
 'tag-local-identifier',
 'individual-local-identifier',
 'study-name']

df.drop(columns=columns_to_drop, inplace=True)

# Xử lý missing values
df['ground-speed'].fillna(df['ground-speed'].mean(), inplace=True)
df['heading'].fillna(df['heading'].mean(), inplace=True)
df['height-above-msl'].fillna(df['height-above-msl'].mean(), inplace=True)

# Xử lý dữ liệu không hợp lệ trong height-above-msl (loại bỏ giá trị ngoại lệ)
df = df[(df['height-above-msl'] >= 0) & (df['height-above-msl'] <= 5000)]

# Xử lý dữ liệu không hợp lệ trong heading (giới hạn trong phạm vi 0-360)
df['heading'] = df['heading'].apply(lambda x: x if 0 <= x <= 360 else np.nan)

# Loại bỏ các giá trị thiếu trong heading và height-above-msl sau khi xử lý
df.dropna(subset=['heading', 'height-above-msl'], inplace=True)

# Chuyển đổi timestamp thành datetime
df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%d %H:%M:%S.%f')

# Tách ra các cột
df['year'] = df['timestamp'].dt.year
df['month'] = df['timestamp'].dt.month
# df['day'] = df['timestamp'].dt.day
df['hour'] = df['timestamp'].dt.hour
# df['minute'] = df['timestamp'].dt.minute
# df['second'] = df['timestamp'].dt.second
df['day_of_year'] = df['timestamp'].dt.dayofyear
# df['weekday'] = df['timestamp'].dt.weekday

df = df.drop(columns=['timestamp'])

df.to_csv('data/data_preprocessed_geese.csv', index=False)
