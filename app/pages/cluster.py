import streamlit as st
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Polygon
from scipy.spatial import Voronoi
from sklearn.cluster import DBSCAN
import folium
from streamlit_folium import folium_static
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
# from sklearn.metrics.pairwise import haversine_distances # Không thấy sử dụng trong main()
# from utils import get_weather_data # Giả sử bạn có file utils.py với hàm này

# ---- MOCK FUNCTION cho get_weather_data ----
# Thay thế bằng hàm thực của bạn trong utils.py
def get_weather_data(lat, lon, api_key):
    # Đây là dữ liệu giả lập
    return {
        "temperature_celsius": np.random.uniform(10, 35),
        "humidity": np.random.uniform(30, 90),
        "weather_description": np.random.choice(["Sunny", "Cloudy", "Rainy"])
    }
# ---- KẾT THÚC MOCK FUNCTION ----

api_key = "d62e6942105fef7a514b277c5bbbc956" # API key của bạn

# Các hàm helper của bạn (get_bounding_area, clip_voronoi_cells, create_voronoi_grid)
# Nên giữ chúng ở đây hoặc chuyển vào utils.py và import nếu muốn tái sử dụng ở nơi khác.
# Để đơn giản, tôi sẽ giữ chúng ở đây.

def get_bounding_area(gdf, buffer_deg=0.01):
    bounds = gdf.total_bounds
    minx, miny, maxx, maxy = bounds
    return Polygon([
        (minx - buffer_deg, miny - buffer_deg),
        (minx - buffer_deg, maxy + buffer_deg),
        (maxx + buffer_deg, maxy + buffer_deg),
        (maxx + buffer_deg, miny - buffer_deg)
    ])

def clip_voronoi_cells(voronoi_gdf, radius_deg=0.05):
    clipped_polygons = []
    for poly in voronoi_gdf.geometry:
        if poly is None or poly.is_empty:
            continue
        centroid = poly.centroid
        circle = centroid.buffer(radius_deg)
        clipped = poly.intersection(circle)
        if clipped.is_valid and not clipped.is_empty:
            clipped_polygons.append(clipped)
    return gpd.GeoDataFrame(geometry=clipped_polygons, crs="EPSG:4326")

@st.cache_data # Sử dụng cache cho các hàm tính toán nặng
def create_voronoi_grid(coords_tuple, _boundary_geom=None): # coords phải là tuple để hashable
    coords = np.array(coords_tuple)
    if len(coords) < 4:
        st.warning("Not enough points to create a Voronoi diagram.")
        return gpd.GeoDataFrame(geometry=[], crs='EPSG:4326')

    vor = Voronoi(coords)
    polygons = []
    for region in vor.regions:
        if len(region) > 0 and -1 not in region:
            polygon_vertices = [vor.vertices[i] for i in region if i < len(vor.vertices)]
            if len(polygon_vertices) >= 3:
                polygon = Polygon(polygon_vertices)
                if _boundary_geom and polygon.is_valid:
                    polygon = polygon.intersection(_boundary_geom)
                if polygon.is_valid and not polygon.is_empty:
                    polygons.append(polygon)
    return gpd.GeoDataFrame(geometry=polygons, crs='EPSG:4326')

st.set_page_config(page_title="Clustering Analysis", layout="wide") # Đặt tiêu đề cho tab

def main_app_page():
    st.title("Clustering Map of Migration Trajectories") # Tiêu đề trong trang

    if 'gdf_data' not in st.session_state:
        st.warning("Please upload data through the main app page (app.py) first.")
        st.info("Use the sidebar in the main 'app' page to upload your CSV.")
        return
    gdf = st.session_state.gdf_data.copy()
    gdf = gdf.dropna(subset=['geometry'])
    # Chuyển đổi coords sang tuple để st.cache_data hoạt động với create_voronoi_grid
    coords_list = list(zip(gdf.geometry.x, gdf.geometry.y))
    coords_tuple = tuple(map(tuple, coords_list))


    if len(coords_list) == 0:
        st.warning("No valid coordinates found in the data.")
        return

    boundary_geom = get_bounding_area(gdf, buffer_deg=0.01)
    # Truyền coords_tuple vào hàm create_voronoi_grid
    voronoi_grid = create_voronoi_grid(coords_tuple, _boundary_geom=boundary_geom)

    if not isinstance(voronoi_grid, gpd.GeoDataFrame) or voronoi_grid.empty:
        st.warning("Voronoi grid could not be generated. Ensure enough unique points.")
        # Có thể hiển thị gdf gốc nếu voronoi không tạo được
        # m_fallback = folium.Map(location=[gdf.geometry.y.mean(), gdf.geometry.x.mean()], zoom_start=5)
        # folium.GeoJson(gdf.to_json(), name="Input Points").add_to(m_fallback)
        # folium_static(m_fallback, width=1200, height=600)
        return

    voronoi_grid = clip_voronoi_cells(voronoi_grid, radius_deg=0.05)

    gdf = gdf[gdf.geometry.notnull()]
    voronoi_grid = voronoi_grid[voronoi_grid.geometry.notnull()]
    voronoi_grid = voronoi_grid[voronoi_grid.geometry.is_valid]
    voronoi_grid = voronoi_grid[~voronoi_grid.geometry.is_empty]
    voronoi_grid.reset_index(drop=True, inplace=True)

    if voronoi_grid.empty:
        st.warning("Voronoi grid is empty after clipping.")
        return

    try:
        # Đảm bảo cả hai GeoDataFrames có cùng CRS trước khi sjoin
        if gdf.crs != voronoi_grid.crs:
            voronoi_grid = voronoi_grid.to_crs(gdf.crs)
        joined = gpd.sjoin(gdf, voronoi_grid, how='left', predicate='intersects')
    except Exception as e:
        st.error(f"Error during spatial join: {e}")
        st.write("GDF Info:", gdf.info())
        st.write("Voronoi Grid Info:", voronoi_grid.info())
        st.write("GDF CRS:", gdf.crs)
        st.write("Voronoi Grid CRS:", voronoi_grid.crs)
        return

    density = joined.groupby('index_right').size()
    voronoi_grid['trajectory_count'] = voronoi_grid.index.map(density).fillna(0).astype(int)

    if voronoi_grid['trajectory_count'].empty or voronoi_grid['trajectory_count'].max() == 0:
        st.warning("No trajectories found in any Voronoi cell or max count is 0.")
        # Hiển thị bản đồ Voronoi cơ bản nếu không có density
        m_no_density = folium.Map(location=[gdf.geometry.y.mean(), gdf.geometry.x.mean()], zoom_start=5)
        if not voronoi_grid.empty:
            folium.GeoJson(voronoi_grid.to_json(), name="Voronoi Cells").add_to(m_no_density)
        folium_static(m_no_density, width=1200, height=600)
        return

    # Đảm bảo threshold không gây lỗi nếu max count là 0 hoặc rất nhỏ
    max_count = int(voronoi_grid['trajectory_count'].max())
    default_threshold = min(10, max_count) if max_count > 0 else 1
    slider_min = 1 if max_count > 0 else 0 # Sửa lỗi nếu max_count là 0
    slider_max = max_count if max_count > 0 else 1 # Sửa lỗi nếu max_count là 0

    if slider_min >= slider_max and slider_max > 0: # Nếu min và max bằng nhau và > 0
        threshold = slider_max
        st.write(f"Preservable Area Threshold (fixed): {threshold}")
    elif slider_max == 0 : # Nếu không có cell nào có trajectory
        st.warning("No trajectory counts available to set a threshold.")
        threshold = 0 # Hoặc xử lý khác
    else:
        threshold = st.slider("Preservable Area Threshold", slider_min, slider_max, default_threshold, key="main_threshold")


    high_density_polygons = voronoi_grid[voronoi_grid['trajectory_count'] >= threshold].copy()

    if high_density_polygons.empty:
        st.warning("No high-density polygons found with the current threshold.")
        # Hiển thị bản đồ Voronoi với trajectory_count nếu không có high-density
        m_all_voronoi = folium.Map(location=[gdf.geometry.y.mean(), gdf.geometry.x.mean()], zoom_start=5)
        if not voronoi_grid.empty:
            folium.GeoJson(
                voronoi_grid.to_json(),
                tooltip=folium.GeoJsonTooltip(fields=['trajectory_count'])
            ).add_to(m_all_voronoi)
        folium_static(m_all_voronoi, width=1200, height=600)
        return

    weather_info = []
    for _, row in high_density_polygons.iterrows():
        if row.geometry is not None and not row.geometry.is_empty:
            centroid = row.geometry.centroid
            weather = get_weather_data(centroid.y, centroid.x, api_key)
            weather_info.append(weather)
        else:
            weather_info.append(None)

    high_density_polygons["weather"] = weather_info
    high_density_polygons = high_density_polygons.dropna(subset=['weather'])

    if len(high_density_polygons) < 3: # DBSCAN cần ít nhất `min_samples` điểm, thường là >=2 hoặc 3.
        st.warning(f"Need at least 3 high-density polygons with valid weather data for clustering, found {len(high_density_polygons)}.")
        return

    valid_polygons_df_list = []
    features_list = [] # Đổi tên từ features để tránh nhầm lẫn
    for _, row in high_density_polygons.iterrows():
        if row["weather"] and row.geometry is not None and not row.geometry.is_empty:
            centroid = row.geometry.centroid
            lat = centroid.y
            lon = centroid.x
            temp = row["weather"]["temperature_celsius"]
            humidity = row["weather"]["humidity"]

            valid_polygons_df_list.append(row.to_frame().T)
            features_list.append([lat, lon, temp, humidity])

    if not valid_polygons_df_list:
        st.warning("No polygons with valid weather data remaining for clustering.")
        return

    valid_polygons_gdf = gpd.GeoDataFrame(pd.concat(valid_polygons_df_list, ignore_index=True), crs=high_density_polygons.crs)
    features_np = np.array(features_list) # Đổi tên từ features

    st.subheader("Feature Weights")
    col1, col2 = st.columns(2)
    with col1:
        temp_weight = st.slider("Temperature Weight", 0.1, 2.0, 0.5, key="temp_w_main")
    with col2:
        humid_weight = st.slider("Humidity Weight", 0.1, 2.0, 0.5, key="humid_w_main")
    lat_weight = 1.0
    lon_weight = 1.0
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_np)
    weighted_features = features_scaled * np.array([lat_weight, lon_weight, temp_weight, humid_weight])

    st.subheader("Clustering Parameters")
    eps_max = 5.0
    # Đảm bảo có đủ điểm cho min_samples
    max_min_samples = len(weighted_features) if len(weighted_features) > 0 else 1
    min_samples_default = min(2, max_min_samples)

    eps = st.slider("EPS (scaled units)", 0.01, eps_max, 1.0, step=0.01, key="eps_s_main")
    min_samples = st.slider("Minimum samples", 1, max(1, max_min_samples), min_samples_default, key="min_s_main")


    if weighted_features.shape[0] < min_samples:
        st.warning(f"Minimum samples ({min_samples}) is greater than the number of available data points ({weighted_features.shape[0]}). Please adjust parameters or upload more data.")
        return

    try:
        db = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean').fit(weighted_features)
        clusters = db.labels_
    except ValueError as e:
        st.error(f"Error during DBSCAN fitting: {e}")
        st.write("Weighted features shape:", weighted_features.shape)
        return


    valid_polygons_gdf['cluster'] = clusters

    map_center_lat = features_np[:,0].mean() if features_np.shape[0] > 0 else gdf.geometry.y.mean()
    map_center_lon = features_np[:,1].mean() if features_np.shape[0] > 0 else gdf.geometry.x.mean()
    m = folium.Map(location=[map_center_lat, map_center_lon], zoom_start=5)

    if not valid_polygons_gdf.empty:
        folium.GeoJson(
            valid_polygons_gdf.to_json(),
            style_function=lambda feature: {
                'fillColor': f'hsl({abs(feature["properties"]["cluster"])*60 % 360}, 70%, 50%)' if feature["properties"]["cluster"] != -1 else '#808080',
                'color': 'black',
                'weight': 1,
                'fillOpacity': 0.6,
            },
            tooltip=folium.GeoJsonTooltip(fields=['cluster', 'trajectory_count']),
        ).add_to(m)

        for _, row in valid_polygons_gdf.iterrows():
            if row.geometry is not None and not row.geometry.is_empty:
                centroid = row.geometry.centroid
                weather = row["weather"]
                if weather: # Kiểm tra weather có tồn tại không
                    popup_content = f"""
                    Cluster: {row['cluster']}<br>
                    Count: {row['trajectory_count']}<br>
                    Temp: {weather.get('temperature_celsius', 'N/A'):.1f}°C<br>
                    Humidity: {weather.get('humidity', 'N/A')}%<br>
                    Conditions: {weather.get('weather_description', 'N/A')}
                    """
                    folium.CircleMarker(
                        location=[centroid.y, centroid.x],
                        radius=5,
                        color='#333333',
                        fill_color=f'hsl({abs(row["cluster"])*60 % 360}, 70%, 50%)' if row["cluster"] != -1 else '#808080', # Thêm điều kiện cho noise points
                        fill_opacity=0.7,
                        popup=folium.Popup(popup_content, max_width=300), # Tăng max_width
                    ).add_to(m)

    st.subheader("Clustering Map of Migration Trajectories")
    folium_static(m, width=1200, height=600)

    # Tính Silhouette Score
    # Đảm bảo có ít nhất 2 cụm (không bao gồm nhiễu) và nhiều hơn 1 điểm dữ liệu để tính silhouette score
    unique_clusters = np.unique(clusters)
    num_actual_clusters = len(unique_clusters) - (1 if -1 in unique_clusters else 0)

    if num_actual_clusters >= 2 and weighted_features.shape[0] > num_actual_clusters :
        try:
            # Loại bỏ nhiễu (-1) khỏi labels nếu có để tính silhouette score
            labels_for_score = clusters[clusters != -1]
            features_for_score = weighted_features[clusters != -1]

            if len(np.unique(labels_for_score)) >= 2 and len(labels_for_score) > len(np.unique(labels_for_score)): # Cần ít nhất 2 cụm và nhiều điểm hơn số cụm
                silhouette_avg = silhouette_score(features_for_score, labels_for_score)
                st.metric("Silhouette Score", f"{silhouette_avg:.3f}")
            elif weighted_features.shape[0] > 1 and len(unique_clusters) == 1 and -1 not in unique_clusters:
                 st.info("Silhouette Score is not defined for a single cluster. All points belong to one cluster.")
            else:
                st.warning("Silhouette Score cannot be calculated. Not enough clusters or data points after removing noise.")
        except ValueError as e:
            st.warning(f"Could not calculate Silhouette Score: {e}")
    elif weighted_features.shape[0] <=1:
        st.warning("Not enough data points to calculate Silhouette Score.")
    else:
        st.warning("Need at least 2 clusters (excluding noise) to calculate Silhouette Score.")


if __name__ == "__main__":
    # Điều này sẽ không chạy trực tiếp khi Streamlit chạy app.py
    # Streamlit sẽ import và chạy nội dung của file này khi được điều hướng tới.
    # Để chạy trang này, Streamlit sẽ gọi hàm main_app_page() nếu được cấu hình trong app.py
    # (Tuy nhiên, với cấu trúc pages của Streamlit, nó sẽ tự động chạy file này khi được chọn)
    main_app_page()