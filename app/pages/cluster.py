# pages/1_Main_App.py (hoặc tên file chứa main_app_page của bạn)
import streamlit as st
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union # << QUAN TRỌNG: Đảm bảo bạn đã import dòng này
from scipy.spatial import Voronoi
from sklearn.cluster import DBSCAN
import folium
from streamlit_folium import folium_static
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# --- MOCK FUNCTION cho get_weather_data (giữ nguyên) ---
def get_weather_data(lat, lon, api_key):
    return {
        "temperature_celsius": np.random.uniform(10, 35),
        "humidity": np.random.uniform(30, 90),
        "weather_description": np.random.choice(["Sunny", "Cloudy", "Rainy", "Windy", "Foggy"])
    }
api_key = "d62e6942105fef7a514b277c5bbbc956"

# --- Các hàm helper (giữ nguyên) ---
def get_bounding_area(gdf, buffer_deg=0.01):
    bounds = gdf.total_bounds
    minx, miny, maxx, maxy = bounds
    return Polygon([
        (minx - buffer_deg, miny - buffer_deg), (minx - buffer_deg, maxy + buffer_deg),
        (maxx + buffer_deg, maxy + buffer_deg), (maxx + buffer_deg, miny - buffer_deg)
    ])

def clip_voronoi_cells(voronoi_gdf, radius_deg=0.05):
    clipped_polygons = []
    for poly in voronoi_gdf.geometry:
        if poly is None or poly.is_empty: continue
        centroid = poly.centroid
        circle = centroid.buffer(radius_deg)
        clipped = poly.intersection(circle)
        if clipped.is_valid and not clipped.is_empty:
            clipped_polygons.append(clipped)
    return gpd.GeoDataFrame(geometry=clipped_polygons, crs="EPSG:4326")

@st.cache_data
def create_voronoi_grid(coords_tuple, _boundary_geom=None):
    coords = np.array(coords_tuple)
    if len(coords) < 4:
        # st.warning("Not enough points to create a Voronoi diagram.") # Đã xử lý ở dưới
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

# --- Hàm chính của trang ---
def main_app_page():
    # Đặt st.set_page_config ở đầu file hoặc trong app.py chính
    # st.set_page_config(page_title="Clustering Analysis", layout="wide") # Nếu file này là một trang riêng
    st.title("Clustering Map of Migration Trajectories")

    if 'gdf_data' not in st.session_state or st.session_state.gdf_data.empty:
        st.warning("Please upload data through the main app page (app.py) first.")
        return # Bỏ st.info vì nó không cần thiết nếu đã có warning

    gdf_input = st.session_state.gdf_data.copy() # Đổi tên để rõ ràng gdf là input ban đầu
    gdf_input = gdf_input.dropna(subset=['geometry'])
    coords_list = list(zip(gdf_input.geometry.x, gdf_input.geometry.y))
    coords_tuple = tuple(map(tuple, coords_list))

    if len(coords_list) < 4:
        st.warning("Not enough valid coordinates (need at least 4) to create a Voronoi diagram.")
        return

    boundary_geom = get_bounding_area(gdf_input, buffer_deg=0.01)
    voronoi_grid = create_voronoi_grid(coords_tuple, _boundary_geom=boundary_geom)

    if voronoi_grid.empty: # Sửa lại kiểm tra
        st.warning("Voronoi grid could not be generated or is empty.")
        return

    voronoi_grid = clip_voronoi_cells(voronoi_grid, radius_deg=0.05)
    voronoi_grid = voronoi_grid[voronoi_grid.geometry.notnull() & voronoi_grid.geometry.is_valid & ~voronoi_grid.geometry.is_empty].reset_index(drop=True)
    if voronoi_grid.empty:
        st.warning("Voronoi grid is empty after clipping.")
        return

    if gdf_input.crs != voronoi_grid.crs:
        voronoi_grid = voronoi_grid.to_crs(gdf_input.crs)
    
    try:
        joined = gpd.sjoin(gdf_input, voronoi_grid, how='left', predicate='intersects')
    except Exception as e: # Bắt lỗi cụ thể hơn nếu có thể
        st.error(f"Error during spatial join: {e}")
        return
        
    density = joined.groupby('index_right').size()
    voronoi_grid['trajectory_count'] = voronoi_grid.index.map(density).fillna(0).astype(int)

    if voronoi_grid['trajectory_count'].empty or voronoi_grid['trajectory_count'].max() == 0:
        st.warning("No trajectories found in any Voronoi cell or max count is 0.")
        # ... (code hiển thị bản đồ Voronoi cơ bản nếu không có density giữ nguyên) ...
        return

    # --- Sidebar cho các tham số ---
    # st.sidebar.header("⚙️ Control Panel") # Có thể đặt ở app.py nếu dùng chung
    max_count = int(voronoi_grid['trajectory_count'].max())
    default_threshold = min(10, max_count) if max_count > 0 else 1
    slider_min_thresh = 1 if max_count > 0 else 0
    slider_max_thresh = max_count if max_count > 0 else 1
    
    threshold = default_threshold # Giá trị mặc định
    if slider_min_thresh < slider_max_thresh:
        threshold = st.slider("Preservable Area Threshold", slider_min_thresh, slider_max_thresh, default_threshold, key="main_threshold_v4")
    elif slider_max_thresh > 0 :
        st.write(f"Preservable Area Threshold (fixed): {slider_max_thresh}")
        threshold = slider_max_thresh
    else:
        st.warning("No trajectory counts available to set a threshold.")
        return # Không thể tiếp tục nếu không có threshold hợp lệ


    high_density_polygons = voronoi_grid[voronoi_grid['trajectory_count'] >= threshold].copy()
    if high_density_polygons.empty:
        st.warning("No high-density polygons found with the current threshold.")
        # ... (code hiển thị bản đồ Voronoi với trajectory_count nếu không có high-density giữ nguyên) ...
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

    if len(high_density_polygons) < 2: # DBSCAN cần ít nhất `min_samples`, mà min_samples ít nhất là 1. Để có cluster ý nghĩa, cần >1 điểm.
        st.warning(f"Need at least 2 high-density polygons with valid weather data for clustering, found {len(high_density_polygons)}.")
        return

    # --- Chuẩn bị features (BAO GỒM CẢ LAT, LON, TEMP, HUMID) ---
    features_list = []
    valid_rows_indices = [] # Để tạo valid_polygons_gdf chính xác
    for index, row in high_density_polygons.iterrows():
        if row["weather"] and row.geometry is not None and not row.geometry.is_empty:
            centroid = row.geometry.centroid
            features_list.append([
                centroid.y,  # Latitude
                centroid.x,  # Longitude
                row["weather"]["temperature_celsius"],
                row["weather"]["humidity"]
            ])
            valid_rows_indices.append(index)
    
    if not features_list:
        st.warning("No polygons with valid weather data remaining for clustering features.")
        return
    
    valid_polygons_gdf = high_density_polygons.loc[valid_rows_indices].copy()
    features_np = np.array(features_list)

    if features_np.shape[0] != len(valid_polygons_gdf):
        st.error("Mismatch between number of features and valid polygons. Check filtering logic.")
        return

    # --- Feature Weights (THEO YÊU CẦU CỦA BẠN) ---
    st.subheader("Feature Weights (Adjustable for Weather)") # Đặt subheader ở main page thay vì sidebar
    col_w1, col_w2 = st.columns(2)
    with col_w1:
        temp_weight = st.slider("Temperature Weight", 0.1, 2.0, 0.5, key="temp_w_main_v4")
    with col_w2:
        humid_weight = st.slider("Humidity Weight", 0.1, 2.0, 0.5, key="humid_w_main_v4")
    
    lat_weight = 1.0 # Trọng số cố định
    lon_weight = 1.0 # Trọng số cố định
    
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_np)
    weighted_features = features_scaled * np.array([lat_weight, lon_weight, temp_weight, humid_weight])

    # --- Clustering Parameters ---
    st.subheader("Clustering Parameters (DBSCAN)") # Đặt subheader ở main page
    eps_max_val = 5.0
    min_samples_max_val = max(1, len(weighted_features))
    min_samples_default_val = min(2, min_samples_max_val) if min_samples_max_val > 0 else 1

    col_c1, col_c2 = st.columns(2)
    with col_c1:
        eps = st.slider("EPS (scaled units)", 0.01, eps_max_val, 1.0, step=0.01, key="eps_s_main_v4")
    with col_c2:
        min_samples = st.slider("Minimum samples", 1, min_samples_max_val, min_samples_default_val, key="min_s_main_v4")

    if weighted_features.shape[0] == 0: st.warning("No features to cluster."); return
    if weighted_features.shape[0] < min_samples:
        st.warning(f"Minimum samples ({min_samples}) is greater than the number of available data points ({weighted_features.shape[0]}).")
        return

    try:
        db = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean').fit(weighted_features)
        valid_polygons_gdf['cluster'] = db.labels_
    except ValueError as e:
        st.error(f"Error during DBSCAN fitting: {e}"); return

    # --- TÙY CHỌN HIỂN THỊ BẢN ĐỒ (TỪ SIDEBAR HOẶC MAIN PAGE) ---
    # st.sidebar.markdown("---") # Nếu các tùy chọn này ở sidebar
    # st.sidebar.subheader("🗺️ Map Display Options")
    # show_cluster_hulls = st.sidebar.checkbox("Hiển thị vùng bao Cluster", value=True, key="show_hulls_v4")
    # show_voronoi_cells = st.sidebar.checkbox("Hiển thị ô Voronoi chi tiết", value=False, key="show_cells_v4")
    # show_circle_markers = st.sidebar.checkbox("Hiển thị điểm tâm", value=True, key="show_markers_v4")
    
    # Hoặc đặt ở main page cho dễ thấy
    st.subheader("Map Display Options")
    display_col1, display_col2, display_col3 = st.columns(3)
    with display_col1:
        show_cluster_hulls = st.checkbox("Show Cluster Hulls", value=True, key="show_hulls_v4_main")
    with display_col2:
        show_voronoi_cells = st.checkbox("Show Voronoi Cells", value=False, key="show_cells_v4_main")
    with display_col3:
        show_circle_markers = st.checkbox("Show Center Markers", value=True, key="show_markers_v4_main")


    # Lọc Cluster theo ID
    unique_cluster_ids = sorted([c_id for c_id in valid_polygons_gdf['cluster'].unique() if c_id != -1])
    if unique_cluster_ids:
        selected_cluster_ids = st.multiselect( # Đặt ở main page
            "Filter and Display Clusters IDs (Noise (-1) always shown if present):",
            options=unique_cluster_ids, default=unique_cluster_ids, key="select_cluster_ids_v4_main"
        )
    else:
        selected_cluster_ids = []; st.info("No actual clusters formed (excluding noise).")

    # --- Tạo bản đồ ---
    map_center_lat = features_np[:,0].mean() if features_np.shape[0] > 0 else gdf_input.geometry.y.mean()
    map_center_lon = features_np[:,1].mean() if features_np.shape[0] > 0 else gdf_input.geometry.x.mean()
    m = folium.Map(location=[map_center_lat, map_center_lon], zoom_start=4, tiles="CartoDB positron")

    # Lọc các đa giác sẽ hiển thị dựa trên lựa chọn cluster ID (luôn bao gồm nhiễu -1 nếu có)
    polygons_to_display_on_map = valid_polygons_gdf[valid_polygons_gdf['cluster'].isin(selected_cluster_ids + [-1] if -1 in valid_polygons_gdf['cluster'].values else selected_cluster_ids)]

    # >>> THÊM PHẦN VẼ CONVEX HULL VÀO ĐÂY <<<
    if show_cluster_hulls and selected_cluster_ids and not polygons_to_display_on_map.empty: # Chỉ vẽ hull nếu được chọn
        hulls_data = []
        # Tính hull cho các cluster được chọn và có trong polygons_to_display_on_map
        for c_id in selected_cluster_ids: # Chỉ lặp qua các cluster ID được chọn (không phải -1)
            cluster_polygons_for_hull = polygons_to_display_on_map[polygons_to_display_on_map['cluster'] == c_id] # Lấy các polygon của cluster hiện tại
            
            if not cluster_polygons_for_hull.empty and len(cluster_polygons_for_hull.geometry) > 0:
                # Gộp tất cả các geometries trong cụm này thành một MultiPolygon hoặc Polygon duy nhất
                united_geometry = unary_union(cluster_polygons_for_hull.geometry.tolist())
                
                if united_geometry.is_empty: continue

                hull = united_geometry.convex_hull # Tính convex hull
                
                # Thu thập thông tin cho popup của hull
                avg_temp = cluster_polygons_for_hull['weather'].apply(lambda x: x['temperature_celsius'] if isinstance(x, dict) else np.nan).mean()
                avg_humidity = cluster_polygons_for_hull['weather'].apply(lambda x: x['humidity'] if isinstance(x, dict) else np.nan).mean()
                weather_descs_series = cluster_polygons_for_hull['weather'].apply(lambda x: x['weather_description'] if isinstance(x, dict) else None).dropna()
                weather_descs = list(weather_descs_series.value_counts().nlargest(3).index) # Lấy 3 mô tả phổ biến nhất
                
                popup_html = f"""
                <b>Cluster ID: {c_id}</b><br>
                Số ô Voronoi: {len(cluster_polygons_for_hull)}<br>
                Tổng số quỹ đạo (ước tính): {cluster_polygons_for_hull['trajectory_count'].sum()}<br>
                Nhiệt độ TB: {avg_temp:.1f}°C<br>
                Độ ẩm TB: {avg_humidity:.1f}%<br>
                Thời tiết phổ biến: {', '.join(weather_descs)}
                """
                hulls_data.append({'cluster': c_id, 'geometry': hull, 'popup_html': popup_html, 'num_cells': len(cluster_polygons_for_hull)})
        
        if hulls_data:
            cluster_hulls_gdf = gpd.GeoDataFrame(hulls_data, crs=valid_polygons_gdf.crs)
            folium.GeoJson(
                cluster_hulls_gdf,
                style_function=lambda feature: {
                    'fillColor': f'hsl({abs(feature["properties"]["cluster"])*60 % 360}, 80%, 60%)',
                    'color': f'hsl({abs(feature["properties"]["cluster"])*60 % 360}, 100%, 30%)',
                    'weight': 2.5,
                    'fillOpacity': 0.35, # Độ mờ để thấy bên dưới
                },
                tooltip=folium.features.GeoJsonTooltip(fields=['cluster', 'num_cells'], aliases=['Cluster ID:', 'Số ô Voronoi:']),
                popup=folium.features.GeoJsonPopup(fields=['popup_html'], labels=False, parse_html=True, max_width=300),
                name="Cluster Hulls" # Đặt tên cho lớp này
            ).add_to(m)

    # Vẽ các ô Voronoi chi tiết (NẾU ĐƯỢC CHỌN)
    if show_voronoi_cells and not polygons_to_display_on_map.empty:
        folium.GeoJson(
            polygons_to_display_on_map.to_json(), # Chỉ vẽ các ô đã được lọc
            style_function=lambda feature: {
                'fillColor': f'hsl({abs(feature["properties"]["cluster"])*60 % 360}, 70%, 50%)' if feature["properties"]["cluster"] != -1 else '#AAAAAA',
                'color': 'black',
                'weight': 0.5, # Viền mỏng hơn cho ô con
                'fillOpacity': 0.6 if feature["properties"]["cluster"] != -1 else 0.3, # Độ mờ khác nhau cho nhiễu
            },
            tooltip=folium.GeoJsonTooltip(fields=['cluster', 'trajectory_count'], aliases=['Cluster ID:', 'Trajectory Count:']),
            name="Detailed Voronoi Cells" # Đặt tên cho lớp này
        ).add_to(m)

    # Vẽ CircleMarkers cho tâm các ô (NẾU ĐƯỢC CHỌN)
    if show_circle_markers and not polygons_to_display_on_map.empty:
        for _, row in polygons_to_display_on_map.iterrows(): # Chỉ vẽ marker cho các ô đã được lọc
            if row.geometry is not None and not row.geometry.is_empty:
                centroid = row.geometry.centroid
                weather = row["weather"] # Đã dropna nên weather sẽ tồn tại
                # if weather: # Không cần kiểm tra lại vì đã dropna
                popup_content = f"""
                <b>Cluster: {row['cluster']}</b> ({'Noise' if row['cluster'] == -1 else 'Cluster Point'})<br>
                Voronoi Cell Traj. Count: {row['trajectory_count']}<br>
                Temp: {weather.get('temperature_celsius', 'N/A'):.1f}°C<br>
                Humidity: {weather.get('humidity', 'N/A')}%<br>
                Conditions: {weather.get('weather_description', 'N/A')}
                """
                folium.CircleMarker(
                    location=[centroid.y, centroid.x],
                    radius=4, # Bán kính nhỏ hơn
                    color=f'hsl({abs(row["cluster"])*60 % 360}, 100%, 25%)' if row["cluster"] != -1 else '#666666',
                    fill_color=f'hsl({abs(row["cluster"])*60 % 360}, 70%, 50%)' if row["cluster"] != -1 else '#AAAAAA',
                    fill_opacity=0.9, # Nổi bật hơn
                    popup=folium.Popup(popup_content, max_width=300),
                    tooltip=f"Cluster: {row['cluster']}, Count: {row['trajectory_count']}",
                    name="Center Markers" # Đặt tên cho lớp này (nếu muốn kiểm soát riêng từng marker, nhưng thường không cần)
                ).add_to(m)
    
    # Chỉ thêm LayerControl nếu có ít nhất một lớp có thể bật/tắt
    if show_cluster_hulls or show_voronoi_cells or show_circle_markers:
        folium.LayerControl(collapsed=False).add_to(m) # collapsed=False để mở sẵn

    st.subheader("Clustering Map Results")
    folium_static(m, height=650) # Bỏ width="100%"

    # --- Tính Silhouette Score (giữ nguyên) ---
    unique_clusters_obj = np.unique(db.labels_)
    num_actual_clusters = len(unique_clusters_obj) - (1 if -1 in unique_clusters_obj else 0)

    if num_actual_clusters >= 2 and weighted_features.shape[0] > num_actual_clusters :
        try:
            labels_for_score = db.labels_[db.labels_ != -1]
            # Đảm bảo features_for_score có cùng số hàng với labels_for_score
            if len(labels_for_score) > 0: # Chỉ tính nếu có điểm không nhiễu
                features_for_score = weighted_features[db.labels_ != -1]
                if len(np.unique(labels_for_score)) >= 2 and len(labels_for_score) > len(np.unique(labels_for_score)):
                    silhouette_avg = silhouette_score(features_for_score, labels_for_score)
                    st.metric("Silhouette Score (Non-noise points)", f"{silhouette_avg:.3f}")
                else: st.info("Không đủ cụm (>1) hoặc điểm trong các cụm để tính Silhouette Score sau khi loại bỏ nhiễu.")
            else:
                st.info("Không có điểm nào thuộc các cụm (chỉ có nhiễu) để tính Silhouette Score.")
        except ValueError as e: st.warning(f"Could not calculate Silhouette Score: {e}")
    elif weighted_features.shape[0] <=1: st.warning("Not enough data points to calculate Silhouette Score.")
    else: st.info("Need at least 2 clusters (excluding noise) to calculate Silhouette Score.")

if __name__ == "__main__":
    main_app_page()