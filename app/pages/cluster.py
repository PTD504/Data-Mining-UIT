# pages/1_Main_App.py (hoặc tên file chứa main_app_page của bạn)
import streamlit as st
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union # Import thêm
from scipy.spatial import Voronoi
from sklearn.cluster import DBSCAN
import folium
from streamlit_folium import folium_static
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# --- MOCK FUNCTION cho get_weather_data (giữ nguyên hoặc thay bằng hàm thực) ---
def get_weather_data(lat, lon, api_key):
    return {
        "temperature_celsius": np.random.uniform(10, 35),
        "humidity": np.random.uniform(30, 90),
        "weather_description": np.random.choice(["Sunny", "Cloudy", "Rainy", "Windy", "Foggy"])
    }
api_key = "d62e6942105fef7a514b277c5bbbc956" # API key của bạn

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
    if len(coords) < 4: return gpd.GeoDataFrame(geometry=[], crs='EPSG:4326') # Sửa warning thành return sớm
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
    st.set_page_config(page_title="Clustering Analysis", layout="wide")
    st.title("🗺️ Clustering Map of Migration Trajectories")

    if 'gdf_data' not in st.session_state or st.session_state.gdf_data.empty:
        st.warning("⚠️ Vui lòng tải dữ liệu lên từ trang chính (app.py) trước.")
        return

    gdf_input = st.session_state.gdf_data.copy() # Đổi tên để rõ ràng hơn
    gdf_input = gdf_input.dropna(subset=['geometry'])
    coords_list = list(zip(gdf_input.geometry.x, gdf_input.geometry.y))
    coords_tuple = tuple(map(tuple, coords_list))

    if len(coords_list) < 4: # Cần ít nhất 4 điểm cho Voronoi
        st.warning("Không đủ điểm dữ liệu hợp lệ (cần ít nhất 4) để tạo biểu đồ Voronoi.")
        return

    # --- Các bước xử lý Voronoi, mật độ, thời tiết (giữ nguyên logic cốt lõi) ---
    boundary_geom = get_bounding_area(gdf_input, buffer_deg=0.01)
    voronoi_grid = create_voronoi_grid(coords_tuple, _boundary_geom=boundary_geom)
    if voronoi_grid.empty: st.warning("Lưới Voronoi không thể được tạo hoặc rỗng."); return
    voronoi_grid = clip_voronoi_cells(voronoi_grid, radius_deg=0.05)
    voronoi_grid = voronoi_grid[voronoi_grid.geometry.notnull() & voronoi_grid.geometry.is_valid & ~voronoi_grid.geometry.is_empty].reset_index(drop=True)
    if voronoi_grid.empty: st.warning("Lưới Voronoi rỗng sau khi cắt."); return

    if gdf_input.crs != voronoi_grid.crs: voronoi_grid = voronoi_grid.to_crs(gdf_input.crs)
    joined = gpd.sjoin(gdf_input, voronoi_grid, how='left', predicate='intersects') # Sửa 'op' thành 'predicate'
    density = joined.groupby('index_right').size()
    voronoi_grid['trajectory_count'] = voronoi_grid.index.map(density).fillna(0).astype(int)

    # --- Sidebar cho các tham số (giữ nguyên hoặc điều chỉnh nếu cần) ---
    st.sidebar.header("⚙️ Control Panel")
    max_traj_count = int(voronoi_grid['trajectory_count'].max()) if not voronoi_grid['trajectory_count'].empty else 1
    threshold_default = min(10, max_traj_count) if max_traj_count > 0 else 1
    threshold = st.sidebar.slider("Preservable Area Threshold (Min Trajectory Count):", 1, max(1, max_traj_count), threshold_default, key="main_threshold")

    high_density_polygons = voronoi_grid[voronoi_grid['trajectory_count'] >= threshold].copy()
    if high_density_polygons.empty: st.warning("Không có đa giác mật độ cao nào được tìm thấy với ngưỡng hiện tại."); return

    # Lấy thông tin thời tiết (giữ nguyên)
    weather_info = []
    for _, row in high_density_polygons.iterrows():
        if row.geometry is not None and not row.geometry.is_empty:
            centroid = row.geometry.centroid
            weather = get_weather_data(centroid.y, centroid.x, api_key) # api_key cần được định nghĩa
            weather_info.append(weather)
        else: weather_info.append(None)
    high_density_polygons["weather"] = weather_info
    high_density_polygons = high_density_polygons.dropna(subset=['weather'])
    if len(high_density_polygons) < 2: st.warning(f"Cần ít nhất 2 đa giác mật độ cao có dữ liệu thời tiết hợp lệ để phân cụm, tìm thấy {len(high_density_polygons)}."); return

    # Chuẩn bị features (giữ nguyên)
    valid_polygons_df_list = []
    features_list = []
    for _, row in high_density_polygons.iterrows():
        if row["weather"] and row.geometry is not None and not row.geometry.is_empty:
            centroid = row.geometry.centroid
            valid_polygons_df_list.append(row.to_frame().T)
            features_list.append([centroid.y, centroid.x, row["weather"]["temperature_celsius"], row["weather"]["humidity"]])
    if not valid_polygons_df_list: st.warning("Không còn đa giác nào có dữ liệu thời tiết hợp lệ."); return
    
    valid_polygons_gdf = gpd.GeoDataFrame(pd.concat(valid_polygons_df_list, ignore_index=True), crs=high_density_polygons.crs)
    features_np = np.array(features_list)

    # Feature Weights (giữ nguyên)
    st.sidebar.subheader("Feature Weights")
    lat_weight = st.sidebar.slider("Latitude", 0.1, 2.0, 1.0, key="lat_w_main")
    lon_weight = st.sidebar.slider("Longitude", 0.1, 2.0, 1.0, key="lon_w_main")
    temp_weight = st.sidebar.slider("Temperature", 0.1, 2.0, 0.5, key="temp_w_main")
    humid_weight = st.sidebar.slider("Humidity", 0.1, 2.0, 0.5, key="humid_w_main")
    
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_np)
    weighted_features = features_scaled * np.array([lat_weight, lon_weight, temp_weight, humid_weight])

    # Clustering Parameters (giữ nguyên)
    st.sidebar.subheader("Clustering Parameters (DBSCAN)")
    eps = st.sidebar.slider("EPS (Max Distance between Samples)", 0.01, 5.0, 1.0, step=0.01, key="eps_s_main")
    min_samples_max = max(1, len(weighted_features))
    min_samples_default = min(2, min_samples_max)
    min_samples = st.sidebar.slider("Minimum Samples per Cluster", 1, min_samples_max, min_samples_default, key="min_s_main")
    if weighted_features.shape[0] < min_samples: st.warning(f"Số mẫu tối thiểu ({min_samples}) lớn hơn số điểm dữ liệu ({weighted_features.shape[0]})."); return

    db = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean').fit(weighted_features)
    valid_polygons_gdf['cluster'] = db.labels_

    # --- TÍNH NĂNG TƯƠNG TÁC MỚI ---
    st.sidebar.markdown("---")
    st.sidebar.subheader("🗺️ Map Display Options")
    show_cluster_hulls = st.sidebar.checkbox("Hiển thị vùng bao Cluster (Convex Hull)", value=True, key="show_hulls")
    show_voronoi_cells = st.sidebar.checkbox("Hiển thị các ô Voronoi chi tiết", value=False, key="show_cells") # Mặc định ẩn đi để tập trung vào hull

    # Lọc Cluster theo ID
    unique_cluster_ids = sorted([c_id for c_id in valid_polygons_gdf['cluster'].unique() if c_id != -1]) # Bỏ qua nhiễu -1
    if unique_cluster_ids:
        selected_cluster_ids = st.sidebar.multiselect(
            "Lọc và hiển thị Clusters IDs:",
            options=unique_cluster_ids,
            default=unique_cluster_ids, # Mặc định chọn tất cả
            key="select_cluster_ids"
        )
    else:
        selected_cluster_ids = []
        st.sidebar.info("Không có cluster nào được hình thành (ngoại trừ nhiễu).")


    # --- Tạo bản đồ ---
    map_center_lat = features_np[:,0].mean() if features_np.shape[0] > 0 else gdf_input.geometry.y.mean()
    map_center_lon = features_np[:,1].mean() if features_np.shape[0] > 0 else gdf_input.geometry.x.mean()
    m = folium.Map(location=[map_center_lat, map_center_lon], zoom_start=5, tiles="CartoDB positron")

    # Lọc valid_polygons_gdf dựa trên selected_cluster_ids (bao gồm cả nhiễu nếu người dùng muốn)
    # Hoặc chỉ hiển thị các cluster được chọn và các ô nhiễu (nếu có)
    polygons_to_display_on_map = valid_polygons_gdf[
        valid_polygons_gdf['cluster'].isin(selected_cluster_ids + [-1]) # Thêm -1 để luôn hiển thị nhiễu nếu có
    ]


    # 1. Vẽ vùng bao Cluster (Convex Hull) - NẾU ĐƯỢC CHỌN
    if show_cluster_hulls and selected_cluster_ids:
        hulls_data = []
        for c_id in selected_cluster_ids: # Chỉ vẽ hull cho các cluster được chọn
            cluster_polygons = valid_polygons_gdf[valid_polygons_gdf['cluster'] == c_id]
            if not cluster_polygons.empty:
                # Gộp các geometry của cluster lại
                # dissolved_geometry = cluster_polygons.dissolve(by='cluster').geometry.iloc[0] # Cách 1: Dissolve
                united_geometry = unary_union(cluster_polygons.geometry.tolist()) # Cách 2: Unary Union
                
                if united_geometry.is_empty: continue

                hull = united_geometry.convex_hull
                
                # Thu thập thông tin cho popup của hull
                avg_temp = cluster_polygons['weather'].apply(lambda x: x['temperature_celsius'] if x else np.nan).mean()
                avg_humidity = cluster_polygons['weather'].apply(lambda x: x['humidity'] if x else np.nan).mean()
                weather_descs = list(set(wp['weather_description'] for wp in cluster_polygons['weather'] if wp))
                
                popup_html = f"""
                <b>Cluster ID: {c_id}</b><br>
                Số ô Voronoi: {len(cluster_polygons)}<br>
                Tổng số quỹ đạo (ước tính): {cluster_polygons['trajectory_count'].sum()}<br>
                Nhiệt độ TB: {avg_temp:.1f}°C<br>
                Độ ẩm TB: {avg_humidity:.1f}%<br>
                Thời tiết phổ biến: {', '.join(weather_descs[:3])}{'...' if len(weather_descs) > 3 else ''}
                """
                hulls_data.append({'cluster': c_id, 'geometry': hull, 'popup_html': popup_html, 'num_cells': len(cluster_polygons)})
        
        if hulls_data:
            cluster_hulls_gdf = gpd.GeoDataFrame(hulls_data, crs=valid_polygons_gdf.crs)
            folium.GeoJson(
                cluster_hulls_gdf,
                style_function=lambda feature: {
                    'fillColor': f'hsl({abs(feature["properties"]["cluster"])*60 % 360}, 80%, 60%)',
                    'color': f'hsl({abs(feature["properties"]["cluster"])*60 % 360}, 100%, 30%)',
                    'weight': 2.5,
                    'fillOpacity': 0.35,
                },
                tooltip=folium.features.GeoJsonTooltip(fields=['cluster', 'num_cells'], aliases=['Cluster ID:', 'Số ô Voronoi:']),
                popup=folium.features.GeoJsonPopup(fields=['popup_html'], labels=False, parse_html=True, max_width=300),
                name="Cluster Hulls"
            ).add_to(m)

    # 2. Vẽ các ô Voronoi chi tiết - NẾU ĐƯỢC CHỌN
    if show_voronoi_cells and not polygons_to_display_on_map.empty:
        folium.GeoJson(
            polygons_to_display_on_map.to_json(), # Chỉ vẽ các ô đã được lọc
            style_function=lambda feature: {
                'fillColor': f'hsl({abs(feature["properties"]["cluster"])*60 % 360}, 70%, 50%)' if feature["properties"]["cluster"] != -1 else '#AAAAAA', # Màu xám cho nhiễu
                'color': 'black',
                'weight': 0.5,
                'fillOpacity': 0.6 if feature["properties"]["cluster"] != -1 else 0.3,
            },
            tooltip=folium.GeoJsonTooltip(fields=['cluster', 'trajectory_count'], aliases=['Cluster ID:', 'Trajectory Count:']),
            name="Detailed Voronoi Cells"
        ).add_to(m)

    # 3. Vẽ CircleMarkers cho tâm các ô (có thể ẩn nếu đã có Hulls)
    # Có thể thêm một checkbox để bật/tắt CircleMarkers
    show_circle_markers = st.sidebar.checkbox("Hiển thị điểm tâm (CircleMarkers)", value=True, key="show_markers_cluster")
    if show_circle_markers and not polygons_to_display_on_map.empty:
        for _, row in polygons_to_display_on_map.iterrows(): # Chỉ vẽ marker cho các ô đã được lọc
            if row.geometry is not None and not row.geometry.is_empty:
                centroid = row.geometry.centroid
                weather = row["weather"]
                if weather:
                    popup_content = f"""
                    <b>Cluster: {row['cluster']}</b> ({'Noise' if row['cluster'] == -1 else 'Cluster Point'})<br>
                    Voronoi Cell Traj. Count: {row['trajectory_count']}<br>
                    Temp: {weather.get('temperature_celsius', 'N/A'):.1f}°C<br>
                    Humidity: {weather.get('humidity', 'N/A')}%<br>
                    Conditions: {weather.get('weather_description', 'N/A')}
                    """
                    folium.CircleMarker(
                        location=[centroid.y, centroid.x],
                        radius=4,
                        color=f'hsl({abs(row["cluster"])*60 % 360}, 100%, 25%)' if row["cluster"] != -1 else '#666666', # Viền đậm hơn
                        fill_color=f'hsl({abs(row["cluster"])*60 % 360}, 70%, 50%)' if row["cluster"] != -1 else '#AAAAAA',
                        fill_opacity=0.8,
                        popup=folium.Popup(popup_content, max_width=300),
                        tooltip=f"Cluster: {row['cluster']}, Count: {row['trajectory_count']}"
                    ).add_to(m)
    
    if show_cluster_hulls or show_voronoi_cells or show_circle_markers : # Chỉ thêm LayerControl nếu có gì đó để kiểm soát
        folium.LayerControl(collapsed=False).add_to(m)

    st.subheader("Clustering Map Results")
    folium_static(m, width=1300, height=650) # Sử dụng width 100%

    # --- Tính Silhouette Score (giữ nguyên) ---
    unique_clusters_obj = np.unique(db.labels_) # Sử dụng db.labels_ gốc cho unique_clusters
    num_actual_clusters = len(unique_clusters_obj) - (1 if -1 in unique_clusters_obj else 0)

    if num_actual_clusters >= 2 and weighted_features.shape[0] > num_actual_clusters :
        try:
            labels_for_score = db.labels_[db.labels_ != -1]
            features_for_score = weighted_features[db.labels_ != -1]
            if len(np.unique(labels_for_score)) >= 2 and len(labels_for_score) > len(np.unique(labels_for_score)):
                silhouette_avg = silhouette_score(features_for_score, labels_for_score)
                st.metric("Silhouette Score (cho các điểm không nhiễu)", f"{silhouette_avg:.3f}")
            else: st.info("Không đủ cụm hoặc điểm để tính Silhouette Score sau khi loại bỏ nhiễu.")
        except ValueError as e: st.warning(f"Không thể tính Silhouette Score: {e}")
    elif weighted_features.shape[0] <=1: st.warning("Không đủ điểm dữ liệu để tính Silhouette Score.")
    else: st.info("Cần ít nhất 2 cụm (không bao gồm nhiễu) để tính Silhouette Score.")

if __name__ == "__main__":
    main_app_page()