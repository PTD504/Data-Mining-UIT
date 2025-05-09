# pages/3_Trajectory_Visualization.py
import streamlit as st
import pandas as pd
import geopandas as gpd
import folium
from streamlit_folium import folium_static
import random
import numpy as np
# import matplotlib.colors as mcolors # Không cần cho màu ngẫu nhiên đơn giản nữa
from shapely.geometry import LineString

# --- Hàm tiện ích ---
def get_random_html_color():
    """Tạo một mã màu HTML ngẫu nhiên."""
    return "#{:06x}".format(random.randint(0, 0xFFFFFF))

# --- Hàm chính của trang ---
def trajectory_visualization_page():
    st.set_page_config(page_title="Trajectory Visualization", layout="wide")
    st.title("🛰️ Interactive Trajectory Visualization")

    if 'gdf_data' not in st.session_state or st.session_state.gdf_data.empty:
        st.warning("⚠️ Vui lòng tải dữ liệu lên từ trang chính (app.py) trước.")
        return

    gdf_single_file = st.session_state.gdf_data.copy()

    st.sidebar.header("🚀 Trajectory Options")

    # --- Chọn cột định danh và timestamp (Giữ nguyên logic cũ) ---
    available_columns = gdf_single_file.columns.tolist()
    default_id_col_candidates = ['generated_individual_id', 'individual-local-identifier', 'tag-local-identifier']
    default_id_col = next((col for col in default_id_col_candidates if col in available_columns), None)
    if default_id_col is None and available_columns:
        default_id_col = available_columns[0]

    default_ts_col = 'timestamp' if 'timestamp' in available_columns else None
    if default_ts_col is None and len(available_columns) > 1:
         default_ts_col = available_columns[1] if default_id_col != available_columns[1] else (available_columns[0] if len(available_columns)>0 and default_id_col != available_columns[0] else None)

    id_column = st.sidebar.selectbox(
        "Cột định danh cá thể (Individual Identifier):",
        options=available_columns,
        index=(available_columns.index(default_id_col) if default_id_col and default_id_col in available_columns else 0),
        key="traj_single_id_col_v2" # Đổi key để tránh xung đột nếu có state cũ
    )
    timestamp_column = st.sidebar.selectbox(
        "Cột thời gian (Timestamp):",
        options=available_columns,
        index=(available_columns.index(default_ts_col) if default_ts_col and default_ts_col in available_columns else (1 if len(available_columns)>1 else 0) ),
        key="traj_single_ts_col_v2" # Đổi key
    )

    if not id_column or not timestamp_column:
        st.error("Vui lòng chọn cả cột định danh và cột thời gian."); return
    if id_column == timestamp_column:
        st.error("Cột định danh và cột thời gian không được trùng nhau."); return

    # --- Xử lý timestamp và sắp xếp (Giữ nguyên logic cũ) ---
    try:
        gdf_processed = gdf_single_file.copy()
        gdf_processed[timestamp_column] = pd.to_datetime(gdf_processed[timestamp_column], errors='coerce')
        gdf_processed.dropna(subset=[timestamp_column], inplace=True)
    except Exception as e:
        st.error(f"Lỗi chuyển đổi cột thời gian '{timestamp_column}': {e}. Đảm bảo cột có định dạng ngày giờ hợp lệ."); return
    if gdf_processed.empty:
        st.warning("Không còn dữ liệu sau khi xử lý timestamp hoặc dữ liệu đầu vào rỗng."); return
    try:
        gdf_sorted = gdf_processed.sort_values(by=[id_column, timestamp_column])
    except KeyError as e:
        st.error(f"Lỗi khi sắp xếp dữ liệu. Cột '{str(e)}' không tồn tại. Vui lòng kiểm tra lại lựa chọn cột.")
        return

    # --- Lọc theo khoảng thời gian (Giữ nguyên logic cũ, có thể cần key mới cho session state) ---
    st.sidebar.markdown("---")
    st.sidebar.subheader("⏳ Lọc theo Thời gian")
    if gdf_sorted.empty or not pd.api.types.is_datetime64_any_dtype(gdf_sorted[timestamp_column]):
        st.sidebar.warning("Không thể xác định khoảng thời gian từ dữ liệu hiện tại.")
    else:
        min_date_available = gdf_sorted[timestamp_column].min().date()
        max_date_available = gdf_sorted[timestamp_column].max().date()
        start_date_key_v2 = 'traj_single_date_filter_start_v2'
        end_date_key_v2 = 'traj_single_date_filter_end_v2'

        if start_date_key_v2 not in st.session_state or st.session_state[start_date_key_v2] < min_date_available or st.session_state[start_date_key_v2] > max_date_available:
            st.session_state[start_date_key_v2] = min_date_available
        if end_date_key_v2 not in st.session_state or st.session_state[end_date_key_v2] < min_date_available or st.session_state[end_date_key_v2] > max_date_available:
            st.session_state[end_date_key_v2] = max_date_available
        if st.session_state[start_date_key_v2] > st.session_state[end_date_key_v2] :
             st.session_state[start_date_key_v2] = st.session_state[end_date_key_v2]

        start_date_filter = st.sidebar.date_input("Ngày bắt đầu", value=st.session_state[start_date_key_v2],
                                                min_value=min_date_available, max_value=max_date_available,
                                                key="traj_single_start_date_picker_v2")
        end_date_filter = st.sidebar.date_input("Ngày kết thúc", value=st.session_state[end_date_key_v2],
                                              min_value=min_date_available, max_value=max_date_available,
                                              key="traj_single_end_date_picker_v2")
        if start_date_filter > end_date_filter:
            st.sidebar.error("Ngày bắt đầu không được lớn hơn ngày kết thúc.")
        else:
            st.session_state[start_date_key_v2] = start_date_filter
            st.session_state[end_date_key_v2] = end_date_filter
            start_datetime = pd.to_datetime(start_date_filter)
            end_datetime = pd.to_datetime(end_date_filter) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
            gdf_sorted = gdf_sorted[(gdf_sorted[timestamp_column] >= start_datetime) & (gdf_sorted[timestamp_column] <= end_datetime)]
            if gdf_sorted.empty:
                st.info(f"Không có quỹ đạo nào trong khoảng thời gian từ {start_date_filter.strftime('%Y-%m-%d')} đến {end_date_filter.strftime('%Y-%m-%d')}.")

    # --- Lọc cá thể và giới hạn quỹ đạo (Giữ nguyên logic cũ, có thể cần key mới) ---
    unique_ids = gdf_sorted[id_column].unique() if not gdf_sorted.empty else np.array([])
    if len(unique_ids) == 0: st.warning("Không có cá thể nào sau khi lọc."); return

    st.sidebar.markdown("---")
    st.sidebar.subheader("🎯 Lọc & Hiển thị Quỹ đạo")
    select_all_individuals = st.sidebar.checkbox("Hiển thị tất cả cá thể", value=True, key="traj_single_select_all_ids_v2")
    ids_to_plot_options = sorted(unique_ids.tolist())
    if select_all_individuals:
        selected_individual_ids = ids_to_plot_options
    else:
        selected_individual_ids = st.sidebar.multiselect("Chọn (các) cá thể:", options=ids_to_plot_options, default=ids_to_plot_options[0] if ids_to_plot_options else [], key="traj_single_multiselect_ids_v2")
    if not selected_individual_ids: st.info("Vui lòng chọn ít nhất một cá thể."); return
    
    max_total_trajectories = st.sidebar.slider("Số quỹ đạo tối đa hiển thị:", 1, max(1, len(selected_individual_ids)), min(50, len(selected_individual_ids)) if selected_individual_ids else 1, 1, key="traj_single_max_total_v2")

    # --- TÙY CHỌN HIỂN THỊ MARKER ĐIỂM ĐẦU/CUỐI (TÍNH NĂNG MỚI) ---
    show_start_end_markers = st.sidebar.checkbox("Hiển thị điểm Bắt đầu/Kết thúc", value=True, key="traj_show_markers")
    st.sidebar.markdown("---")


    # --- Tạo bản đồ (Giữ nguyên logic cũ) ---
    map_data_filtered_by_id = gdf_sorted[gdf_sorted[id_column].isin(selected_individual_ids)]
    if not map_data_filtered_by_id.empty:
        map_center_lat = map_data_filtered_by_id.geometry.y.mean(); map_center_lon = map_data_filtered_by_id.geometry.x.mean(); initial_zoom = 6
    else:
        map_center_lat = gdf_single_file.geometry.y.mean() if not gdf_single_file.empty else 0
        map_center_lon = gdf_single_file.geometry.x.mean() if not gdf_single_file.empty else 0
        initial_zoom = 2
        if selected_individual_ids : st.warning("Không có dữ liệu cho các cá thể đã chọn sau khi lọc.")

    m = folium.Map(location=[map_center_lat, map_center_lon], zoom_start=initial_zoom, tiles="CartoDB positron")
    total_trajectories_plotted = 0
    plotted_trajectories_info = []
    ids_actually_plotting = selected_individual_ids[:max_total_trajectories]

    for individual_id in ids_actually_plotting:
        trajectory_data = map_data_filtered_by_id[map_data_filtered_by_id[id_column] == individual_id].sort_values(by=timestamp_column)
        if len(trajectory_data) < 1: continue # Cần ít nhất 1 điểm để có thể có marker, 2 để có line

        points = list(zip(trajectory_data.geometry.y, trajectory_data.geometry.x))
        line_color = get_random_html_color()
        start_time = trajectory_data[timestamp_column].min()
        end_time = trajectory_data[timestamp_column].max()
        duration = end_time - start_time
        num_points = len(points)
        length_str = "N/A"
        try:
            line_geom_for_length = LineString(trajectory_data.geometry.apply(lambda p: (p.x, p.y)).tolist())
            if not line_geom_for_length.is_empty: # Kiểm tra nếu LineString không rỗng
                 projected_line = gpd.GeoSeries([line_geom_for_length], crs="EPSG:4326").to_crs("EPSG:3857")
                 length_meters = projected_line.length.iloc[0]
                 length_str = f"{length_meters/1000:.2f} km" if length_meters > 0 else "0 km"
        except Exception: pass

        popup_html = f"""<b>ID Cá thể:</b> {individual_id}<br><b>Số điểm:</b> {num_points}<br>
                         <b>Bắt đầu:</b> {start_time.strftime('%Y-%m-%d %H:%M:%S')}<br>
                         <b>Kết thúc:</b> {end_time.strftime('%Y-%m-%d %H:%M:%S')}<br>
                         <b>Thời gian bay:</b> {str(duration).split('.')[0]} (HH:MM:SS)<br>
                         <b>Chiều dài (ước tính):</b> {length_str}"""
        
        # Vẽ đường quỹ đạo nếu có đủ điểm
        if num_points >= 2:
            folium.PolyLine(
                points, color=line_color, tooltip=f"<b>ID:</b> {individual_id}<br><b>Điểm:</b> {num_points}",
                popup=folium.Popup(popup_html, max_width=350), weight=2.5, opacity=0.8
            ).add_to(m)

        # --- THÊM MARKER ĐIỂM ĐẦU VÀ CUỐI (TÍNH NĂNG MỚI) ---
        if show_start_end_markers and points: # Đảm bảo có `points`
            # Điểm bắt đầu
            folium.Marker(
                location=points[0],
                popup=f"<b>Bắt đầu: {individual_id}</b><br>{start_time.strftime('%Y-%m-%d %H:%M:%S')}",
                tooltip=f"Bắt đầu: {individual_id}",
                icon=folium.Icon(color='green', icon='play', prefix='fa') # Font Awesome icon
            ).add_to(m)

            # Điểm kết thúc (chỉ thêm nếu có nhiều hơn 1 điểm để phân biệt với điểm bắt đầu)
            if num_points > 1:
                folium.Marker(
                    location=points[-1],
                    popup=f"<b>Kết thúc: {individual_id}</b><br>{end_time.strftime('%Y-%m-%d %H:%M:%S')}",
                    tooltip=f"Kết thúc: {individual_id}",
                    icon=folium.Icon(color='red', icon='stop', prefix='fa') # Font Awesome icon
                ).add_to(m)
            # Nếu chỉ có 1 điểm, marker bắt đầu đã đủ
            elif num_points == 1 : # Trường hợp quỹ đạo chỉ có 1 điểm (chỉ vẽ marker bắt đầu)
                 pass # Marker bắt đầu đã được vẽ ở trên

        total_trajectories_plotted += 1
        plotted_trajectories_info.append({
            "Individual ID": individual_id, "Points": num_points, "Start Time": start_time,
            "End Time": end_time, "Duration": str(duration).split('.')[0],
            "Length (km, approx.)": length_str if length_str != "N/A" else None,
            "_Color_HTML_": f"<div style='width:15px; height:15px; background-color:{line_color if num_points >=2 else '#808080'}; border-radius:50%;'></div>" # Màu xám nếu chỉ có 1 điểm
        })

    # --- Hiển thị bản đồ và bảng thông tin (Giữ nguyên logic cũ) ---
    if total_trajectories_plotted > 0:
        st.info(f"Hiển thị {total_trajectories_plotted} quỹ đạo (đã lọc và giới hạn).")
        folium_static(m, height=650) # Bỏ width="100%"
        st.subheader("📜 Thông tin các quỹ đạo đang hiển thị:")
        df_display = pd.DataFrame(plotted_trajectories_info)
        df_display_cols = ["Individual ID", "Points", "Start Time", "End Time", "Duration", "Length (km, approx.)", "_Color_HTML_"]
        df_display = df_display[[col for col in df_display_cols if col in df_display.columns]]
        if 'Start Time' in df_display: df_display['Start Time'] = df_display['Start Time'].dt.strftime('%Y-%m-%d %H:%M')
        if 'End Time' in df_display: df_display['End Time'] = df_display['End Time'].dt.strftime('%Y-%m-%d %H:%M')
        # Dòng 217 trong tệp D:\Data-Mining-UIT\app\pages\trajectories.py
        st.dataframe(df_display, use_container_width=True, hide_index=True,
                    column_config={"_Color_HTML_": st.column_config.TextColumn(label="Màu")})
    else: st.info("Không có quỹ đạo nào để hiển thị dựa trên các lựa chọn hiện tại.")

if __name__ == "__main__":
    trajectory_visualization_page()