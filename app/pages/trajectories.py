import streamlit as st
import pandas as pd
import geopandas as gpd
import numpy as np
import folium
from streamlit_folium import folium_static
import random # Để tạo màu ngẫu nhiên cho các quỹ đạo

def get_random_color():
    return "#{:06x}".format(random.randint(0, 0xFFFFFF))

def trajectory_visualization_page():
    st.title("Trajectory Visualization")

    if 'gdf_data' not in st.session_state or st.session_state.gdf_data.empty:
        st.warning("Please upload data through the sidebar first.")
        # Nút này có thể không hoạt động như mong đợi để điều hướng giữa các trang đa trang của Streamlit
        # Người dùng nên được hướng dẫn sử dụng sidebar chính của Streamlit
        # if st.button("Go to Upload Data"):
        #     st.info("Please use the sidebar to navigate to the data upload section.")
        return

    gdf_full = st.session_state.gdf_data.copy() # Sử dụng bản sao để thao tác

    st.header("Select Columns for Trajectory Plotting")

    available_columns = gdf_full.columns.tolist()

    # Cố gắng tự động phát hiện các cột thường dùng
    default_id_col = None
    if 'generated_individual_id' in available_columns:
        default_id_col = 'generated_individual_id'
    elif 'individual-local-identifier' in available_columns:
        default_id_col = 'individual-local-identifier'
    elif 'tag-local-identifier' in available_columns:
        default_id_col = 'tag-local-identifier'
    # Bỏ qua event-id vì nó thường không phải là định danh cá thể cho quỹ đạo dài
    # elif 'event-id' in available_columns:
    #     pass

    default_ts_col = None
    if 'timestamp' in available_columns:
        default_ts_col = 'timestamp'

    # Cho phép người dùng chọn cột định danh và cột timestamp
    col1, col2 = st.columns(2)
    with col1:
        id_column_index = available_columns.index(default_id_col) if default_id_col and default_id_col in available_columns else 0
        id_column = st.selectbox(
            "Select Individual Identifier Column:",
            options=available_columns,
            index=id_column_index
        )
    with col2:
        timestamp_column_index = available_columns.index(default_ts_col) if default_ts_col and default_ts_col in available_columns else 0
        timestamp_column = st.selectbox(
            "Select Timestamp Column:",
            options=available_columns,
            index=timestamp_column_index
        )

    if not id_column or not timestamp_column:
        st.error("Please select both an identifier and a timestamp column.")
        return
    
    if id_column == timestamp_column:
        st.error("Identifier column and Timestamp column cannot be the same.")
        return

    # Kiểm tra xem cột timestamp có thể chuyển đổi sang datetime không
    try:
        # Tạo bản sao để tránh SettingWithCopyWarning nếu gdf_full được dùng lại
        gdf_processed = gdf_full.copy()
        gdf_processed[timestamp_column] = pd.to_datetime(gdf_processed[timestamp_column], errors='coerce')
        if gdf_processed[timestamp_column].isnull().any():
            st.warning(f"Some values in '{timestamp_column}' could not be converted to datetime and were set to NaT. These rows will be dropped for trajectory plotting.")
            gdf_processed.dropna(subset=[timestamp_column], inplace=True)
    except Exception as e:
        st.error(f"Could not convert timestamp column '{timestamp_column}' to datetime: {e}")
        st.info("Please ensure the timestamp column is in a recognizable format (e.g., YYYY-MM-DD HH:MM:SS, or Unix timestamp).")
        return

    if gdf_processed.empty:
        st.warning("No data remaining after attempting to process timestamps.")
        return

    # Sắp xếp dữ liệu theo ID và timestamp
    try:
        gdf_sorted = gdf_processed.sort_values(by=[id_column, timestamp_column])
    except KeyError as e:
        st.error(f"Selected column for sorting ('{id_column}' or '{timestamp_column}') not found: {e}. This might indicate an issue with column selection or data processing.")
        return


    # Lấy danh sách các ID duy nhất để người dùng có thể chọn
    unique_ids = gdf_sorted[id_column].unique() # Đây là một mảng NumPy
    if len(unique_ids) == 0: # Kiểm tra kích thước của mảng
        st.warning(f"No unique IDs found in column '{id_column}'. Cannot plot trajectories.")
        return

    st.header("Filter Trajectories")
    # Cho phép chọn tất cả hoặc một số ID cụ thể
    selected_ids_option = st.radio(
        "Show trajectories for:",
        ("All Individuals", "Selected Individuals"),
        index=0, key="select_ids_radio"
    )

    ids_to_plot = [] # Khởi tạo là danh sách rỗng
    if selected_ids_option == "All Individuals":
        ids_to_plot = unique_ids # unique_ids là một mảng NumPy
    else:
        # st.multiselect trả về một danh sách
        ids_to_plot = st.multiselect(
            "Select individuals to plot:",
            options=unique_ids.tolist(), # Chuyển mảng NumPy thành list cho options
            default=unique_ids[0] if len(unique_ids) > 0 else []
        )

    # *** SỬA LỖI Ở ĐÂY ***
    if len(ids_to_plot) == 0: # Kiểm tra xem danh sách/mảng ids_to_plot có rỗng không
        st.info("Please select/ensure there is at least one individual to plot a trajectory.")
        return

    # Tạo bản đồ Folium
    if gdf_sorted.empty or 'geometry' not in gdf_sorted.columns or gdf_sorted['geometry'].is_empty.all():
        st.error("No valid geometries to plot.")
        return

    # Tính toán vị trí trung tâm dựa trên dữ liệu được lọc
    # Ensure ids_to_plot is a list for isin if it came from unique_ids (numpy array)
    ids_to_plot_list = ids_to_plot.tolist() if isinstance(ids_to_plot, np.ndarray) else ids_to_plot
    
    filtered_gdf_for_map = gdf_sorted[gdf_sorted[id_column].isin(ids_to_plot_list)]
    
    map_center_lat = 0
    map_center_lon = 0
    zoom_start = 2 # Mặc định nếu không có dữ liệu

    if not filtered_gdf_for_map.empty:
        map_center_lat = filtered_gdf_for_map.geometry.y.mean()
        map_center_lon = filtered_gdf_for_map.geometry.x.mean()
        zoom_start = 5
    elif not gdf_full.empty : # Fallback về toàn bộ dữ liệu nếu lựa chọn lọc không có kết quả
        st.warning("No data for the selected individuals. Showing map centered on all data.")
        map_center_lat = gdf_full.geometry.y.mean()
        map_center_lon = gdf_full.geometry.x.mean()
        zoom_start = 4


    m = folium.Map(location=[map_center_lat, map_center_lon], zoom_start=zoom_start)

    # Vẽ quỹ đạo cho các ID đã chọn
    num_plotted = 0
    
    # Đảm bảo max_trajectories_to_display có giá trị hợp lý
    min_slider_val = 1
    max_slider_val = max(min_slider_val, len(ids_to_plot_list)) # Đảm bảo max >= min
    default_slider_val = min(50, max_slider_val) if max_slider_val > 0 else min_slider_val

    if max_slider_val == min_slider_val and max_slider_val == 1 and len(ids_to_plot_list) ==1 : # Trường hợp chỉ có 1 trajectory
         max_trajectories_to_display = 1
         st.caption(f"Displaying 1 trajectory.")
    elif max_slider_val > min_slider_val:
        max_trajectories_to_display = st.slider(
            "Max trajectories to display at once (if 'All' is selected and count is high)",
            min_slider_val,
            max_slider_val,
            default_slider_val,
            key="max_traj_slider",
            help="Adjust to prevent browser slowdowns with many trajectories."
        )
    else: # Trường hợp không có trajectory nào hoặc lỗi slider
        max_trajectories_to_display = 0


    ids_actually_plotting = []
    if selected_ids_option == "All Individuals" and len(ids_to_plot_list) > max_trajectories_to_display:
        st.info(f"Showing the first {max_trajectories_to_display} trajectories out of {len(ids_to_plot_list)} due to performance limit.")
        ids_actually_plotting = ids_to_plot_list[:max_trajectories_to_display]
    else:
        ids_actually_plotting = ids_to_plot_list


    for individual_id in ids_actually_plotting:
        trajectory_data = gdf_sorted[gdf_sorted[id_column] == individual_id]
        if len(trajectory_data) < 2: # Cần ít nhất 2 điểm để vẽ đường
            continue

        points = list(zip(trajectory_data.geometry.y, trajectory_data.geometry.x))

        # Thêm đường quỹ đạo
        line_color = get_random_color()
        folium.PolyLine(
            points,
            color=line_color,
            tooltip=f"ID: {individual_id}", # Thêm tooltip cho đường
            weight=2, # Độ dày của đường
            opacity=0.8
        ).add_to(m)

        # Tùy chọn: thêm marker cho điểm đầu và cuối
        if points:
            folium.Marker(
                location=points[0],
                popup=f"Start: {individual_id}<br>Time: {trajectory_data.iloc[0][timestamp_column]}",
                tooltip=f"Start: {individual_id}",
                icon=folium.Icon(color='green', icon='play')
            ).add_to(m)
            folium.Marker(
                location=points[-1],
                popup=f"End: {individual_id}<br>Time: {trajectory_data.iloc[-1][timestamp_column]}",
                tooltip=f"End: {individual_id}",
                icon=folium.Icon(color='red', icon='stop')
            ).add_to(m)
        num_plotted +=1

    if num_plotted > 0:
        st.subheader(f"Displaying {num_plotted} trajectories")
        folium_static(m, width=1200, height=700)
    else:
        st.info("No trajectories to display based on current selection (e.g., individuals might have less than 2 points, or no individuals selected).")

if __name__ == "__main__":
    trajectory_visualization_page()