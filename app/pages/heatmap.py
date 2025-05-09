# pages/2_Heatmap.py
import streamlit as st
import pandas as pd
import geopandas as gpd
import folium
from streamlit_folium import folium_static
from folium.plugins import HeatMap
import json # Để lưu/tải cấu hình (nếu muốn mở rộng)

# --- Các hàm tiện ích ---
def get_default_heatmap_options():
    """Trả về một dictionary chứa các tùy chọn heatmap mặc định."""
    return {
        "radius": 15,
        "blur": 10,
        "tile_layer": "OpenStreetMap",
        "filter_column": None,
        "selected_filter_values": [],
    }

def initialize_heatmap_session_state():
    """Khởi tạo các giá trị session_state cho heatmap nếu chúng chưa tồn tại."""
    default_options = get_default_heatmap_options()
    for key, value in default_options.items():
        if f"heatmap_{key}" not in st.session_state:
            st.session_state[f"heatmap_{key}"] = value

def reset_heatmap_options():
    """Đặt lại các tùy chọn heatmap về giá trị mặc định."""
    default_options = get_default_heatmap_options()
    for key, value in default_options.items():
        st.session_state[f"heatmap_{key}"] = value
    # Xóa luôn các state phụ trợ nếu có
    if "heatmap_available_filter_values" in st.session_state:
        del st.session_state.heatmap_available_filter_values
    st.success("Heatmap options have been reset to defaults.")


# --- Hàm chính của trang Heatmap ---
def heatmap_page():
    st.set_page_config(page_title="Enhanced Heatmap", layout="wide")
    st.title("🔥 Enhanced Heatmap Visualization")

    # Khởi tạo session state cho các tùy chọn heatmap
    initialize_heatmap_session_state()

    # Kiểm tra dữ liệu đầu vào
    if 'gdf_data' not in st.session_state or st.session_state.gdf_data.empty:
        st.warning("⚠️ Please upload data through the main app page (app.py) first.")
        st.info("Navigate to the 'app' page using the sidebar to upload your CSV file.")
        return

    gdf_full = st.session_state.gdf_data.copy()
    data_for_heatmap = gdf_full.copy() # Bắt đầu với tất cả dữ liệu

    # --- Sidebar cho các tùy chọn Heatmap ---
    with st.sidebar:
        st.header("Heatmap Customization")

        # 1. Nút Đặt lại về Mặc định
        if st.button("🔄 Reset to Default Options", key="reset_heatmap"):
            reset_heatmap_options()
            # st.experimental_rerun() # Chạy lại để cập nhật widget với giá trị mới

        st.markdown("---") # Ngăn cách

        # 2. Tùy chỉnh Tham số Heatmap cơ bản
        st.subheader("Basic Parameters")
        st.session_state.heatmap_radius = st.slider(
            "Radius (ảnh hưởng của điểm)",
            min_value=1, max_value=50,
            value=st.session_state.heatmap_radius, # Lấy từ session_state
            step=1, key="hm_radius_slider"
        )
        st.session_state.heatmap_blur = st.slider(
            "Blur (độ mượt/lan tỏa)",
            min_value=1, max_value=50,
            value=st.session_state.heatmap_blur, # Lấy từ session_state
            step=1, key="hm_blur_slider"
        )

        st.markdown("---")

        # 3. Lựa chọn Bản đồ Nền
        st.subheader("Map Tiles")
        tile_options = [
            "OpenStreetMap", "CartoDB positron", "CartoDB dark_matter",
            "Stamen Terrain", "Stamen Toner", "Stamen Watercolor",
            "Esri WorldStreetMap", "Esri WorldImagery"
        ]
        # Lấy index của giá trị hiện tại trong session_state để đặt cho selectbox
        current_tile_index = tile_options.index(st.session_state.heatmap_tile_layer) if st.session_state.heatmap_tile_layer in tile_options else 0
        st.session_state.heatmap_tile_layer = st.selectbox(
            "Select Map Tile Layer:",
            options=tile_options,
            index=current_tile_index, # Đặt giá trị mặc định/hiện tại
            key="hm_tile_select"
        )

        st.markdown("---")

        # 4. Lọc Dữ liệu
        st.subheader("Data Filtering")
        # Tìm các cột có thể là danh mục (ít giá trị duy nhất, kiểu object/category)
        potential_filter_cols = [None] + [
            col for col in data_for_heatmap.columns
            if data_for_heatmap[col].dtype == 'object' and data_for_heatmap[col].nunique() < 100 and data_for_heatmap[col].nunique() > 1
        ]
        if len(potential_filter_cols) == 1 and potential_filter_cols[0] is None: # Chỉ có [None]
            st.write("No suitable categorical columns found for filtering.")
        else:
            # Lấy index của cột lọc hiện tại
            current_filter_col_index = potential_filter_cols.index(st.session_state.heatmap_filter_column) if st.session_state.heatmap_filter_column in potential_filter_cols else 0
            new_filter_column = st.selectbox(
                "Filter by Category (Optional):",
                options=potential_filter_cols,
                index=current_filter_col_index,
                key="hm_filter_col_select",
                format_func=lambda x: "None (Show All)" if x is None else x
            )

            # Nếu cột lọc thay đổi, reset các giá trị đã chọn
            if new_filter_column != st.session_state.heatmap_filter_column:
                st.session_state.heatmap_filter_column = new_filter_column
                st.session_state.heatmap_selected_filter_values = [] # Reset lựa chọn
                if "heatmap_available_filter_values" in st.session_state:
                    del st.session_state.heatmap_available_filter_values # Xóa cache giá trị cũ

            if st.session_state.heatmap_filter_column:
                # Lấy các giá trị duy nhất cho cột đã chọn để lọc
                # Cache lại để không phải tính toán mỗi lần
                if "heatmap_available_filter_values" not in st.session_state or \
                   st.session_state.get("heatmap_filter_column_for_cache") != st.session_state.heatmap_filter_column:

                    st.session_state.heatmap_available_filter_values = sorted(list(data_for_heatmap[st.session_state.heatmap_filter_column].dropna().unique()))
                    st.session_state.heatmap_filter_column_for_cache = st.session_state.heatmap_filter_column


                if not st.session_state.heatmap_available_filter_values:
                    st.write(f"No unique values in '{st.session_state.heatmap_filter_column}' to filter by.")
                else:
                    # Khôi phục các giá trị đã chọn nếu chúng vẫn hợp lệ
                    valid_selected_values = [v for v in st.session_state.heatmap_selected_filter_values if v in st.session_state.heatmap_available_filter_values]
                    if not valid_selected_values and st.session_state.heatmap_available_filter_values: # Nếu không có lựa chọn hợp lệ nào, chọn tất cả mặc định
                        valid_selected_values = st.session_state.heatmap_available_filter_values


                    st.session_state.heatmap_selected_filter_values = st.multiselect(
                        f"Select values from '{st.session_state.heatmap_filter_column}':",
                        options=st.session_state.heatmap_available_filter_values,
                        default=valid_selected_values, # Giữ lại lựa chọn trước đó nếu có thể
                        key="hm_filter_values_multiselect"
                    )

                    if st.session_state.heatmap_selected_filter_values:
                        data_for_heatmap = data_for_heatmap[data_for_heatmap[st.session_state.heatmap_filter_column].isin(st.session_state.heatmap_selected_filter_values)]
                    else: # Nếu không chọn giá trị nào (sau khi đã chọn cột), coi như không lọc theo cột đó
                        st.info(f"No values selected for '{st.session_state.heatmap_filter_column}'. Showing all data for other filters.")
                        # Hoặc có thể coi là không hiển thị gì:
                        # data_for_heatmap = pd.DataFrame(columns=data_for_heatmap.columns)


    # --- Xử lý và hiển thị Heatmap ---
    if data_for_heatmap.empty:
        st.warning("ℹ️ No data available for heatmap after applying filters.")
        return

    st.subheader("Generated Heatmap")
    num_points_for_heatmap = len(data_for_heatmap)
    st.caption(f"Displaying heatmap for {num_points_for_heatmap} data points based on current settings.")

    # Tính toán vị trí trung tâm cho bản đồ
    try:
        map_center_lat = data_for_heatmap.geometry.y.mean()
        map_center_lon = data_for_heatmap.geometry.x.mean()
        initial_zoom = 5
    except Exception: # Nếu không có điểm nào hoặc lỗi tính mean
        map_center_lat = 0 # fallback
        map_center_lon = 0 # fallback
        initial_zoom = 2
        st.error("Could not determine map center. Using default (0,0).")


    # Tạo bản đồ Folium
    m = folium.Map(
        location=[map_center_lat, map_center_lon],
        zoom_start=initial_zoom,
        tiles=st.session_state.heatmap_tile_layer # Sử dụng tile đã chọn
    )

    # Chuẩn bị dữ liệu cho HeatMap plugin
    heat_data = [[point.y, point.x]
                 for point in data_for_heatmap.geometry
                 if pd.notna(point.y) and pd.notna(point.x)]

    if heat_data:
        HeatMap(
            heat_data,
            radius=st.session_state.heatmap_radius,
            blur=st.session_state.heatmap_blur,
        ).add_to(m)
    else:
        st.warning("⚠️ No valid point data to generate heatmap after processing all filters.")

    # Hiển thị bản đồ
    folium_static(m, width=1200, height=650)

    st.markdown("---")

    # 5. Tùy chọn Tải xuống Dữ liệu Đã lọc
    st.subheader("Download Filtered Data")
    if not data_for_heatmap.empty:
        # Chuyển đổi GeoDataFrame thành DataFrame (loại bỏ cột geometry) để dễ dàng xuất CSV
        df_to_download = pd.DataFrame(data_for_heatmap.drop(columns='geometry'))
        csv_data = df_to_download.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Filtered Data as CSV",
            data=csv_data,
            file_name=f"filtered_heatmap_data_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            key="download_filtered_heatmap_data"
        )
    else:
        st.info("No data to download based on current filters.")


if __name__ == "__main__":
    heatmap_page()