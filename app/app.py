import streamlit as st
import pandas as pd
import geopandas as gpd

# Sử dụng @st.cache_data cho hàm tải dữ liệu để tối ưu hóa
@st.cache_data
def load_data_from_uploaded_file(uploaded_file):
    st.write(f"Loading data from: {uploaded_file.name}") # Debug
    df = pd.read_csv(uploaded_file)
    # Kiểm tra xem có cột 'longitude' và 'latitude' không
    if 'longitude' not in df.columns or 'latitude' not in df.columns:
        st.error("CSV file must contain 'longitude' and 'latitude' columns.")
        return None
    df = df.dropna(subset=['longitude', 'latitude'])
    if df.empty:
        st.warning("No valid data after dropping rows with missing longitude/latitude.")
        return None
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['longitude'], df['latitude']))
    if gdf.crs is None:
        gdf.set_crs(epsg=4326, inplace=True)
    else:
        gdf = gdf.to_crs(epsg=4326)
    return gdf

st.sidebar.title("Data Upload")
# st.sidebar.header("Upload Your Data") # Có thể bỏ header này nếu title đã rõ ràng

uploaded_file = st.sidebar.file_uploader("Upload your CSV", type=["csv"], key="global_uploader")

if uploaded_file is not None:
    # Logic để chỉ load lại khi file thực sự thay đổi
    # (Giữ nguyên logic kiểm tra tên và kích thước file nếu bạn muốn)
    if 'uploaded_file_obj_name' not in st.session_state or \
       st.session_state.uploaded_file_obj_name != uploaded_file.name or \
       'uploaded_file_obj_size' not in st.session_state or \
       st.session_state.uploaded_file_obj_size != uploaded_file.size:

        st.session_state.uploaded_file_obj_name = uploaded_file.name
        st.session_state.uploaded_file_obj_size = uploaded_file.size

        # Xóa dữ liệu đã xử lý cũ nếu có file mới được tải lên
        if 'gdf_data' in st.session_state:
            del st.session_state.gdf_data
        st.sidebar.info(f"New file detected: {uploaded_file.name}. Processing...")


    if 'gdf_data' not in st.session_state: # Chỉ load nếu chưa có trong session
        with st.spinner("Loading and processing data..."):
            data = load_data_from_uploaded_file(uploaded_file)
            if data is not None and not data.empty:
                st.session_state.gdf_data = data
                st.sidebar.success("Data loaded successfully!")
            elif data is None: # Lỗi đã được xử lý trong hàm load
                pass
            else:
                st.sidebar.warning("Data loaded but it's empty or invalid.")
                if 'gdf_data' in st.session_state: # Xóa nếu data không hợp lệ
                    del st.session_state.gdf_data

elif 'gdf_data' in st.session_state : # Nếu không có file nào được upload nhưng vẫn còn data cũ
    st.sidebar.info(f"Using previously loaded data from: {st.session_state.get('uploaded_file_obj_name', 'previous session')}")


# Nội dung chính của app.py (trang chủ)
st.title("Welcome to Bird Trajectory Analysis App!")
st.markdown("""
This application helps you analyze bird migration trajectories.
Please upload your CSV data using the sidebar to begin.

Once data is uploaded, you can navigate to different analysis pages using the links
that will appear at the top of the sidebar:
- **Main App:** For clustering according to migration trajectories.
- **Heatmap:** To visualize data density.
- **Trajectory Visualization:** To view individual flight paths.
""")

if 'gdf_data' not in st.session_state:
    st.warning("No data loaded. Please upload a CSV file using the sidebar.")
else:
    st.success("Data is loaded. Please use the navigation links in the sidebar (above the 'Data Upload' section) to explore different analyses.")
    st.subheader("Quick data preview (first 5 rows):")
    st.dataframe(st.session_state.gdf_data.drop(columns='geometry').head() if 'gdf_data' in st.session_state else pd.DataFrame())