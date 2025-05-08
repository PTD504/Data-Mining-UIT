import streamlit as st
import pandas as pd
import geopandas as gpd
import folium
from streamlit_folium import st_folium
from folium.plugins import HeatMap

@st.cache_data
def load_data(uploaded_file):
    df = pd.read_csv(uploaded_file)
    df = df.dropna()
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['location-long'], df['location-lat']))
    gdf.set_crs(epsg=4326, inplace=True)
    return gdf

def main():
    st.title("Heatmap Visualization")

    uploaded_file = st.file_uploader("Upload your CSV", type=["csv"], key="heatmap_upload")
    if uploaded_file is not None:
        gdf = load_data(uploaded_file)

        # Create Folium Map
        map_center = [gdf.geometry.y.mean(), gdf.geometry.x.mean()]
        m = folium.Map(location=map_center, zoom_start=12)

        # HeatMap data
        heat_data = [[point.y, point.x] for point in gdf.geometry]

        HeatMap(heat_data, radius=10).add_to(m)

        st.subheader("Interactive Heatmap")
        st_folium(m, width=900, height=600, key="heatmap")

if __name__ == "__main__":
    main()
