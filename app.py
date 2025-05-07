import streamlit as st
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Polygon
from scipy.spatial import Voronoi
from sklearn.cluster import DBSCAN
import folium
from streamlit_folium import st_folium

@st.cache_data
def create_voronoi_grid(coords):
    vor = Voronoi(coords)
    polygons = []
    for region in vor.regions:
        if len(region) > 0 and -1 not in region:
            try:
                polygon = Polygon([vor.vertices[i] for i in region])
                polygons.append(polygon)
            except:
                continue
    return gpd.GeoDataFrame(geometry=polygons, crs='EPSG:4326')

@st.cache_data
def load_data(uploaded_file):
    df = pd.read_csv(uploaded_file)
    df = df.dropna()
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['longitude'], df['latitude']))  # thong nhat ten columns di
    gdf.set_crs(epsg=4326, inplace=True)
    gdf = gdf.sample(n=5000, random_state=42)
    return gdf

def main():
    uploaded_file = st.file_uploader("Upload your CSV", type=["csv"])
    if uploaded_file is not None:
        gdf = load_data(uploaded_file)
        coords = np.array(list(zip(gdf.geometry.x, gdf.geometry.y)))
        voronoi_grid = create_voronoi_grid(coords)
        
        # Assign points to polygons
        joined = gpd.sjoin(gdf, voronoi_grid, how='left')
        density = joined.groupby('index_right').size()
        voronoi_grid['trajectory_count'] = density
        voronoi_grid['trajectory_count'] = voronoi_grid['trajectory_count'].fillna(0)

        threshold = st.slider("Preservable Area Threshold", 1, 100, 10, key='threshold')
        high_density_polygons = voronoi_grid[voronoi_grid['trajectory_count'] > threshold].copy()

        if len(high_density_polygons) >= 3:
            coords_cluster = np.array([[geom.centroid.x, geom.centroid.y] for geom in high_density_polygons.geometry])

            eps = st.slider("DBSCAN - eps (radius)", 0.01, 1.0, 0.1, 0.01, key='eps')
            min_samples = st.slider("DBSCAN - min_samples", 1, 20, 3, 1, key='min_samples')

            db = DBSCAN(eps=eps, min_samples=min_samples).fit(coords_cluster)
            high_density_polygons['cluster'] = db.labels_

            # Create Folium Map
            map_center = [gdf.geometry.y.mean(), gdf.geometry.x.mean()]
            m = folium.Map(location=map_center, zoom_start=12)
            
            folium.GeoJson(
                high_density_polygons.to_json(),
                style_function=lambda feature: {
                    'fillColor': f'#{hex(abs(feature["properties"]["cluster"] * 137) % 256)[2:].zfill(2)}55aa55',
                    'color': 'black',
                    'weight': 2,
                    'fillOpacity': 0.6,
                },
                tooltip=folium.GeoJsonTooltip(fields=['cluster', 'trajectory_count']),
            ).add_to(m)

            for _, row in high_density_polygons.iterrows():
                centroid = row.geometry.centroid
                folium.CircleMarker(
                    location=[centroid.y, centroid.x],
                    radius=8,
                    color='green',
                    fill=True,
                    fill_opacity=0.6,
                    popup=f"Preservable Area: {row['trajectory_count']} points",
                ).add_to(m)

            st.subheader("Interactive Preservable Area Map")
            st_folium(m, width=900, height=600, key='folium_map')

        else:
            st.warning("Not enough high-density polygons to form clusters. Try lowering the threshold or adjusting DBSCAN parameters.")

if __name__ == "__main__":
    main()
