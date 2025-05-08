import streamlit as st
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Polygon
from scipy.spatial import Voronoi
from sklearn.cluster import DBSCAN
import folium
from streamlit_folium import st_folium
from sklearn.metrics import silhouette_score
from shapely.geometry import MultiPoint

def get_bounding_area(gdf, buffer_deg=0.01):
    bounds = gdf.total_bounds  # (minx, miny, maxx, maxy)
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
        centroid = poly.centroid
        circle = centroid.buffer(radius_deg)  # buffer in degrees (~111 km/deg latitude)
        clipped = poly.intersection(circle)
        clipped_polygons.append(clipped)
    return gpd.GeoDataFrame(geometry=clipped_polygons, crs="EPSG:4326")


@st.cache_data
def create_voronoi_grid(coords, _boundary_geom=None):
    vor = Voronoi(coords)
    polygons = []
    for region in vor.regions:
        if len(region) > 0 and -1 not in region:
            try:
                polygon = Polygon([vor.vertices[i] for i in region])
                if _boundary_geom:
                    polygon = polygon.intersection(_boundary_geom)
                if polygon.is_valid and not polygon.is_empty:
                    polygons.append(polygon)
            except:
                continue
    return gpd.GeoDataFrame(geometry=polygons, crs='EPSG:4326')

@st.cache_data
def load_data(uploaded_file):
    df = pd.read_csv(uploaded_file)
    df = df.dropna()
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['longitude'], df['latitude']))
    gdf.set_crs(epsg=4326, inplace=True)
    return gdf

def main():
    uploaded_file = st.file_uploader("Upload your CSV", type=["csv"])
    if uploaded_file is not None:
        gdf = load_data(uploaded_file)
        coords = np.array(list(zip(gdf.geometry.x, gdf.geometry.y)))
        
        # 1. Create boundary around your data (with buffer)
        boundary_geom = get_bounding_area(gdf, buffer_deg=0.01)
        voronoi_grid = create_voronoi_grid(coords, _boundary_geom=boundary_geom)
        voronoi_grid = clip_voronoi_cells(voronoi_grid, radius_deg=0.05)

        # 3. Ensure geometries are valid
        gdf = gdf[gdf.geometry.notnull()]
        voronoi_grid = voronoi_grid[voronoi_grid.geometry.notnull()]
        voronoi_grid.reset_index(drop=True, inplace=True)

        # 4. Assign points to polygons using spatial join
        joined = gpd.sjoin(gdf, voronoi_grid, how='left', predicate='intersects')

        # 5. Count how many points fall into each polygon
        density = joined.groupby('index_right').size()
        voronoi_grid['trajectory_count'] = voronoi_grid.index.map(density).fillna(0)


        threshold = st.slider("Preservable Area Threshold", 1, 100, 10, key='threshold')
        high_density_polygons = voronoi_grid[voronoi_grid['trajectory_count'] > threshold].copy()

        if len(high_density_polygons) >= 3:
            coords_cluster = np.array([[geom.centroid.x, geom.centroid.y] for geom in high_density_polygons.geometry])

            # Convert to radians for haversine distance
            coords_rad = np.radians(coords_cluster)

            # Earth radius in km
            kms_per_radian = 6371.0088

            # User input for clustering radius in kilometers
            eps_km = st.slider("DBSCAN eps (in km)", 10, 500, 100)  # Adjust range as needed
            eps_rad = eps_km / kms_per_radian

            # User input for minimum samples
            min_samples = st.slider("DBSCAN - min_samples", 1, 10, 2)

            # Run DBSCAN with Haversine metric
            db = DBSCAN(eps=eps_rad, min_samples=min_samples, metric='haversine').fit(coords_rad)
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
                popup_content = f"Centroid Longitude: {centroid.x:.6f}<br>Centroid Latitude: {centroid.y:.6f}"
                folium.CircleMarker(
                    location=[centroid.y, centroid.x],
                    radius=3,
                    color='red',
                    fill=True,
                    fill_opacity=0.6,
                    popup=folium.Popup(popup_content, max_width=200),
                ).add_to(m)

            st.subheader("Interactive Preservable Area Map")
            st_folium(m, width=900, height=600, key='folium_map')

            # Exclude noise points
            mask = high_density_polygons['cluster'] != -1
            if len(set(high_density_polygons[mask]['cluster'])) > 1:
                score = silhouette_score(coords_rad[mask], high_density_polygons[mask]['cluster'], metric='euclidean')
                st.write(f"Silhouette Score: {score:.3f}")
            else:
                st.warning("Not enough clusters to compute Silhouette Score.")
        else:
            st.warning("Not enough high-density polygons to form clusters. Try lowering the threshold or adjusting DBSCAN parameters.")

if __name__ == "__main__":
    main()
