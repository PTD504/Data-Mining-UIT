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
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import haversine_distances
from utils import get_weather_data  # Your existing utility function

api_key = "d62e6942105fef7a514b277c5bbbc956"

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
        centroid = poly.centroid
        circle = centroid.buffer(radius_deg)
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
    st.title("Weather-Aware Spatial Clustering")
    
    uploaded_file = st.file_uploader("Upload your CSV", type=["csv"])
    if uploaded_file is not None:
        gdf = load_data(uploaded_file)
        coords = np.array(list(zip(gdf.geometry.x, gdf.geometry.y)))
        
        # Create Voronoi grid
        boundary_geom = get_bounding_area(gdf, buffer_deg=0.01)
        voronoi_grid = create_voronoi_grid(coords, _boundary_geom=boundary_geom)
        voronoi_grid = clip_voronoi_cells(voronoi_grid, radius_deg=0.05)

        # Process data
        gdf = gdf[gdf.geometry.notnull()]
        voronoi_grid = voronoi_grid[voronoi_grid.geometry.notnull()]
        voronoi_grid.reset_index(drop=True, inplace=True)

        # Spatial join and count
        joined = gpd.sjoin(gdf, voronoi_grid, how='left', predicate='intersects')
        density = joined.groupby('index_right').size()
        voronoi_grid['trajectory_count'] = voronoi_grid.index.map(density).fillna(0)

        # Get weather data
        threshold = st.slider("Preservable Area Threshold", 1, 100, 10)
        high_density_polygons = voronoi_grid[voronoi_grid['trajectory_count'] > threshold].copy()
        
        weather_info = []
        for _, row in high_density_polygons.iterrows():
            centroid = row.geometry.centroid
            weather = get_weather_data(centroid.y, centroid.x, api_key)
            weather_info.append(weather)
        
        high_density_polygons["weather"] = weather_info

        if len(high_density_polygons) >= 3:
            # Prepare features
            valid_polygons = []
            features = []
            
            for _, row in high_density_polygons.iterrows():
                if row["weather"]:
                    centroid = row.geometry.centroid
                    # Spatial features
                    lat = centroid.y
                    lon = centroid.x
                    # Weather features
                    temp = row["weather"]["temperature_celsius"]
                    humidity = row["weather"]["humidity"]
                    
                    valid_polygons.append(row)
                    features.append([lat, lon, temp, humidity])
            
            if len(features) < 2:
                st.warning("Not enough valid weather data points for clustering")
                return

            # Convert to numpy array
            features = np.array(features)
            
            # Normalize features
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            
            # Create weights for features
            st.subheader("Feature Weights")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                lat_weight = st.slider("Latitude Weight", 0.1, 2.0, 1.0)
            with col2:
                lon_weight = st.slider("Longitude Weight", 0.1, 2.0, 1.0)
            with col3:
                temp_weight = st.slider("Temperature Weight", 0.1, 2.0, 0.5)
            with col4:
                humid_weight = st.slider("Humidity Weight", 0.1, 2.0, 0.5)

            # Apply weights
            weighted_features = features_scaled * np.array([lat_weight, lon_weight, temp_weight, humid_weight])
            
            # DBSCAN parameters
            st.subheader("Clustering Parameters")
            eps = st.slider("EPS (scaled units)", 0.1, 5.0, 1.0)
            min_samples = st.slider("Minimum samples", 1, 10, 2)
            
            # Run DBSCAN
            db = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean').fit(weighted_features)
            clusters = db.labels_
            
            # Update polygons with clusters
            high_density_polygons = high_density_polygons.iloc[[i for i, row in enumerate(high_density_polygons.iterrows()) if row[1]["weather"]]]
            high_density_polygons['cluster'] = clusters

            # Visualization
            map_center = [features[:,0].mean(), features[:,1].mean()]
            m = folium.Map(location=map_center, zoom_start=12)
            
            # Add polygons
            folium.GeoJson(
                high_density_polygons.to_json(),
                style_function=lambda feature: {
                    'fillColor': f'hsl({abs(feature["properties"]["cluster"])*60}, 70%, 50%)',
                    'color': 'black',
                    'weight': 1,
                    'fillOpacity': 0.6,
                },
                tooltip=folium.GeoJsonTooltip(fields=['cluster', 'trajectory_count']),
            ).add_to(m)

            # Add weather markers
            for idx, row in high_density_polygons.iterrows():
                centroid = row.geometry.centroid
                weather = row["weather"]
                popup_content = f"""
                    Temperature: {weather['temperature_celsius']:.1f}°C<br>
                    Humidity: {weather['humidity']}%<br>
                    Conditions: {weather['weather_description']}
                """
                folium.CircleMarker(
                    location=[centroid.y, centroid.x],
                    radius=5,
                    color='#333333',
                    fill_color=f'hsl({abs(row["cluster"])*60}, 70%, 50%)',
                    fill_opacity=0.7,
                    popup=folium.Popup(popup_content, max_width=250),
                ).add_to(m)

            st.subheader("Weather-Aware Clusters")
            st_folium(m, width=900, height=600)

            # Metrics
            unique_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
            if unique_clusters > 1:
                score = silhouette_score(weighted_features, clusters)
                st.metric("Silhouette Score", f"{score:.3f}")
                st.caption(f"Identified {unique_clusters} distinct weather-spatial patterns")
            else:
                st.warning("Not enough clusters for meaningful evaluation")

        else:
            st.warning("Not enough high-density areas for clustering")

if __name__ == "__main__":
    main()
