# CS313.P21---Data Mining
![UIT](https://img.shields.io/badge/from-UIT%20VNUHCM-blue?style=for-the-badge&link=https%3A%2F%2Fwww.uit.edu.vn%2F)

 <h2 align="center"> ANIMAL MIGRATION CLUSTERING </h2>

<p align="center">
  <img src="https://en.uit.edu.vn/sites/vi/files/banner_en.png" alt="Alt text">
</p>

## Contributors

| Contributor           | Email                         |
|-----------------------|-------------------------------|
| Hoàng Công Chiến       | [22520155@gm.uit.edu.vn](mailto:22520155@gm.uit.edu.vn) |
| Phan Thanh Đăng       | [22520193@gm.uit.edu.vn](mailto:22520193@gm.uit.edu.vn) |
| Trần Đình Khánh Đăng | [22520195@gm.uit.edu.vn](mailto:22520195@gm.uit.edu.vn) |
| Dương Đình Phương Dao | [22520202@gm.uit.edu.vn](mailto:22520202@gm.uit.edu.vn) |
| Trần Quang Đạt        | [22520236@gm.uit.edu.vn](mailto:22520236@gm.uit.edu.vn) |
| Nguyễn Hữu Đức        | [22520270@gm.uit.edu.vn](mailto:22520270@gm.uit.edu.vn) |

## Supervisors  
- **PhD. Võ Nguyễn Lê Duy**  
  Email: [duyvnl@uit.edu.vn](mailto:tiendv@uit.edu.vn)

---

<h1 align="center">Animal Migration Clustering</h1>

*Our project is an Animal Migration Clustering system, focusing on analyzing the past movement data of Western Palearctic greater white-fronted geese to identify key areas for conservation. Furthermore, we will explore applying advanced data generation algorithms such as GANs and VAEs to augment and enrich the existing dataset, thereby enhancing the effectiveness of our clustering model.*

## Dataset

- *The Western Palearctic greater white-fronted geese dataset*: Contains migration data for greater white-fronted geese.

## Features

- **Migration Simulation**: Generates realistic migration paths based on historical data.
- **Migration Clustering**: Implements various clustering algorithms (e.g., K-Means, DBSCAN) to group similar migration patterns.
- **Cluster Visualization**: Displays identified migration clusters on an interactive map, highlighting distinct movement behaviors.
- **Interactive Visualization**: Displays migration routes and patterns on an interactive map.
- **Data Insights**: Provides analytical insights into migration behaviors and environmental factors.

## Installation

1. **Clone the repository**

```bash
git clone https://github.com/PTD504/Data-Mining-UIT.git
cd Data-Mining-UIT
```


2. **Create a virtual environment**
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows use "venv\Scripts\activate"
```


3. **Install dependencies**
```bash
pip install -r requirements.txt
```


4. **Run the application**
```bash
streamlit run app/app.py
```

## Future Work

- **Expand Dataset**: Incorporate additional species and migration datasets for broader analysis.
- **Advanced Modeling**: Implement deep learning or agent-based models for more accurate simulations.
- **Enhanced Visualizations**: Integrate real-time weather data or 3D visualizations for a richer user experience.

---

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
