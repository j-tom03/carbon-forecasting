from diagrams import Diagram, Cluster, Edge
from diagrams.onprem.analytics import Spark
from diagrams.onprem.database import PostgreSQL
from diagrams.onprem.compute import Server
from diagrams.programming.language import Python
from diagrams.custom import Custom
from pathlib import Path
import json

# Paths
META_PATH = Path("data/processed/dataset_v1_meta.json")
OUTPUT_PATH = Path("docs/data_lineage")

def draw_lineage():
    # Load real metadata to annotate the diagram
    try:
        with open(META_PATH, "r") as f:
            meta = json.load(f)
            row_count = meta.get("rows", "Unknown")
            start_date = meta.get("start_date", "").split("T")[0]
            end_date = meta.get("end_date", "").split("T")[0]
            data_label = f"{row_count} rows\n({start_date} to {end_date})"
    except FileNotFoundError:
        data_label = "Metadata Missing"

    # specific graphviz attributes for a cleaner look
    graph_attr = {
        "fontsize": "20",
        "bgcolor": "transparent"
    }

    with Diagram("Carbon Forecasting Data Pipeline", show=False, filename=str(OUTPUT_PATH), graph_attr=graph_attr, direction="LR"):
        
        with Cluster("Ingestion (Bronze)"):
            carbon_api = Server("Carbon Intensity API")
            weather_api = Server("Open-Meteo API")
            
            raw_c = Python("fetch_carbon.py")
            raw_w = Python("fetch_weather.py")
            
            carbon_api >> Edge(label="Daily JSON") >> raw_c
            weather_api >> Edge(label="Monthly JSON") >> raw_w

        with Cluster("Processing (Silver)"):
            norm = Python("normalise.py")
            
            [raw_c, raw_w] >> Edge(label="Raw JSONs") >> norm
            
            parquet_c = PostgreSQL("Carbon Parquet")
            parquet_w = PostgreSQL("Weather Parquet")
            
            norm >> parquet_c
            norm >> parquet_w

        with Cluster("Feature Engineering (Gold)"):
            merger = Python("merge_sources.py")
            
            [parquet_c, parquet_w] >> Edge(label="Left Join\n(Forward Fill)") >> merger
            
            final_ds = PostgreSQL(f"Final Dataset\n{data_label}")
            
            merger >> final_ds

if __name__ == "__main__":
    draw_lineage()
    print(f"Diagram generated at {OUTPUT_PATH}.png")