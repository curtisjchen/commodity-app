# Sustainability Data Hub

A centralized intelligence dashboard designed to aggregate and visualize supply chain sustainability metrics. This tool simplifies the complex landscape of spend, emissions, vendor surveys, and supply chain maps into a single, interactive interface.

## 📌 Project Overview
The primary goal of this application is to provide a "one-stop shop" for sustainability analysts. By consolidating data sources, the app enables users to monitor sustainability related characteristics of items.

### Key Data Pillars:
* **Spend Analytics:** Global receiving data from FY20-25.
* **Emissions Tracking:** Scope 3 Category 1 emissions of items.
* **Survey Integration:** Commodity survey data for available items.
* **Supply Chain Mapping:** Supply Chain Maps of available items.
* **Commodity Prediction:** A machine learning classifier that suggests commodity labels for unlabeled items based on item descriptions.



## 🛠️ Technology Stack
* **UI/Frontend:** [Streamlit](https://streamlit.io/)
* **Database:** [DuckDB](https://duckdb.io/) (High-performance local SQL engine)
* **Data Format:** Parquet or CSVs, or connections to BigQuery if allowed
* **ML Framework:** PyTorch, Hugging Face Transformers
* **Environment:** Python 3.12+ managed by `uv`

## 🚀 Quick Start

1. **Clone the Repo:**
   ```powershell
   git clone [https://github.com/YOUR_USERNAME/commodity-app.git](https://github.com/YOUR_USERNAME/commodity-app.git)
   cd commodity-app