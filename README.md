# Customer-Segmentation-Using-Unsupervised-Learning

This project implements **customer segmentation** on the Mall Customers dataset using **K-Means clustering**, includes **EDA**, visualizes clusters with **PCA**, and proposes **marketing strategies** for each segment.

## What this solution covers
- Exploratory Data Analysis (EDA)
- K-Means customer segmentation
- Cluster selection support (Elbow + Silhouette)
- PCA-based 2D cluster visualization
- Segment-wise marketing strategy suggestions

## Setup
```bash
pip install -r requirements.txt
```

## Run
Place the Mall Customers CSV in one of these default paths:
- `Mall_Customers.csv`
- `mall_customers.csv`
- `data/Mall_Customers.csv`
- `data/mall_customers.csv`

Or pass it directly:

```bash
python customer_segmentation.py --dataset /absolute/path/to/Mall_Customers.csv
```

Optional: force a fixed number of clusters

```bash
python customer_segmentation.py --dataset /absolute/path/to/Mall_Customers.csv --clusters 5
```

## Outputs
Generated in `outputs/`:
- `k_selection.png` (Elbow and Silhouette curves)
- `customer_clusters_pca.png` (PCA cluster visualization)
- `customers_with_clusters.csv` (dataset + cluster labels)
- `segment_marketing_strategies.csv` (cluster profile + strategy)
