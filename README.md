# Customer-Segmentation-Using-Unsupervised-Learning

This project implements **customer segmentation** on the Mall Customers dataset using **K-Means clustering**, includes **EDA**, visualizes clusters with **PCA/t-SNE**, and proposes **marketing strategies** for each segment.

## What this solution covers
- Exploratory Data Analysis (EDA) with data quality checks, outlier reporting, and optional extended visual EDA
- K-Means customer segmentation
- Cluster selection support (Elbow + Silhouette)
- PCA and t-SNE cluster visualization comparison
- Advanced cluster validation metrics (Silhouette, Davies-Bouldin, Calinski-Harabasz)
- Segment-wise marketing strategy suggestions and enriched segment profiles

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

Optional flags:

```bash
python customer_segmentation.py --dataset /absolute/path/to/Mall_Customers.csv --clusters 5 --verbose --extended-eda
```

## Outputs
Generated in `outputs/`:
- `k_selection.png` (Elbow and Silhouette curves, when `--clusters` is not provided)
- `customer_clusters_pca.png` (PCA cluster visualization)
- `cluster_comparison_pca_tsne.png` (side-by-side PCA vs t-SNE comparison)
- `silhouette_analysis.png` (per-cluster silhouette visualization)
- `cluster_size_distribution.png` (cluster size distribution)
- `eda_summary_report.txt` (data quality, missing values, outliers, numeric summary)
- `eda_distributions.png` (feature distributions, with `--extended-eda`)
- `eda_correlation_heatmap.png` (feature correlation heatmap, with `--extended-eda`)
- `cluster_metrics_report.csv` (Silhouette, Davies-Bouldin, Calinski-Harabasz metrics)
- `cluster_quality_report.txt` (comprehensive quality summary)
- `customers_with_clusters.csv` (dataset + cluster labels + PCA/t-SNE coordinates)
- `segment_marketing_strategies.csv` (segment-level strategy summary)
- `customer_segments_profile.csv` (detailed segment profiles with counts, ranges, insights, revenue potential)
