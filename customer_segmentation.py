from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_samples,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler


REQUIRED_COLUMNS = ["Annual Income (k$)", "Spending Score (1-100)"]
RANDOM_STATE = 42
TSNE_PERPLEXITY_CANDIDATES = (5, 10, 20, 30, 40)
TSNE_MIN_PERPLEXITY = 2
TSNE_MAX_PERPLEXITY = 40
MAX_HISTOGRAM_COLUMNS = 3
SILHOUETTE_LABEL_X_OFFSET = -0.08
LOGGER = logging.getLogger(__name__)


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="[%(levelname)s] %(message)s")


def resolve_dataset_path(dataset_path: str | None) -> Path:
    if dataset_path:
        path = Path(dataset_path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")
        return path

    candidates = [
        Path("Mall_Customers.csv"),
        Path("mall_customers.csv"),
        Path("data/Mall_Customers.csv"),
        Path("data/mall_customers.csv"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "Dataset path not provided and no default Mall Customers CSV file was found. "
        "Pass --dataset <path-to-csv>."
    )


def run_eda(df: pd.DataFrame, output_dir: Path, extended_eda: bool = False) -> None:
    LOGGER.info("Running EDA")
    numeric_df = df.select_dtypes(include="number")
    if numeric_df.empty:
        raise ValueError("Dataset must contain numeric columns for EDA and clustering")

    missing_values = df.isna().sum()
    duplicate_rows = int(df.duplicated().sum())
    outlier_counts: dict[str, int] = {}

    for column in numeric_df.columns:
        q1 = numeric_df[column].quantile(0.25)
        q3 = numeric_df[column].quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            outlier_counts[column] = 0
            continue
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outlier_counts[column] = int(((numeric_df[column] < lower) | (numeric_df[column] > upper)).sum())

    quality_metrics = {
        "rows": len(df),
        "columns": len(df.columns),
        "missing_cells": int(missing_values.sum()),
        "duplicate_rows": duplicate_rows,
        "numeric_columns": len(numeric_df.columns),
    }

    report_lines = [
        "=== EDA SUMMARY REPORT ===",
        f"Rows: {quality_metrics['rows']}",
        f"Columns: {quality_metrics['columns']}",
        f"Numeric columns: {quality_metrics['numeric_columns']}",
        f"Missing cells: {quality_metrics['missing_cells']}",
        f"Duplicate rows: {quality_metrics['duplicate_rows']}",
        "",
        "Missing values per column:",
        missing_values.to_string(),
        "",
        "Outlier counts (IQR method):",
        pd.Series(outlier_counts).to_string(),
        "",
        "Numeric summary:",
        numeric_df.describe().to_string(),
    ]
    (output_dir / "eda_summary_report.txt").write_text("\n".join(report_lines), encoding="utf-8")

    print("\n=== EDA SUMMARY ===")
    print(f"Rows: {len(df)}, Columns: {len(df.columns)}")
    print("\nMissing values per column:")
    print(missing_values)
    print("\nNumeric summary:")
    print(numeric_df.describe())
    print("\nOutlier counts (IQR method):")
    print(pd.Series(outlier_counts))

    if not extended_eda:
        return

    LOGGER.info("Generating extended EDA visualizations")
    plt.figure(figsize=(10, 8))
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", square=True)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(output_dir / "eda_correlation_heatmap.png", dpi=150)
    plt.close()

    n_cols = min(MAX_HISTOGRAM_COLUMNS, len(numeric_df.columns))
    n_rows = int(np.ceil(len(numeric_df.columns) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes_arr = np.atleast_1d(axes).ravel()
    for index, column in enumerate(numeric_df.columns):
        sns.histplot(numeric_df[column], kde=True, ax=axes_arr[index], color="steelblue")
        axes_arr[index].set_title(f"Distribution: {column}")
    for index in range(len(numeric_df.columns), len(axes_arr)):
        axes_arr[index].axis("off")
    plt.tight_layout()
    plt.savefig(output_dir / "eda_distributions.png", dpi=150)
    plt.close(fig)


def find_best_k(scaled_features: np.ndarray, output_dir: Path) -> int:
    LOGGER.info("Selecting optimal cluster count")
    k_values = list(range(2, 11))
    inertias = []
    silhouettes = []

    for k in k_values:
        model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = model.fit_predict(scaled_features)
        inertias.append(model.inertia_)
        silhouettes.append(silhouette_score(scaled_features, labels))

    scores = pd.DataFrame({"k": k_values, "inertia": inertias, "silhouette": silhouettes})
    best_k = int(scores.loc[scores["silhouette"].idxmax(), "k"])

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(k_values, inertias, marker="o")
    plt.title("Elbow Method")
    plt.xlabel("k")
    plt.ylabel("Inertia")

    plt.subplot(1, 2, 2)
    plt.plot(k_values, silhouettes, marker="o", color="green")
    plt.title("Silhouette Score")
    plt.xlabel("k")
    plt.ylabel("Score")

    plt.tight_layout()
    plt.savefig(output_dir / "k_selection.png", dpi=150)
    plt.close()

    return best_k


def optimize_tsne_projection(scaled_features: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, int]:
    sample_count = len(scaled_features)
    if sample_count <= TSNE_MIN_PERPLEXITY:
        raise ValueError(
            f"t-SNE requires more than {TSNE_MIN_PERPLEXITY} rows to compute a valid perplexity"
        )

    valid_perplexities = [p for p in TSNE_PERPLEXITY_CANDIDATES if TSNE_MIN_PERPLEXITY < p < sample_count]

    if not valid_perplexities:
        fallback = min(TSNE_MAX_PERPLEXITY, sample_count - 1)
        tsne = TSNE(n_components=2, random_state=RANDOM_STATE, perplexity=fallback, init="pca")
        return tsne.fit_transform(scaled_features), fallback

    best_score = -np.inf
    best_embedding: np.ndarray | None = None
    best_perplexity = valid_perplexities[0]

    for perplexity in valid_perplexities:
        tsne = TSNE(n_components=2, random_state=RANDOM_STATE, perplexity=perplexity, init="pca")
        embedding = tsne.fit_transform(scaled_features)
        score = silhouette_score(embedding, labels)
        if score > best_score:
            best_score = score
            best_embedding = embedding
            best_perplexity = perplexity

    if best_embedding is None:
        raise ValueError("t-SNE optimization failed to produce an embedding")

    return best_embedding, best_perplexity


def create_cluster_visualizations(
    df: pd.DataFrame,
    pca_components: np.ndarray,
    tsne_components: np.ndarray,
    output_dir: Path,
) -> None:
    df["PCA1"] = pca_components[:, 0]
    df["PCA2"] = pca_components[:, 1]
    df["TSNE1"] = tsne_components[:, 0]
    df["TSNE2"] = tsne_components[:, 1]

    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x="PCA1", y="PCA2", hue="Cluster", palette="tab10", s=60)
    plt.title("Customer Segments (K-Means + PCA)")
    plt.tight_layout()
    plt.savefig(output_dir / "customer_clusters_pca.png", dpi=150)
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sns.scatterplot(data=df, x="PCA1", y="PCA2", hue="Cluster", palette="tab10", s=55, ax=axes[0], legend=False)
    axes[0].set_title("PCA Projection")
    sns.scatterplot(data=df, x="TSNE1", y="TSNE2", hue="Cluster", palette="tab10", s=55, ax=axes[1])
    axes[1].set_title("t-SNE Projection")
    plt.tight_layout()
    plt.savefig(output_dir / "cluster_comparison_pca_tsne.png", dpi=150)
    plt.close(fig)


def save_cluster_metrics(scaled_features: np.ndarray, labels: np.ndarray, output_dir: Path) -> pd.DataFrame:
    silhouette_avg = silhouette_score(scaled_features, labels)
    davies_bouldin = davies_bouldin_score(scaled_features, labels)
    calinski_harabasz = calinski_harabasz_score(scaled_features, labels)

    metrics_df = pd.DataFrame(
        [
            {
                "silhouette_score": silhouette_avg,
                "davies_bouldin_index": davies_bouldin,
                "calinski_harabasz_index": calinski_harabasz,
            }
        ]
    )
    metrics_df.to_csv(output_dir / "cluster_metrics_report.csv", index=False)
    return metrics_df


def save_silhouette_analysis(scaled_features: np.ndarray, labels: np.ndarray, output_dir: Path) -> pd.DataFrame:
    sample_silhouettes = silhouette_samples(scaled_features, labels)
    clusters = np.unique(labels)

    fig, ax = plt.subplots(figsize=(9, 6))
    y_lower = 10
    silhouette_rows = []

    for cluster in clusters:
        values = sample_silhouettes[labels == cluster]
        values.sort()
        size = len(values)
        y_upper = y_lower + size
        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, values, alpha=0.7)
        ax.text(SILHOUETTE_LABEL_X_OFFSET, y_lower + 0.5 * size, str(cluster))
        y_lower = y_upper + 10
        silhouette_rows.append(
            {
                "Cluster": int(cluster),
                "cluster_silhouette_mean": float(values.mean()),
                "cluster_size": int(size),
            }
        )

    ax.axvline(x=float(sample_silhouettes.mean()), color="red", linestyle="--")
    ax.set_title("Silhouette Analysis by Cluster")
    ax.set_xlabel("Silhouette Coefficient")
    ax.set_ylabel("Cluster")
    plt.tight_layout()
    plt.savefig(output_dir / "silhouette_analysis.png", dpi=150)
    plt.close(fig)

    return pd.DataFrame(silhouette_rows)


def save_cluster_size_distribution(labels: np.ndarray, output_dir: Path) -> pd.DataFrame:
    counts = pd.Series(labels).value_counts().sort_index()
    distribution = counts.rename_axis("Cluster").reset_index(name="Customer Count")
    distribution["Cluster"] = distribution["Cluster"].astype(int)

    plt.figure(figsize=(8, 5))
    colors = sns.color_palette("viridis", n_colors=len(distribution))
    plt.bar(distribution["Cluster"], distribution["Customer Count"], color=colors)
    plt.xticks(distribution["Cluster"])
    plt.title("Cluster Size Distribution")
    plt.xlabel("Cluster")
    plt.ylabel("Customer Count")
    plt.tight_layout()
    plt.savefig(output_dir / "cluster_size_distribution.png", dpi=150)
    plt.close()

    return distribution


def build_marketing_strategy(profile_row: pd.Series, income_median: float, spending_median: float) -> tuple[str, str]:
    income = float(profile_row["Income Mean (k$)"])
    spending = float(profile_row["Spending Mean"])

    if income >= income_median and spending >= spending_median:
        return (
            "Premium loyalty offers, early access launches, VIP experiences",
            "High-value customers with strong spend potential; prioritize retention and exclusivity campaigns.",
        )
    if income >= income_median and spending < spending_median:
        return (
            "Targeted upsell bundles, personalized recommendations, exclusive trials",
            "Strong earning power but lower spend engagement; use conversion journeys and product education.",
        )
    if income < income_median and spending >= spending_median:
        return (
            "Flash deals, referral rewards, social-proof campaigns",
            "Price-sensitive but highly active segment; leverage promotional urgency and referral loops.",
        )
    return (
        "Value-focused discounts, retention coupons, basic membership perks",
        "Lower income and spending behavior; focus on affordability and consistent reactivation touchpoints.",
    )


def estimate_revenue_potential(customer_count: int, income_mean: float, spending_mean: float) -> float:
    """Estimate a relative segment potential score (not currency) from count × income × spending percentage."""
    return float(customer_count * income_mean * spending_mean / 100)


def create_segment_profiles(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    profile = (
        df.groupby("Cluster", as_index=False)
        .agg(
            **{
                "Customer Count": ("Cluster", "size"),
                "Income Mean (k$)": ("Annual Income (k$)", "mean"),
                "Income Min (k$)": ("Annual Income (k$)", "min"),
                "Income Max (k$)": ("Annual Income (k$)", "max"),
                "Spending Mean": ("Spending Score (1-100)", "mean"),
                "Spending Min": ("Spending Score (1-100)", "min"),
                "Spending Max": ("Spending Score (1-100)", "max"),
            }
        )
        .sort_values(by=["Income Mean (k$)", "Spending Mean"])
        .reset_index(drop=True)
    )

    income_median = profile["Income Mean (k$)"].median()
    spending_median = profile["Spending Mean"].median()

    strategies = profile.apply(
        lambda row: build_marketing_strategy(row, income_median=income_median, spending_median=spending_median),
        axis=1,
    )
    profile["Marketing Strategy"] = [item[0] for item in strategies]
    profile["Actionable Insight"] = [item[1] for item in strategies]
    profile["Segment Profile"] = profile.apply(
        lambda row: (
            f"Income ${row['Income Min (k$)']:.1f}k-${row['Income Max (k$)']:.1f}k, "
            f"Spending {row['Spending Min']:.1f}-{row['Spending Max']:.1f}"
        ),
        axis=1,
    )
    profile["Estimated Revenue Potential"] = profile.apply(
        lambda row: estimate_revenue_potential(
            customer_count=row["Customer Count"],
            income_mean=row["Income Mean (k$)"],
            spending_mean=row["Spending Mean"],
        ),
        axis=1,
    ).round(2)

    profile.to_csv(output_dir / "customer_segments_profile.csv", index=False)

    strategy_summary = profile[
        ["Cluster", "Income Mean (k$)", "Spending Mean", "Customer Count", "Marketing Strategy"]
    ]
    strategy_summary.to_csv(output_dir / "segment_marketing_strategies.csv", index=False)

    return profile


def save_cluster_quality_report(
    metrics_df: pd.DataFrame,
    silhouette_by_cluster: pd.DataFrame,
    cluster_distribution: pd.DataFrame,
    selected_k: int,
    selected_perplexity: int,
    output_dir: Path,
) -> None:
    report_lines = [
        "=== CLUSTER QUALITY REPORT ===",
        f"Selected clusters (k): {selected_k}",
        f"Selected t-SNE perplexity: {selected_perplexity}",
        "",
        "Overall metrics:",
        metrics_df.to_string(index=False),
        "",
        "Per-cluster silhouette summary:",
        silhouette_by_cluster.to_string(index=False),
        "",
        "Cluster size distribution:",
        cluster_distribution.to_string(index=False),
    ]
    (output_dir / "cluster_quality_report.txt").write_text("\n".join(report_lines), encoding="utf-8")


def segment_customers(
    dataset: Path,
    clusters: int | None = None,
    verbose: bool = False,
    extended_eda: bool = False,
) -> None:
    setup_logging(verbose=verbose)
    LOGGER.info("Loading dataset from %s", dataset)
    df = pd.read_csv(dataset)

    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Dataset is missing required columns: {missing_cols}. "
            f"Found columns: {list(df.columns)}"
        )

    if clusters is not None and clusters < 2:
        raise ValueError("--clusters must be at least 2")
    if len(df) < 3:
        raise ValueError("Dataset must contain at least 3 rows for clustering and t-SNE")
    if clusters is not None and clusters >= len(df):
        raise ValueError("Number of clusters must be less than the number of data points")

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    run_eda(df, output_dir=output_dir, extended_eda=extended_eda)

    features = df[REQUIRED_COLUMNS].copy()
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    used_auto_k = clusters is None
    k = clusters if clusters is not None else find_best_k(scaled_features, output_dir=output_dir)

    LOGGER.info("Training K-Means with k=%d", k)
    model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    df["Cluster"] = model.fit_predict(scaled_features)

    LOGGER.info("Computing PCA and t-SNE projections")
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    pca_components = pca.fit_transform(scaled_features)
    tsne_components, best_perplexity = optimize_tsne_projection(scaled_features, df["Cluster"].to_numpy())
    create_cluster_visualizations(df, pca_components, tsne_components, output_dir=output_dir)

    LOGGER.info("Calculating cluster validation metrics")
    labels = df["Cluster"].to_numpy()
    metrics_df = save_cluster_metrics(scaled_features, labels, output_dir=output_dir)
    silhouette_by_cluster = save_silhouette_analysis(scaled_features, labels, output_dir=output_dir)
    cluster_distribution = save_cluster_size_distribution(labels, output_dir=output_dir)

    LOGGER.info("Building segment profiles and strategies")
    profile_df = create_segment_profiles(df, output_dir=output_dir)
    save_cluster_quality_report(
        metrics_df=metrics_df,
        silhouette_by_cluster=silhouette_by_cluster,
        cluster_distribution=cluster_distribution,
        selected_k=k,
        selected_perplexity=best_perplexity,
        output_dir=output_dir,
    )

    df.to_csv(output_dir / "customers_with_clusters.csv", index=False)

    print("\n=== SEGMENT SUMMARY ===")
    print(profile_df[["Cluster", "Customer Count", "Income Mean (k$)", "Spending Mean", "Marketing Strategy"]])
    print("\nSaved:")
    if used_auto_k:
        print("- outputs/k_selection.png")
    print("- outputs/customer_clusters_pca.png")
    print("- outputs/customers_with_clusters.csv")
    print("- outputs/segment_marketing_strategies.csv")
    print("- outputs/cluster_metrics_report.csv")
    print("- outputs/customer_segments_profile.csv")
    print("- outputs/cluster_quality_report.txt")
    print("- outputs/cluster_comparison_pca_tsne.png")
    print("- outputs/silhouette_analysis.png")
    print("- outputs/cluster_size_distribution.png")
    print("- outputs/eda_summary_report.txt")
    if extended_eda:
        print("- outputs/eda_distributions.png")
        print("- outputs/eda_correlation_heatmap.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Customer segmentation using K-Means clustering")
    parser.add_argument("--dataset", type=str, default=None, help="Path to Mall Customers CSV")
    parser.add_argument("--clusters", type=int, default=None, help="Force number of clusters")
    parser.add_argument("--verbose", action="store_true", help="Enable detailed logging")
    parser.add_argument(
        "--extended-eda",
        action="store_true",
        help="Generate extended EDA visualizations (distributions and correlation heatmap)",
    )
    args = parser.parse_args()

    try:
        dataset = resolve_dataset_path(args.dataset)
        segment_customers(
            dataset=dataset,
            clusters=args.clusters,
            verbose=args.verbose,
            extended_eda=args.extended_eda,
        )
    except Exception as exc:
        LOGGER.error("Segmentation failed: %s", exc)
        raise


if __name__ == "__main__":
    main()
