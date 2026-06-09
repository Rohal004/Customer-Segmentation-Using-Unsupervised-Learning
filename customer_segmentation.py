from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


REQUIRED_COLUMNS = ["Annual Income (k$)", "Spending Score (1-100)"]


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


def run_eda(df: pd.DataFrame) -> None:
    print("\n=== EDA SUMMARY ===")
    print(f"Rows: {len(df)}, Columns: {len(df.columns)}")
    print("\nMissing values per column:")
    print(df.isna().sum())
    print("\nNumeric summary:")
    print(df.describe(include="number"))



def find_best_k(scaled_features: pd.DataFrame) -> int:
    k_values = list(range(2, 11))
    inertias = []
    silhouettes = []

    for k in k_values:
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
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
    Path("outputs").mkdir(exist_ok=True)
    plt.savefig("outputs/k_selection.png", dpi=150)
    plt.close()

    return best_k



def build_marketing_strategy(segment_stats: pd.DataFrame) -> pd.DataFrame:
    income_median = segment_stats["Annual Income (k$)"].median()
    spending_median = segment_stats["Spending Score (1-100)"].median()

    strategies = []
    for _, row in segment_stats.iterrows():
        income = float(row["Annual Income (k$)"])
        spending = float(row["Spending Score (1-100)"])
        segment = int(row["Cluster"])

        if income >= income_median and spending >= spending_median:
            strategy = "Premium loyalty offers, early access launches, VIP experiences"
        elif income >= income_median and spending < spending_median:
            strategy = "Targeted upsell bundles, personalized recommendations, exclusive trials"
        elif income < income_median and spending >= spending_median:
            strategy = "Flash deals, referral rewards, social-proof campaigns"
        else:
            strategy = "Value-focused discounts, retention coupons, basic membership perks"

        strategies.append({"Cluster": segment, "Marketing Strategy": strategy})

    return pd.DataFrame(strategies)



def segment_customers(dataset: Path, clusters: int | None = None) -> None:
    df = pd.read_csv(dataset)

    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Dataset is missing required columns: {missing_cols}. "
            f"Found columns: {list(df.columns)}"
        )

    run_eda(df)

    features = df[REQUIRED_COLUMNS].copy()
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)

    used_auto_k = clusters is None
    k = clusters if clusters is not None else find_best_k(scaled)
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    df["Cluster"] = model.fit_predict(scaled)

    pca = PCA(n_components=2, random_state=42)
    components = pca.fit_transform(scaled)
    df["PCA1"] = components[:, 0]
    df["PCA2"] = components[:, 1]

    Path("outputs").mkdir(exist_ok=True)
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x="PCA1", y="PCA2", hue="Cluster", palette="tab10", s=60)
    plt.title("Customer Segments (K-Means + PCA)")
    plt.tight_layout()
    plt.savefig("outputs/customer_clusters_pca.png", dpi=150)
    plt.close()

    segment_stats = (
        df.groupby("Cluster", as_index=False)[REQUIRED_COLUMNS]
        .mean()
        .sort_values(by=["Annual Income (k$)", "Spending Score (1-100)"])
        .reset_index(drop=True)
    )
    strategy_df = build_marketing_strategy(segment_stats)

    segment_summary = segment_stats.merge(strategy_df, on="Cluster", how="left")
    df.to_csv("outputs/customers_with_clusters.csv", index=False)
    segment_summary.to_csv("outputs/segment_marketing_strategies.csv", index=False)

    print("\n=== SEGMENT SUMMARY ===")
    print(segment_summary)
    print("\nSaved:")
    if used_auto_k:
        print("- outputs/k_selection.png")
    print("- outputs/customer_clusters_pca.png")
    print("- outputs/customers_with_clusters.csv")
    print("- outputs/segment_marketing_strategies.csv")



def main() -> None:
    parser = argparse.ArgumentParser(description="Customer segmentation using K-Means clustering")
    parser.add_argument("--dataset", type=str, default=None, help="Path to Mall Customers CSV")
    parser.add_argument("--clusters", type=int, default=None, help="Force number of clusters")
    args = parser.parse_args()

    dataset = resolve_dataset_path(args.dataset)
    segment_customers(dataset=dataset, clusters=args.clusters)


if __name__ == "__main__":
    main()
