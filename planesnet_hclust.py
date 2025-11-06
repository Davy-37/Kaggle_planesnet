#!/usr/bin/env python3
"""
Hierarchical Clustering (Agglomerative) on PlanesNet (JSON)

Features:
- Loads planesnet.json (1200-dim pixel vectors scaled to [0,1])
- Optional PCA for dimensionality reduction before clustering
- Agglomerative clustering (linkage: ward/average/complete, metric: euclidean/cosine*)
- Metrics vs. ground-truth labels (unsupervised eval): ARI, NMI, Homogeneity, Completeness
- Visualizations: dendrogram (on a sample), PCA 2D scatter colored by clusters,
  silhouette scores, cluster size bar chart, and average image per cluster (20x20 RGB)
- Saves cluster assignments to CSV

(*) Note: 'ward' linkage requires euclidean distances.

Usage examples
--------------
# Basic: 2 clusters (plane/no-plane) with PCA=50
python planesnet_hclust.py --json Data/planesnet/planesnet.json --n-clusters 2 --pca 50

# Try cosine metric with average linkage (requires sklearn >=1.2)
python planesnet_hclust.py --json Data/planesnet/planesnet.json --n-clusters 2 --pca 100 --linkage average --metric cosine

# Produce a dendrogram on a 2000-sample subset (fast) and cluster full set
python planesnet_hclust.py --json Data/planesnet/planesnet.json --n-clusters 2 --pca 50 --dendro-sample 2000

Outputs (./runs_hclust)
-----------------------
- clusters.csv                  : id, cluster, true_label
- metrics.json                  : ARI/NMI/Homogeneity/Completeness/Silhouette
- pca_scatter_clusters.png      : PCA(2D) colored by cluster
- silhouette.png                : Silhouette plot
- cluster_sizes.png             : Bar chart of cluster sizes
- dendrogram.png                : Dendrogram for the sampled subset (if --dendro-sample>0)
- cluster_means.png             : Grid of average 20x20 RGB images per cluster
"""

import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, homogeneity_score, completeness_score, silhouette_samples, silhouette_score
from sklearn.neighbors import NearestCentroid

from scipy.cluster.hierarchy import linkage, dendrogram


plt.rcParams['figure.dpi'] = 120


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def load_planesnet_json(path):
    with open(path, 'r') as f:
        d = json.load(f)
    X = np.asarray(d['data'], dtype=np.float32) / 255.0
    y = np.asarray(d['labels'], dtype=np.int64)
    return X, y


def preprocess_X(X, pca_components=0):
    scaler = StandardScaler(with_mean=True, with_std=True)
    Xs = scaler.fit_transform(X)
    pca = None
    if pca_components and pca_components > 0:
        pca = PCA(n_components=pca_components, random_state=42)
        Xr = pca.fit_transform(Xs)
    else:
        Xr = Xs
    return Xr, scaler, pca


def run_agglomerative(Xr, n_clusters=2, linkage_method='ward', metric='euclidean'):
    # sklearn's AgglomerativeClustering uses 'metric' for distances (>=1.2). For ward, metric must be euclidean.
    if linkage_method == 'ward' and metric != 'euclidean':
        print("[warn] 'ward' linkage forces 'euclidean' metric. Overriding metric to euclidean.")
        metric = 'euclidean'
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_method, metric=metric)
    labels = model.fit_predict(Xr)
    return model, labels


def plot_pca_scatter(Xr, labels, out_path):
    # If Xr already 2D (PCA used), use it directly; else, reduce to 2D for plotting only
    if Xr.shape[1] > 2:
        pca_plot = PCA(n_components=2, random_state=42)
        Z = pca_plot.fit_transform(Xr)
    else:
        Z = Xr
    fig = plt.figure()
    for c in np.unique(labels):
        idx = labels == c
        plt.scatter(Z[idx,0], Z[idx,1], s=5, alpha=0.7, label=f'cluster {c}')
    plt.legend()
    plt.title('PCA scatter by cluster')
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_cluster_sizes(labels, out_path):
    uniq, counts = np.unique(labels, return_counts=True)
    fig = plt.figure()
    plt.bar([str(u) for u in uniq], counts)
    plt.title('Cluster sizes')
    plt.xlabel('Cluster')
    plt.ylabel('Count')
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_silhouette(Xr, labels, out_path):
    try:
        sil_vals = silhouette_samples(Xr, labels, metric='euclidean')
        sil_avg = silhouette_score(Xr, labels, metric='euclidean')
    except Exception as e:
        print('[warn] Silhouette failed:', e)
        return None

    fig = plt.figure(figsize=(6,4))
    y_lower = 10
    for c in sorted(np.unique(labels)):
        c_sil = sil_vals[labels==c]
        c_sil.sort()
        size_c = c_sil.shape[0]
        y_upper = y_lower + size_c
        plt.fill_betweenx(np.arange(y_lower, y_upper), 0, c_sil)
        plt.text(-0.05, y_lower + 0.5*size_c, str(c))
        y_lower = y_upper + 10
    plt.axvline(sil_avg, linestyle='--')
    plt.xlabel('Silhouette coefficient')
    plt.ylabel('Samples (stacked by cluster)')
    plt.title(f'Silhouette plot (avg={sil_avg:.3f})')
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return float(sil_avg)


def make_cluster_means_png(X_raw01, labels, out_path):
    """Compute average 20x20 RGB image per cluster and save as a grid."""
    H=W=20
    clusters = np.unique(labels)
    imgs = []
    for c in clusters:
        Xc = X_raw01[labels==c]
        if Xc.size == 0:
            avg = np.zeros((H,W,3), dtype=np.float32)
        else:
            # Split channels: first 400 R, next 400 G, last 400 B
            R = Xc[:, :400].mean(axis=0).reshape(H,W)
            G = Xc[:, 400:800].mean(axis=0).reshape(H,W)
            B = Xc[:, 800:1200].mean(axis=0).reshape(H,W)
            avg = np.stack([R,G,B], axis=2)
        imgs.append((c, np.clip(avg, 0, 1)))

    # Build grid
    cols = min(8, len(clusters))
    rows = int(np.ceil(len(clusters)/cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols*2, rows*2))
    if rows==1 and cols==1:
        axes = np.array([[axes]])
    axes = axes.reshape(rows, cols)
    for i,(c,img) in enumerate(imgs):
        r = i//cols; k = i%cols
        ax = axes[r,k]
        ax.imshow(img)
        ax.set_title(f'cluster {c}')
        ax.axis('off')
    # Hide empty
    for j in range(len(imgs), rows*cols):
        r = j//cols; k = j%cols
        axes[r,k].axis('off')
    plt.suptitle('Average image per cluster')
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def make_dendrogram(Xr, out_path, method='ward', metric='euclidean', sample=2000, seed=42):
    if sample and Xr.shape[0] > sample:
        rng = np.random.RandomState(seed)
        idx = rng.choice(np.arange(Xr.shape[0]), size=sample, replace=False)
        Xd = Xr[idx]
    else:
        Xd = Xr
    try:
        Z = linkage(Xd, method=method, metric=metric)
        fig = plt.figure(figsize=(10, 5))
        dendrogram(Z, no_labels=True, color_threshold=None)
        plt.title(f'Dendrogram (sample n={Xd.shape[0]}, {method}/{metric})')
        plt.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)
    except Exception as e:
        print('[warn] Dendrogram failed:', e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--json', required=True)
    ap.add_argument('--n-clusters', type=int, default=2)
    ap.add_argument('--pca', type=int, default=50)
    ap.add_argument('--linkage', type=str, default='ward', choices=['ward','average','complete','single'])
    ap.add_argument('--metric', type=str, default='euclidean', choices=['euclidean','cosine'])
    ap.add_argument('--dendro-sample', type=int, default=2000, help='samples for dendrogram (0=disable)')
    args = ap.parse_args()

    out_dir = 'runs_hclust'
    ensure_dir(out_dir)

    # 1) Load & preprocess
    X_raw01, y = load_planesnet_json(args.json)
    Xr, scaler, pca = preprocess_X(X_raw01, pca_components=args.pca)

    # 2) Cluster
    model, clusters = run_agglomerative(Xr, n_clusters=args.n_clusters, linkage_method=args.linkage, metric=args.metric)

    # 3) Metrics vs. ground truth (unsupervised eval)
    ari = adjusted_rand_score(y, clusters)
    nmi = normalized_mutual_info_score(y, clusters)
    h = homogeneity_score(y, clusters)
    c = completeness_score(y, clusters)

    # 4) Visualizations
    plot_pca_scatter(Xr, clusters, os.path.join(out_dir, 'pca_scatter_clusters.png'))
    plot_cluster_sizes(clusters, os.path.join(out_dir, 'cluster_sizes.png'))
    sil_avg = plot_silhouette(Xr, clusters, os.path.join(out_dir, 'silhouette.png'))
    if args.dendro_sample and args.dendro_sample>0:
        make_dendrogram(Xr, os.path.join(out_dir, 'dendrogram.png'), method=args.linkage, metric=args.metric, sample=args.dendro_sample)
    make_cluster_means_png(X_raw01, clusters, os.path.join(out_dir, 'cluster_means.png'))

    # 5) Save cluster assignments & metrics
    df = pd.DataFrame({
        'id': np.arange(Xr.shape[0]),
        'cluster': clusters,
        'true_label': y
    })
    df.to_csv(os.path.join(out_dir, 'clusters.csv'), index=False)

    import json as _json
    with open(os.path.join(out_dir, 'metrics.json'), 'w') as f:
        _json.dump({
            'n_clusters': int(args.n_clusters),
            'pca_components': int(args.pca),
            'linkage': args.linkage,
            'metric': args.metric,
            'ARI': float(ari),
            'NMI': float(nmi),
            'homogeneity': float(h),
            'completeness': float(c),
            'silhouette': None if sil_avg is None else float(sil_avg)
        }, f, indent=2)

    print('Saved artifacts to ./runs_hclust')
    print(f'ARI={ari:.4f}  NMI={nmi:.4f}  Homo={h:.4f}  Comp={c:.4f}  Sil={sil_avg if sil_avg is not None else float("nan"):.4f}')


if __name__ == '__main__':
    main()
