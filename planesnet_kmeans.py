#!/usr/bin/env python3
"""
K-Means clustering on PlanesNet (JSON)

Features
--------
- Loads planesnet.json (1200-dim pixel vectors scaled to [0,1])
- Optional PCA for dimensionality reduction before clustering
- KMeans with configurable n_clusters and init runs
- Unsupervised metrics vs. ground truth (optional): ARI, NMI, Homogeneity, Completeness
- Visualizations: PCA(2D) scatter by cluster, cluster size bars, silhouette plot,
  average image per cluster (20x20 RGB), and elbow curve (optional)
- Saves cluster assignments to CSV and model with joblib

Usage examples
--------------
# Basic: 2 clusters (plane/no-plane) with PCA=50
python planesnet_kmeans.py --json Data/planesnet/planesnet.json --n-clusters 2 --pca 50

# Run elbow curve for k in [2..10]
python planesnet_kmeans.py --json Data/planesnet/planesnet.json --pca 50 --elbow-max 10

# Save and reload model
python planesnet_kmeans.py --json Data/planesnet/planesnet.json --n-clusters 2 --pca 100 --save-model
python planesnet_kmeans.py --json Data/planesnet/planesnet.json --load-model runs_kmeans/kmeans_model.joblib
"""

import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, homogeneity_score, completeness_score, silhouette_samples, silhouette_score

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


def plot_pca_scatter(Xr, labels, out_path):
    if Xr.shape[1] > 2:
        Z = PCA(n_components=2, random_state=42).fit_transform(Xr)
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


def cluster_means_png(X_raw01, labels, out_path):
    H=W=20
    clusters = np.unique(labels)
    imgs = []
    for c in clusters:
        Xc = X_raw01[labels==c]
        if Xc.size == 0:
            avg = np.zeros((H,W,3), dtype=np.float32)
        else:
            R = Xc[:, :400].mean(axis=0).reshape(H,W)
            G = Xc[:, 400:800].mean(axis=0).reshape(H,W)
            B = Xc[:, 800:1200].mean(axis=0).reshape(H,W)
            avg = np.stack([R,G,B], axis=2)
        imgs.append((c, np.clip(avg, 0, 1)))
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
    for j in range(len(imgs), rows*cols):
        r = j//cols; k = j%cols
        axes[r,k].axis('off')
    plt.suptitle('Average image per cluster')
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def elbow_curve(Xr, kmin=2, kmax=10, out_path='elbow.png', n_init=10):
    inertias = []
    ks = list(range(kmin, kmax+1))
    for k in ks:
        km = KMeans(n_clusters=k, n_init=n_init, random_state=42)
        km.fit(Xr)
        inertias.append(km.inertia_)
    fig = plt.figure()
    plt.plot(ks, inertias, marker='o')
    plt.xlabel('k')
    plt.ylabel('Inertia (within-cluster SSE)')
    plt.title('Elbow curve')
    plt.xticks(ks)
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--json', required=True)
    ap.add_argument('--n-clusters', type=int, default=2)
    ap.add_argument('--pca', type=int, default=50)
    ap.add_argument('--n-init', type=int, default=10)
    ap.add_argument('--max-iter', type=int, default=300)
    ap.add_argument('--save-model', action='store_true')
    ap.add_argument('--load-model', type=str, default='')
    ap.add_argument('--elbow-max', type=int, default=0, help='If >0, compute elbow curve up to this k and exit')
    args = ap.parse_args()

    out_dir = 'runs_kmeans'
    ensure_dir(out_dir)

    X_raw01, y = load_planesnet_json(args.json)
    Xr, scaler, pca = preprocess_X(X_raw01, pca_components=args.pca)

    if args.elbow_max and args.elbow_max > 1:
        elbow_curve(Xr, kmin=2, kmax=args.elbow_max, out_path=os.path.join(out_dir, 'elbow.png'))
        print(f'Saved elbow curve to {os.path.join(out_dir, "elbow.png")}')
        return

    if args.load_model:
        km = joblib.load(args.load_model)
        clusters = km.predict(Xr)
    else:
        km = KMeans(n_clusters=args.n_clusters, n_init=args.n_init, max_iter=args.max_iter, random_state=42)
        clusters = km.fit_predict(Xr)
        if args.save_model:
            joblib.dump(km, os.path.join(out_dir, 'kmeans_model.joblib'))

    # Unsupervised metrics vs ground-truth
    ari = adjusted_rand_score(y, clusters)
    nmi = normalized_mutual_info_score(y, clusters)
    h = homogeneity_score(y, clusters)
    c = completeness_score(y, clusters)

    # Visualizations
    plot_pca_scatter(Xr, clusters, os.path.join(out_dir, 'pca_scatter_clusters.png'))
    plot_cluster_sizes(clusters, os.path.join(out_dir, 'cluster_sizes.png'))
    sil_avg = plot_silhouette(Xr, clusters, os.path.join(out_dir, 'silhouette.png'))
    cluster_means_png(X_raw01, clusters, os.path.join(out_dir, 'cluster_means.png'))

    # Save assignments
    df = pd.DataFrame({'id': np.arange(Xr.shape[0]), 'cluster': clusters, 'true_label': y})
    df.to_csv(os.path.join(out_dir, 'clusters.csv'), index=False)

    import json as _json
    with open(os.path.join(out_dir, 'metrics.json'), 'w') as f:
        _json.dump({
            'n_clusters': int(args.n_clusters),
            'pca_components': int(args.pca),
            'ARI': float(ari),
            'NMI': float(nmi),
            'homogeneity': float(h),
            'completeness': float(c),
            'silhouette': None if sil_avg is None else float(sil_avg)
        }, f, indent=2)

    print('Saved artifacts to ./runs_kmeans')
    print(f'ARI={ari:.4f}  NMI={nmi:.4f}  Homo={h:.4f}  Comp={c:.4f}  Sil={sil_avg if sil_avg is not None else float("nan"):.4f}')


if __name__ == '__main__':
    main()
