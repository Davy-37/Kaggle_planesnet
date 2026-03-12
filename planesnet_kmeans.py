import argparse
import json
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    homogeneity_score,
    completeness_score,
    silhouette_samples,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler

plt.rcParams["figure.dpi"] = 120


def mkdir_runs(p):
    os.makedirs(p, exist_ok=True)


def load_json(path):
    with open(path) as f:
        d = json.load(f)
    X = np.asarray(d["data"], dtype=np.float32) / 255.0
    y = np.asarray(d["labels"], dtype=np.int64)
    return X, y


def preprocess(X, pca_components=0):
    scaler = StandardScaler(with_mean=True, with_std=True)
    Xs = scaler.fit_transform(X)
    pca = None
    if pca_components and pca_components > 0:
        pca = PCA(n_components=pca_components, random_state=42)
        Xr = pca.fit_transform(Xs)
    else:
        Xr = Xs
    return Xr, scaler, pca


def draw_pca_clusters(Xr, labels, out_path):
    if Xr.shape[1] > 2:
        Z = PCA(n_components=2, random_state=42).fit_transform(Xr)
    else:
        Z = Xr
    fig = plt.figure()
    for c in np.unique(labels):
        idx = labels == c
        plt.scatter(Z[idx, 0], Z[idx, 1], s=5, alpha=0.7, label=f"cluster {c}")
    plt.legend()
    plt.title("PCA par cluster")
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def draw_sizes(labels, out_path):
    uniq, counts = np.unique(labels, return_counts=True)
    fig = plt.figure()
    plt.bar([str(u) for u in uniq], counts)
    plt.xlabel("Cluster")
    plt.ylabel("Effectif")
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def draw_silhouette(Xr, labels, out_path):
    try:
        sil_vals = silhouette_samples(Xr, labels, metric="euclidean")
        sil_avg = silhouette_score(Xr, labels, metric="euclidean")
    except Exception as e:
        print("silhouette failed:", e)
        return None
    fig = plt.figure(figsize=(6, 4))
    y_lower = 10
    for c in sorted(np.unique(labels)):
        c_sil = np.sort(sil_vals[labels == c])
        size_c = c_sil.shape[0]
        y_upper = y_lower + size_c
        plt.fill_betweenx(np.arange(y_lower, y_upper), 0, c_sil)
        plt.text(-0.05, y_lower + 0.5 * size_c, str(c))
        y_lower = y_upper + 10
    plt.axvline(sil_avg, linestyle="--")
    plt.xlabel("Silhouette")
    plt.ylabel("Samples")
    plt.title(f"Silhouette (moy={sil_avg:.3f})")
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return float(sil_avg)


def draw_cluster_means(X_raw, labels, out_path):
    H = W = 20
    clusters = np.unique(labels)
    imgs = []
    for c in clusters:
        Xc = X_raw[labels == c]
        if Xc.size == 0:
            avg = np.zeros((H, W, 3), dtype=np.float32)
        else:
            R = Xc[:, :400].mean(axis=0).reshape(H, W)
            G = Xc[:, 400:800].mean(axis=0).reshape(H, W)
            B = Xc[:, 800:1200].mean(axis=0).reshape(H, W)
            avg = np.stack([R, G, B], axis=2)
        imgs.append((c, np.clip(avg, 0, 1)))
    cols = min(8, len(clusters))
    rows = int(np.ceil(len(clusters) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    axes = axes.reshape(rows, cols)
    for i, (c, img) in enumerate(imgs):
        r, k = i // cols, i % cols
        axes[r, k].imshow(img)
        axes[r, k].set_title(f"cluster {c}")
        axes[r, k].axis("off")
    for j in range(len(imgs), rows * cols):
        r, k = j // cols, j % cols
        axes[r, k].axis("off")
    plt.suptitle("Image moyenne par cluster")
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def elbow(Xr, kmin=2, kmax=10, out_path="elbow.png", n_init=10):
    inertias = []
    ks = list(range(kmin, kmax + 1))
    for k in ks:
        km = KMeans(n_clusters=k, n_init=n_init, random_state=42)
        km.fit(Xr)
        inertias.append(km.inertia_)
    fig = plt.figure()
    plt.plot(ks, inertias, marker="o")
    plt.xlabel("k")
    plt.ylabel("Inertie")
    plt.xticks(ks)
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--n-clusters", type=int, default=2)
    ap.add_argument("--pca", type=int, default=50)
    ap.add_argument("--n-init", type=int, default=10)
    ap.add_argument("--max-iter", type=int, default=300)
    ap.add_argument("--save-model", action="store_true")
    ap.add_argument("--load-model", type=str, default="")
    ap.add_argument("--elbow-max", type=int, default=0)
    args = ap.parse_args()

    out_dir = "runs_kmeans"
    mkdir_runs(out_dir)

    X_raw, y = load_json(args.json)
    Xr, scaler, pca = preprocess(X_raw, pca_components=args.pca)

    if args.elbow_max and args.elbow_max > 1:
        elbow(Xr, kmin=2, kmax=args.elbow_max, out_path=os.path.join(out_dir, "elbow.png"))
        print("Courbe elbow sauvegardée")
        return

    if args.load_model:
        km = joblib.load(args.load_model)
        clusters = km.predict(Xr)
    else:
        km = KMeans(n_clusters=args.n_clusters, n_init=args.n_init, max_iter=args.max_iter, random_state=42)
        clusters = km.fit_predict(Xr)
        if args.save_model:
            joblib.dump(km, os.path.join(out_dir, "kmeans_model.joblib"))

    ari = adjusted_rand_score(y, clusters)
    nmi = normalized_mutual_info_score(y, clusters)
    h = homogeneity_score(y, clusters)
    c = completeness_score(y, clusters)

    draw_pca_clusters(Xr, clusters, os.path.join(out_dir, "pca_scatter_clusters.png"))
    draw_sizes(clusters, os.path.join(out_dir, "cluster_sizes.png"))
    sil_avg = draw_silhouette(Xr, clusters, os.path.join(out_dir, "silhouette.png"))
    draw_cluster_means(X_raw, clusters, os.path.join(out_dir, "cluster_means.png"))

    df = pd.DataFrame({"id": np.arange(Xr.shape[0]), "cluster": clusters, "true_label": y})
    df.to_csv(os.path.join(out_dir, "clusters.csv"), index=False)

    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump({
            "n_clusters": int(args.n_clusters),
            "pca_components": int(args.pca),
            "ARI": float(ari),
            "NMI": float(nmi),
            "homogeneity": float(h),
            "completeness": float(c),
            "silhouette": None if sil_avg is None else float(sil_avg),
        }, f, indent=2)

    print("Résultats dans ./runs_kmeans")
    print("ARI={:.4f} NMI={:.4f} Homo={:.4f} Comp={:.4f} Sil={:.4f}".format(
        ari, nmi, h, c, sil_avg if sil_avg is not None else float("nan")))


if __name__ == "__main__":
    main()
