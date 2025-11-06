#!/usr/bin/env python3
"""
Decision Tree on PlanesNet + Visualizations
- Trains a scikit-learn DecisionTreeClassifier on PlanesNet JSON.
- Saves a matplotlib visualization of the tree (top levels) and feature-importance heatmaps.
- Optionally exports Graphviz DOT to render a high-res tree with `dot`.

Usage:
  pip install scikit-learn matplotlib graphviz
  python planesnet_decision_tree.py --json /path/to/planesnet.json --max-depth 6

Outputs in ./runs_tree:
  - tree_plot.png            (matplotlib plot of the tree)
  - tree.dot                 (Graphviz source; render with: dot -Tpng runs_tree/tree.dot -o runs_tree/tree_graphviz.png)
  - fi_heatmap_all.png       (20x20 heatmap of summed importances across RGB)
  - fi_heatmap_R.png         (20x20 heatmap of R channel importances)
  - fi_heatmap_G.png         (20x20 heatmap of G channel importances)
  - fi_heatmap_B.png         (20x20 heatmap of B channel importances)
"""

import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_graphviz
from sklearn.metrics import classification_report, confusion_matrix


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def load_planesnet_json(path):
    with open(path, 'r') as f:
        d = json.load(f)
    X = np.asarray(d['data'], dtype=np.float32)  # shape: (N, 1200)
    y = np.asarray(d['labels'], dtype=np.int64)
    # scale to [0,1]
    X /= 255.0
    return X, y


def split_indices(y, seed=42):
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=seed)
    train_idx, not_train_idx = next(sss1.split(np.zeros_like(y), y))
    y_not = y[not_train_idx]
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
    val_rel, test_rel = next(sss2.split(np.zeros_like(y_not), y_not))
    val_idx = not_train_idx[val_rel]
    test_idx = not_train_idx[test_rel]
    return train_idx, val_idx, test_idx


def plot_tree_matplotlib(clf, out_path, max_depth_to_plot=None):
    fig = plt.figure(figsize=(20, 10))
    plot_tree(
        clf,
        filled=True,
        rounded=True,
        max_depth=max_depth_to_plot,
        proportion=True,
        impurity=True,
        fontsize=8
    )
    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def feature_importance_heatmaps(importances, out_dir):
    # importances is length 1200: 400 R, 400 G, 400 B in row-major 20x20
    R = importances[:400].reshape(20, 20)
    G = importances[400:800].reshape(20, 20)
    B = importances[800:1200].reshape(20, 20)
    ALL = R + G + B

    def save_heatmap(mat, title, out_path):
        fig = plt.figure()
        plt.imshow(mat)
        plt.title(title)
        plt.colorbar()
        plt.tight_layout()
        fig.savefig(out_path, dpi=200)
        plt.close(fig)

    save_heatmap(ALL, 'Feature importance (R+G+B)', os.path.join(out_dir, 'fi_heatmap_all.png'))
    save_heatmap(R, 'Feature importance (R)', os.path.join(out_dir, 'fi_heatmap_R.png'))
    save_heatmap(G, 'Feature importance (G)', os.path.join(out_dir, 'fi_heatmap_G.png'))
    save_heatmap(B, 'Feature importance (B)', os.path.join(out_dir, 'fi_heatmap_B.png'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--json', required=True, help='Path to planesnet.json')
    ap.add_argument('--max-depth', type=int, default=6)
    ap.add_argument('--min-samples-leaf', type=int, default=10)
    ap.add_argument('--criterion', type=str, default='gini', choices=['gini', 'entropy', 'log_loss'])
    ap.add_argument('--export-graphviz', action='store_true', help='Also export tree.dot for Graphviz')
    args = ap.parse_args()

    out_dir = 'runs_tree'
    ensure_dir(out_dir)

    X, y = load_planesnet_json(args.json)
    tr_idx, va_idx, te_idx = split_indices(y)

    clf = DecisionTreeClassifier(
        criterion=args.criterion,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        class_weight='balanced',
        random_state=42
    )
    clf.fit(X[tr_idx], y[tr_idx])

    # Evaluate
    for name, idx in [('val', va_idx), ('test', te_idx)]:
        y_pred = clf.predict(X[idx])
        print(f"=== {name.upper()} ===")
        print(classification_report(y[idx], y_pred, target_names=['no-plane', 'plane']))
        print(confusion_matrix(y[idx], y_pred))

    # Visualizations
    plot_tree_matplotlib(clf, os.path.join(out_dir, 'tree_plot.png'), max_depth_to_plot=None)

    if args.export_graphviz:
        export_graphviz(
            clf,
            out_file=os.path.join(out_dir, 'tree.dot'),
            filled=True,
            rounded=True,
            proportion=True,
            feature_names=[f'p{i}' for i in range(X.shape[1])],
            class_names=['no-plane', 'plane']
        )
        print('Wrote Graphviz DOT to runs_tree/tree.dot (render with: dot -Tpng runs_tree/tree.dot -o runs_tree/tree_graphviz.png)')

    # Feature importance heatmaps
    feature_importance_heatmaps(clf.feature_importances_, out_dir)
    print('Saved: tree_plot.png, fi_heatmap_*.png in runs_tree/')


if __name__ == '__main__':
    main()
