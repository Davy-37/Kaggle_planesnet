import argparse
import json
import os

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_graphviz
from sklearn.metrics import classification_report, confusion_matrix


def mkdir_runs(p):
    os.makedirs(p, exist_ok=True)


def load_json(path):
    with open(path) as f:
        d = json.load(f)
    X = np.asarray(d["data"], dtype=np.float32) / 255.0
    y = np.asarray(d["labels"], dtype=np.int64)
    return X, y


def split_train_val_test(y, seed=42):
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=seed)
    train_idx, not_train_idx = next(sss1.split(np.zeros_like(y), y))
    y_not = y[not_train_idx]
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
    val_rel, test_rel = next(sss2.split(np.zeros_like(y_not), y_not))
    val_idx = not_train_idx[val_rel]
    test_idx = not_train_idx[test_rel]
    return train_idx, val_idx, test_idx


def draw_tree(clf, out_path, max_depth=None):
    fig = plt.figure(figsize=(20, 10))
    plot_tree(clf, filled=True, rounded=True, max_depth=max_depth, proportion=True, impurity=True, fontsize=8)
    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def draw_importance_heatmaps(importances, out_dir):
    # 1200 = 400 R + 400 G + 400 B en 20x20
    R = importances[:400].reshape(20, 20)
    G = importances[400:800].reshape(20, 20)
    B = importances[800:1200].reshape(20, 20)
    ALL = R + G + B

    def save_heatmap(mat, title, fp):
        fig = plt.figure()
        plt.imshow(mat)
        plt.title(title)
        plt.colorbar()
        plt.tight_layout()
        fig.savefig(fp, dpi=200)
        plt.close(fig)

    save_heatmap(ALL, "Importance (R+G+B)", os.path.join(out_dir, "fi_heatmap_all.png"))
    save_heatmap(R, "Importance R", os.path.join(out_dir, "fi_heatmap_R.png"))
    save_heatmap(G, "Importance G", os.path.join(out_dir, "fi_heatmap_G.png"))
    save_heatmap(B, "Importance B", os.path.join(out_dir, "fi_heatmap_B.png"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--max-depth", type=int, default=6)
    ap.add_argument("--min-samples-leaf", type=int, default=10)
    ap.add_argument("--criterion", type=str, default="gini", choices=["gini", "entropy", "log_loss"])
    ap.add_argument("--export-graphviz", action="store_true")
    args = ap.parse_args()

    out_dir = "runs_tree"
    mkdir_runs(out_dir)

    X, y = load_json(args.json)
    tr_idx, va_idx, te_idx = split_train_val_test(y)

    clf = DecisionTreeClassifier(
        criterion=args.criterion,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        class_weight="balanced",
        random_state=42,
    )
    clf.fit(X[tr_idx], y[tr_idx])

    for name, idx in [("val", va_idx), ("test", te_idx)]:
        y_pred = clf.predict(X[idx])
        print("=== ", name.upper(), " ===")
        print(classification_report(y[idx], y_pred, target_names=["no-plane", "plane"]))
        print(confusion_matrix(y[idx], y_pred))

    draw_tree(clf, os.path.join(out_dir, "tree_plot.png"), max_depth=None)

    if args.export_graphviz:
        export_graphviz(
            clf,
            out_file=os.path.join(out_dir, "tree.dot"),
            filled=True,
            rounded=True,
            proportion=True,
            feature_names=[f"p{i}" for i in range(X.shape[1])],
            class_names=["no-plane", "plane"],
        )
        print("tree.dot écrit (dot -Tpng runs_tree/tree.dot -o ... pour rendre)")

    draw_importance_heatmaps(clf.feature_importances_, out_dir)
    print("Sauvegardé: tree_plot.png, fi_heatmap_*.png dans runs_tree/")


if __name__ == "__main__":
    main()
