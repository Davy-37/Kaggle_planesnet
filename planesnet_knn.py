#!/usr/bin/env python3
"""
KNN on PlanesNet (JSON) with optional PCA + hyperparameter search

Usage:
  pip install scikit-learn matplotlib joblib
  python planesnet_knn.py --json Data/planesnet/planesnet.json --search --max-k 39 --pca 50

Outputs in ./runs_knn:
  - knn_model.joblib         (best KNN pipeline)
  - metrics.json             (test metrics)
  - confusion_matrix.png
  - roc_curve.png
  - pca_scatter.png          (2D PCA scatter of a sample)

Notes:
  - Features are 1200 pixels (R(400)+G(400)+B(400)), scaled to [0,1].
  - KNN benefits from normalization and (often) PCA to reduce noise.
  - Use --search to run a small CV search over k/metric/weights.
"""

import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def load_planesnet_json(path):
    with open(path, 'r') as f:
        d = json.load(f)
    X = np.asarray(d['data'], dtype=np.float32) / 255.0
    y = np.asarray(d['labels'], dtype=np.int64)
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


def plot_confusion_matrix(cm, classes, out_path):
    fig = plt.figure()
    plt.imshow(cm, interpolation='nearest')
    plt.title('Confusion matrix')
    plt.colorbar()
    ticks = np.arange(len(classes))
    plt.xticks(ticks, classes, rotation=45)
    plt.yticks(ticks, classes)
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'), ha='center', va='center',
                     color='white' if cm[i, j] > thresh else 'black')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    fig.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close(fig)


def plot_roc(fpr, tpr, out_path):
    fig = plt.figure()
    plt.plot(fpr, tpr, label='ROC')
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    fig.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close(fig)


def plot_pca_scatter(X, y, out_path, n_samples=2000, seed=42):
    rng = np.random.RandomState(seed)
    idx = rng.choice(np.arange(X.shape[0]), size=min(n_samples, X.shape[0]), replace=False)
    Xs = X[idx]
    ys = y[idx]
    pca = PCA(n_components=2, random_state=seed)
    Z = pca.fit_transform(Xs)
    fig = plt.figure()
    plt.scatter(Z[ys==0,0], Z[ys==0,1], s=5, alpha=0.6, label='no-plane')
    plt.scatter(Z[ys==1,0], Z[ys==1,1], s=5, alpha=0.6, label='plane')
    plt.title('PCA scatter (sample)')
    plt.legend()
    fig.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close(fig)


def build_pipeline(pca_components=None, metric='minkowski', weights='distance', n_neighbors=15):
    steps = []
    # Standardize to zero-mean/unit-variance even if X in [0,1]; helps KNN distance weighting
    steps.append(('scaler', StandardScaler(with_mean=True, with_std=True)))
    if pca_components is not None and pca_components > 0:
        steps.append(('pca', PCA(n_components=pca_components, random_state=42, whiten=False)))
    steps.append(('knn', KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric, weights=weights)))
    return Pipeline(steps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--json', required=True)
    ap.add_argument('--pca', type=int, default=0, help='#components for PCA (0 = no PCA)')
    ap.add_argument('--k', type=int, default=15, help='k neighbors (ignored if --search)')
    ap.add_argument('--metric', type=str, default='minkowski', choices=['minkowski','euclidean','manhattan','chebyshev'])
    ap.add_argument('--weights', type=str, default='distance', choices=['uniform','distance'])
    ap.add_argument('--search', action='store_true', help='Grid search over k/metric/weights and PCA')
    ap.add_argument('--max-k', type=int, default=39)
    ap.add_argument('--cv', type=int, default=3)
    args = ap.parse_args()

    out_dir = 'runs_knn'
    ensure_dir(out_dir)

    X, y = load_planesnet_json(args.json)
    tr_idx, va_idx, te_idx = split_indices(y)

    if args.search:
        # Define grid
        ks = [k for k in range(3, args.max_k+1, 2)]  # odd ks
        grid = {
            'pca__n_components': [0, 25, 50, 100] if args.pca == 0 else [args.pca],
            'knn__n_neighbors': ks,
            'knn__metric': ['euclidean', 'manhattan'],
            'knn__weights': ['uniform', 'distance']
        }
        # Build a pipeline with a *switchable* PCA step via n_components=0 (custom handling)
        # We'll emulate by creating two pipelines: with PCA and without PCA.
        # Easier approach: use a small helper that toggles PCA based on value.
        class PCAOrIdentity(PCA):
            def __init__(self, n_components=0, random_state=42):
                self._enabled = n_components not in (0, None)
                super().__init__(n_components=n_components if self._enabled else None, random_state=random_state)
            def fit(self, X, y=None):
                if not self._enabled:
                    self.mean_ = np.zeros(X.shape[1], dtype=X.dtype)
                    self.components_ = np.eye(X.shape[1], dtype=X.dtype)
                    self.n_features_in_ = X.shape[1]
                    return self
                return super().fit(X, y)
            def transform(self, X):
                if not self._enabled:
                    return X
                return super().transform(X)
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCAOrIdentity(n_components=0, random_state=42)),
            ('knn', KNeighborsClassifier())
        ])
        gs = GridSearchCV(pipe, grid, cv=args.cv, n_jobs=-1, verbose=1, scoring='roc_auc')
        gs.fit(X[tr_idx], y[tr_idx])
        print('Best params:', gs.best_params_)
        model = gs.best_estimator_
    else:
        model = build_pipeline(
            pca_components=args.pca if args.pca>0 else None,
            metric=args.metric,
            weights=args.weights,
            n_neighbors=args.k
        )
        model.fit(X[tr_idx], y[tr_idx])

    # Validate
    from sklearn.metrics import accuracy_score
    val_proba = model.predict_proba(X[va_idx])[:,1]
    val_pred = (val_proba >= 0.5).astype(int)
    val_auc = roc_auc_score(y[va_idx], val_proba)
    val_acc = accuracy_score(y[va_idx], val_pred)
    print(f'VAL  AUC={val_auc:.4f}  ACC={val_acc:.4f}')

    # Test
    test_proba = model.predict_proba(X[te_idx])[:,1]
    test_pred = (test_proba >= 0.5).astype(int)
    test_auc = roc_auc_score(y[te_idx], test_proba)
    test_acc = accuracy_score(y[te_idx], test_pred)

    print('TEST report:')
    print(classification_report(y[te_idx], test_pred, target_names=['no-plane','plane']))
    cm = confusion_matrix(y[te_idx], test_pred)

    # Save artifacts
    joblib.dump(model, os.path.join(out_dir, 'knn_model.joblib'))

    import json as _json
    with open(os.path.join(out_dir, 'metrics.json'), 'w') as f:
        _json.dump({
            'val_auc': float(val_auc), 'val_acc': float(val_acc),
            'test_auc': float(test_auc), 'test_acc': float(test_acc),
            'confusion_matrix': cm.tolist()
        }, f, indent=2)

    plot_confusion_matrix(cm, ['no-plane','plane'], os.path.join(out_dir, 'confusion_matrix.png'))

    fpr, tpr, _ = roc_curve(y[te_idx], test_proba)
    plot_roc(fpr, tpr, os.path.join(out_dir, 'roc_curve.png'))

    plot_pca_scatter(X, y, os.path.join(out_dir, 'pca_scatter.png'))

    print('Artifacts saved in ./runs_knn')


if __name__ == '__main__':
    main()
