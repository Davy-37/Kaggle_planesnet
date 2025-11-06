#!/usr/bin/env python3
"""
Unified runner for PlanesNet experiments

- Lets you choose an algorithm (cnn/mlp/resnet18/knn/bayes/dtree/kmeans/hclust/timeseries)
- Choose mode: train or test (where applicable)
- Or run multiple/all algorithms in one go

This script wraps the other CLIs using subprocess so you don't need to remember all flags.
It assumes the companion scripts are in the same directory:
  - planesnet_train_classifier.py (CNN baseline)
  - planesnet_nn.py                (MLP/CNN/ResNet18)
  - planesnet_knn.py
  - planesnet_bayes.py
  - planesnet_decision_tree.py
  - planesnet_kmeans.py
  - planesnet_hclust.py
  - planesnet_timeseries.py

Examples
--------
# Train a small CNN
python planesnet_runner.py --algo cnn --mode train --json Data/planesnet/planesnet.json

# Test the last best CNN checkpoint
python planesnet_runner.py --algo cnn --mode test --json Data/planesnet/planesnet.json

# Train KNN with search
python planesnet_runner.py --algo knn --mode train --json Data/planesnet/planesnet.json --extra "--search --max-k 39"

# Run *all* supervised baselines in train mode
python planesnet_runner.py --algo all-supervised --mode train --json Data/planesnet/planesnet.json

# Run all unsupervised baselines
python planesnet_runner.py --algo all-unsupervised --mode train --json Data/planesnet/planesnet.json

# Build and forecast time series (global)
python planesnet_runner.py --algo timeseries --mode train --json Data/planesnet/planesnet.json --extra "--h 14"
"""

import argparse
import shlex
import subprocess
import sys
import os

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def run(cmd: str):
    print(f"\n>>> {cmd}")
    ret = subprocess.call(cmd, shell=True)
    if ret != 0:
        print(f"[error] command failed with code {ret}: {cmd}")
        sys.exit(ret)


def build_cmd(algo: str, mode: str, json_path: str | None, images_dir: str | None, extra: str | None):
    extra = extra or ""
    if algo in ("cnn", "mlp", "resnet18"):
        # Use planesnet_nn.py
        base = f"python {os.path.join(THIS_DIR, 'planesnet_nn.py')}"
        src = f"--json {shlex.quote(json_path)}" if json_path else f"--images-dir {shlex.quote(images_dir)}"
        model = f"--model {algo}"
        if mode == 'train':
            return f"{base} {src} {model} --epochs 25 --batch-size 256 {extra}".strip()
        else:  # test
            return f"{base} {src} {model} --evaluate --checkpoint runs_nn/best_model.pt {extra}".strip()

    if algo == 'cnn-baseline':
        # Use the minimal CNN trainer
        base = f"python {os.path.join(THIS_DIR, 'planesnet_train_classifier.py')}"
        src = f"--json {shlex.quote(json_path)}" if json_path else f"--images-dir {shlex.quote(images_dir)}"
        if mode == 'train':
            return f"{base} {src} --epochs 15 {extra}".strip()
        else:
            return f"{base} {src} --evaluate --checkpoint runs/best_model.pt {extra}".strip()

    if algo == 'knn':
        base = f"python {os.path.join(THIS_DIR, 'planesnet_knn.py')}"
        src = f"--json {shlex.quote(json_path)}"
        return f"{base} {src} {extra}".strip()

    if algo == 'bayes':
        base = f"python {os.path.join(THIS_DIR, 'planesnet_bayes.py')}"
        src = f"--json {shlex.quote(json_path)}"
        return f"{base} {src} {extra}".strip()

    if algo == 'dtree':
        base = f"python {os.path.join(THIS_DIR, 'planesnet_decision_tree.py')}"
        src = f"--json {shlex.quote(json_path)}"
        return f"{base} {src} {extra}".strip()

    if algo == 'kmeans':
        base = f"python {os.path.join(THIS_DIR, 'planesnet_kmeans.py')}"
        src = f"--json {shlex.quote(json_path)}"
        return f"{base} {src} {extra}".strip()

    if algo == 'hclust':
        base = f"python {os.path.join(THIS_DIR, 'planesnet_hclust.py')}"
        src = f"--json {shlex.quote(json_path)}"
        return f"{base} {src} {extra}".strip()

    if algo == 'timeseries':
        base = f"python {os.path.join(THIS_DIR, 'planesnet_timeseries.py')}"
        src = f"--json {shlex.quote(json_path)}"
        return f"{base} {src} {extra}".strip()

    raise ValueError(f"Unknown algo: {algo}")


def run_group(group: str, mode: str, json_path: str | None, images_dir: str | None, extra: str | None):
    if group == 'all-supervised':
        algos = ['cnn', 'mlp', 'resnet18', 'knn', 'bayes', 'dtree']
    elif group == 'all-unsupervised':
        algos = ['kmeans', 'hclust']
    elif group == 'all':
        algos = ['cnn', 'mlp', 'resnet18', 'knn', 'bayes', 'dtree', 'kmeans', 'hclust', 'timeseries']
    else:
        raise ValueError(f'Unknown group: {group}')

    for algo in algos:
        cmd = build_cmd(algo, mode, json_path, images_dir, extra)
        run(cmd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--algo', required=True, help='cnn|mlp|resnet18|cnn-baseline|knn|bayes|dtree|kmeans|hclust|timeseries|all-supervised|all-unsupervised|all')
    ap.add_argument('--mode', required=True, choices=['train','test'], help='train or test (unsupervised/time series simply run in train mode)')
    ap.add_argument('--json', type=str, help='Path to planesnet.json')
    ap.add_argument('--images-dir', type=str, help='Directory of PNGs if not using JSON')
    ap.add_argument('--extra', type=str, default='', help='Extra CLI flags to forward to the underlying script (quoted)')
    args = ap.parse_args()

    if not args.json and not args.images_dir and args.algo not in ('timeseries',):
        ap.error('You must provide --json or --images-dir for this algo.')

    if args.algo in ('all-supervised','all-unsupervised','all'):
        run_group(args.algo, args.mode, args.json, args.images_dir, args.extra)
    else:
        cmd = build_cmd(args.algo, args.mode, args.json, args.images_dir, args.extra)
        run(cmd)


if __name__ == '__main__':
    main()
