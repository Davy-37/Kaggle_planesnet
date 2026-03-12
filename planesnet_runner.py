import argparse
import os
import shlex
import subprocess
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))


def exec_cmd(cmd):
    print("\n>>>", cmd)
    code = subprocess.call(cmd, shell=True)
    if code != 0:
        print("[error] exit code", code, ":", cmd)
        sys.exit(code)


def get_command(algo, mode, json_path, images_dir, extra):
    extra = extra or ""
    if algo in ("cnn", "mlp", "resnet18"):
        base = "python " + os.path.join(script_dir, "planesnet_cnn.py")
        src = f"--json {shlex.quote(json_path)}" if json_path else f"--images-dir {shlex.quote(images_dir)}"
        model = f"--model {algo}"
        if mode == "train":
            return f"{base} {src} {model} --epochs 25 --batch-size 256 {extra}".strip()
        return f"{base} {src} {model} --evaluate --checkpoint runs_nn/best_model.pt {extra}".strip()

    if algo == "cnn-baseline":
        base = "python " + os.path.join(script_dir, "planesnet_train_classifier.py")
        src = f"--json {shlex.quote(json_path)}" if json_path else f"--images-dir {shlex.quote(images_dir)}"
        if mode == "train":
            return f"{base} {src} --epochs 15 {extra}".strip()
        return f"{base} {src} --evaluate --checkpoint runs/best_model.pt {extra}".strip()

    scripts = {
        "knn": "planesnet_knn.py",
        "bayes": "planesnet_bayes.py",
        "dtree": "planesnet_decision_tree.py",
    }
    if algo in scripts:
        base = "python " + os.path.join(script_dir, scripts[algo])
        src = f"--json {shlex.quote(json_path)}"
        return f"{base} {src} {extra}".strip()

    raise ValueError("algo inconnu:", algo)


def run_batch(group, mode, json_path, images_dir, extra):
    if group == "all-supervised":
        algos = ["cnn", "mlp", "resnet18", "knn", "bayes", "dtree"]
    elif group == "all":
        algos = ["cnn", "mlp", "resnet18", "knn", "bayes", "dtree"]
    else:
        raise ValueError("groupe inconnu:", group)

    for a in algos:
        cmd = get_command(a, mode, json_path, images_dir, extra)
        exec_cmd(cmd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", required=True,
                        help="cnn|mlp|resnet18|cnn-baseline|knn|bayes|dtree|all-supervised|all")
    parser.add_argument("--mode", required=True, choices=["train", "test"])
    parser.add_argument("--json", type=str)
    parser.add_argument("--images-dir", type=str)
    parser.add_argument("--extra", type=str, default="")
    args = parser.parse_args()

    if not args.json and not args.images_dir:
        parser.error("Il faut --json ou --images-dir.")

    if args.algo in ("all-supervised", "all"):
        run_batch(args.algo, args.mode, args.json, args.images_dir, args.extra)
    else:
        cmd = get_command(args.algo, args.mode, args.json, args.images_dir, args.extra)
        exec_cmd(cmd)
