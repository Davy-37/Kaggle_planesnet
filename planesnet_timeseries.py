import argparse
import json
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX

plt.rcParams["figure.dpi"] = 140


def mkdir_runs(p):
    os.makedirs(p, exist_ok=True)


def parse_scene_date(scene_id):
    # format type PlanetScope: YYYYMMDD_HHMMSS_...
    if not isinstance(scene_id, str):
        return None
    try:
        parts = scene_id.split("_")
        date_part = parts[0]
        time_part = parts[1] if len(parts) > 1 else ""
        if len(date_part) == 8 and len(time_part) >= 6:
            dt = datetime.strptime(date_part + time_part[:6], "%Y%m%d%H%M%S")
            return dt
        if len(date_part) == 8:
            dt = datetime.strptime(date_part, "%Y%m%d")
            return dt
    except Exception:
        pass
    return None


def build_df(json_path):
    with open(json_path) as f:
        d = json.load(f)
    labels = np.array(d["labels"])
    scene_ids = d.get("scene_ids", [])
    locs = d.get("locations", [])
    rows = []
    for i, lab in enumerate(labels):
        sid = scene_ids[i] if i < len(scene_ids) else None
        dt = parse_scene_date(sid) if sid else None
        if dt is None:
            continue
        lon = lat = None
        if i < len(locs) and isinstance(locs[i], (list, tuple)) and len(locs[i]) == 2:
            lon, lat = locs[i][0], locs[i][1]
        rows.append({"dt": dt, "label": int(lab), "lon": lon, "lat": lat})
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("Aucune date exploitable dans scene_ids")
    df.sort_values("dt", inplace=True)
    return df


def make_series(df, geo_bin=0):
    df = df.copy()
    dfp = df[df["label"] == 1].copy()
    if geo_bin and geo_bin > 0 and dfp["lon"].notna().any():
        dfp["lon_bin"] = dfp["lon"].round(geo_bin)
        dfp["lat_bin"] = dfp["lat"].round(geo_bin)
        groups = dfp.groupby(["lon_bin", "lat_bin"])
        out = {}
        for (lo, la), g in groups:
            s = g.set_index("dt").resample("D").size().rename("count").asfreq("D", fill_value=0)
            key = f"series_{la:+.{geo_bin}f}_{lo:+.{geo_bin}f}"
            out[key] = s
        if not out:
            raise RuntimeError("Aucune série geo formée")
        return out
    else:
        s = dfp.set_index("dt").resample("D").size().rename("count").asfreq("D", fill_value=0)
        return {"series_global": s}


def fit_forecast(s, h=14, seasonal=7):
    s = s.astype("float64")
    n = len(s)
    if n < 30:
        raise RuntimeError("Série trop courte pour SARIMAX")
    split = int(n * 0.8)
    train, test = s.iloc[:split], s.iloc[split:]

    order = (1, 1, 1)
    seasonal_order = (1, 0, 1, seasonal) if seasonal else (0, 0, 0, 0)
    model = SARIMAX(train, order=order, seasonal_order=seasonal_order, trend="n",
                    enforce_stationarity=False, enforce_invertibility=False)
    res = model.fit(disp=False)

    pred = res.get_forecast(steps=len(test))
    test_pred = pred.predicted_mean

    fut = res.get_forecast(steps=h)
    fut_mean = fut.predicted_mean
    fut_ci = fut.conf_int(alpha=0.2)

    mae = float(mean_absolute_error(test, test_pred))
    rmse = float(np.sqrt(mean_squared_error(test, test_pred)))
    mape = float((np.abs((test - test_pred) / np.maximum(test, 1e-6))).mean())

    return train, test, test_pred, fut_mean, fut_ci, {"MAE": mae, "RMSE": rmse, "MAPE": mape}


def draw_series(s, out_path):
    fig = plt.figure(figsize=(8, 3))
    s.plot()
    plt.title("Comptage quotidien avions")
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def draw_decomp(s, out_path):
    try:
        dec = seasonal_decompose(s, period=7, model="additive", two_sided=False, extrapolate_trend="freq")
        fig = dec.plot()
        fig.set_size_inches(8, 6)
        fig.savefig(out_path)
        plt.close(fig)
    except Exception as e:
        print("decomp failed:", e)


def draw_forecast(train, test, test_pred, fut_mean, fut_ci, out_path):
    fig = plt.figure(figsize=(9, 4))
    train.plot(label="train")
    test.plot(label="test")
    test_pred.plot(label="test pred")
    fut_mean.plot(label="forecast")
    idx = fut_ci.index
    plt.fill_between(idx, fut_ci.iloc[:, 0], fut_ci.iloc[:, 1], alpha=0.2, label="80% IC")
    plt.legend()
    plt.title("Prévision SARIMAX")
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--geo-bin", type=int, default=0)
    ap.add_argument("--h", type=int, default=14)
    args = ap.parse_args()

    out_dir = "runs_ts"
    mkdir_runs(out_dir)

    df = build_df(args.json)
    series_dict = make_series(df, geo_bin=args.geo_bin)

    metrics_all = {}
    for name, s in series_dict.items():
        s.to_csv(os.path.join(out_dir, f"{name}.csv"), header=True)
        if name == "series_global":
            draw_series(s, os.path.join(out_dir, "series_plot.png"))
            draw_decomp(s, os.path.join(out_dir, "decomp.png"))
        train, test, test_pred, fut_mean, fut_ci, metrics = fit_forecast(s, h=args.h, seasonal=7)
        metrics_all[name] = metrics
        draw_forecast(train, test, test_pred, fut_mean, fut_ci, os.path.join(out_dir, f"forecast_{name}.png"))

    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics_all, f, indent=2)

    print("Résultats dans ./runs_ts")
    for k, v in metrics_all.items():
        print(k, v)


if __name__ == "__main__":
    main()
