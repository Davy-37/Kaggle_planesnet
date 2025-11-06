#!/usr/bin/env python3
"""
Time Series pipeline for PlanesNet (counts of planes over time + forecasting)

What it does
------------
1) Parses timestamps from scene_ids in planesnet.json (PlanetScope-like ids: YYYYMMDD_HHMMSS_...)
2) Builds daily time series of plane counts (label==1) globally or per geo-bin
3) Splits train/test chronologically, evaluates a SARIMAX model, and forecasts H days
4) Saves plots: series, seasonal decomposition, forecast, and metrics

Usage examples
--------------
# Global daily plane counts + 14-day forecast
python planesnet_timeseries.py --json Data/planesnet/planesnet.json --h 14

# Per-geo time series (approx airport) using 2-decimal lat/lon binning + 7-day forecast
python planesnet_timeseries.py --json Data/planesnet/planesnet.json --geo-bin 2 --h 7

Outputs (./runs_ts)
-------------------
- series_global.csv / series_<lat>_<lon>.csv  : daily series
- series_plot.png                             : full series (global)
- decomp.png                                   : seasonal decomposition (global)
- forecast.png                                 : train/test/forecast plot
- metrics.json                                 : MAE/MAPE/RMSE on test split

Notes
-----
- If a scene_id has no parseable datetime, that sample is skipped for TS building.
- For per-geo, locations are rounded to the specified decimals; choose 1..3 for coarse..fine.
"""

import argparse
import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error

plt.rcParams['figure.dpi'] = 140


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def parse_scene_dt(scene_id: str):
    """Parse PlanetScope-style scene_id like '20170518_182130_1010' -> datetime(2017-05-18 18:21:30).
    Returns None if parsing fails.
    """
    if not isinstance(scene_id, str):
        return None
    try:
        # Expect at least YYYYMMDD_HHMMSS
        date_part = scene_id.split('_')[0]
        time_part = scene_id.split('_')[1] if '_' in scene_id else ''
        if len(date_part) == 8 and len(time_part) >= 6:
            dt = datetime.strptime(date_part + time_part[:6], "%Y%m%d%H%M%S")
            return dt
        # fallback: only date
        if len(date_part) == 8:
            dt = datetime.strptime(date_part, "%Y%m%d")
            return dt
    except Exception:
        return None
    return None


def build_dataframe(json_path: str) -> pd.DataFrame:
    with open(json_path, 'r') as f:
        d = json.load(f)
    labels = np.array(d['labels'])
    scene_ids = d.get('scene_ids', [])
    locs = d.get('locations', [])
    rows = []
    for i, lab in enumerate(labels):
        sid = scene_ids[i] if i < len(scene_ids) else None
        dt = parse_scene_dt(sid) if sid is not None else None
        if dt is None:
            continue  # skip samples without time
        lon, lat = (None, None)
        if i < len(locs) and isinstance(locs[i], (list, tuple)) and len(locs[i]) == 2:
            lon, lat = locs[i][0], locs[i][1]
        rows.append({'dt': dt, 'label': int(lab), 'lon': lon, 'lat': lat})
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError('No timestamps could be parsed from scene_ids; cannot build time series.')
    df.sort_values('dt', inplace=True)
    return df


def make_series(df: pd.DataFrame, geo_bin: int = 0):
    """Return a dict of name->daily series (pd.Series indexed by date) of plane counts.
       geo_bin=0 -> global. Else round lon/lat to that many decimals and make per-bin series.
    """
    df = df.copy()
    # Only planes
    dfp = df[df['label'] == 1].copy()
    if geo_bin and geo_bin > 0 and dfp['lon'].notna().any():
        dfp['lon_bin'] = dfp['lon'].round(geo_bin)
        dfp['lat_bin'] = dfp['lat'].round(geo_bin)
        groups = dfp.groupby(['lon_bin', 'lat_bin'])
        series_dict = {}
        for (lo, la), g in groups:
            s = g.set_index('dt').resample('D').size().rename('count').asfreq('D', fill_value=0)
            key = f"series_{la:+.{geo_bin}f}_{lo:+.{geo_bin}f}"
            series_dict[key] = s
        if not series_dict:
            raise RuntimeError('No per-geo series could be formed; check geo_bin or missing locations.')
        return series_dict
    else:
        s = dfp.set_index('dt').resample('D').size().rename('count').asfreq('D', fill_value=0)
        return {'series_global': s}


def train_test_forecast(s: pd.Series, h: int = 14, seasonal: int | None = 7):
    """Chronological split: 80% train, 20% test; fit SARIMAX, forecast h steps.
       seasonal: season length (e.g., 7 for weekly), or None to disable.
    """
    s = s.astype('float64')
    n = len(s)
    if n < 30:
        raise RuntimeError('Time series too short (<30 points) for SARIMAX evaluation.')
    split = int(n * 0.8)
    train, test = s.iloc[:split], s.iloc[split:]

    # Simple SARIMAX config (light grid if seasonal given)
    order = (1,1,1)
    seasonal_order = (1,0,1, seasonal) if seasonal else (0,0,0,0)

    model = SARIMAX(train, order=order, seasonal_order=seasonal_order, trend='n', enforce_stationarity=False, enforce_invertibility=False)
    res = model.fit(disp=False)

    # In-sample one-step forecast on test horizon
    pred = res.get_forecast(steps=len(test))
    test_pred = pred.predicted_mean

    # Future forecast h
    fut = res.get_forecast(steps=h)
    fut_mean = fut.predicted_mean
    fut_ci = fut.conf_int(alpha=0.2)  # 80% CI

    # Metrics
    mae = float(mean_absolute_error(test, test_pred))
    rmse = float(np.sqrt(mean_squared_error(test, test_pred)))
    mape = float((np.abs((test - test_pred) / np.maximum(test, 1e-6))).mean())

    return train, test, test_pred, fut_mean, fut_ci, {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}


def plot_series(s: pd.Series, out_path: str):
    fig = plt.figure(figsize=(8,3))
    s.plot()
    plt.title('Daily plane counts')
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_decomp(s: pd.Series, out_path: str):
    try:
        dec = seasonal_decompose(s, period=7, model='additive', two_sided=False, extrapolate_trend='freq')
        fig = dec.plot()
        fig.set_size_inches(8,6)
        fig.savefig(out_path)
        plt.close(fig)
    except Exception as e:
        print('[warn] decomposition failed:', e)


def plot_forecast(train: pd.Series, test: pd.Series, test_pred: pd.Series, fut_mean: pd.Series, fut_ci: pd.DataFrame, out_path: str):
    fig = plt.figure(figsize=(9,4))
    train.plot(label='train')
    test.plot(label='test')
    test_pred.plot(label='test pred')
    fut_mean.plot(label='forecast')
    # CI shading
    idx = fut_ci.index
    plt.fill_between(idx, fut_ci.iloc[:,0], fut_ci.iloc[:,1], alpha=0.2, label='80% CI')
    plt.legend()
    plt.title('SARIMAX forecast')
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--json', required=True)
    ap.add_argument('--geo-bin', type=int, default=0, help='Round lon/lat to N decimals to create per-geo series')
    ap.add_argument('--h', type=int, default=14, help='Forecast horizon (days)')
    args = ap.parse_args()

    out_dir = 'runs_ts'
    ensure_dir(out_dir)

    df = build_dataframe(args.json)
    series_dict = make_series(df, geo_bin=args.geo_bin)

    metrics_all = {}
    for name, s in series_dict.items():
        # save CSV
        s.to_csv(os.path.join(out_dir, f'{name}.csv'), header=True)
        # plots
        if name == 'series_global':
            plot_series(s, os.path.join(out_dir, 'series_plot.png'))
            plot_decomp(s, os.path.join(out_dir, 'decomp.png'))
        # model
        train, test, test_pred, fut_mean, fut_ci, metrics = train_test_forecast(s, h=args.h, seasonal=7)
        metrics_all[name] = metrics
        # forecast plot
        plot_forecast(train, test, test_pred, fut_mean, fut_ci, os.path.join(out_dir, f'forecast_{name}.png'))

    # save metrics
    import json as _json
    with open(os.path.join(out_dir, 'metrics.json'), 'w') as f:
        _json.dump(metrics_all, f, indent=2)

    print('Saved artifacts to ./runs_ts')
    for k,v in metrics_all.items():
        print(k, v)


if __name__ == '__main__':
    main()
