# app/data/forecasting.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Dict, Any

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

Metric = Literal["rms_vel", "rms_acc"]
Direction = Literal["d1", "d2", "d3"]
ModelName = Literal["linear", "arima", "lgbm"]


@dataclass
class ForecastRequest:
    sn: int | str
    metric: Metric
    direction: Direction
    horizon_days: int = 30
    model: ModelName = "lgbm"


def _prepare_daily_series(df: pd.DataFrame, value_col: str) -> pd.Series:
    """
    Build a daily time series from your historical RMS data.

    Expected columns in df (from EnhancedVibExtractor.get_historical_data):
      - 'mytimestamp' (string or datetime-like)
      - 'rms_vel_d1/2/3'
      - 'rms_acc_d1/2/3'
    """
    if "mytimestamp" not in df.columns:
        raise ValueError("Expected 'mytimestamp' column in historical data")

    # Keep just the column we care about
    work = df[["mytimestamp", value_col]].dropna().copy()

    # Convert timestamp to datetime, then to date
    work["dt"] = pd.to_datetime(work["mytimestamp"], errors="coerce")
    work = work.dropna(subset=["dt"])
    work["date"] = work["dt"].dt.date

    # Daily average RMS
    daily = (
        work.groupby("date")[value_col]
        .mean()
        .reset_index()
        .sort_values("date")
    )

    daily["date"] = pd.to_datetime(daily["date"])
    series = daily.set_index("date")[value_col]

    # Reindex to a regular daily series, leaving gaps as NaN
    series = series.asfreq("D")
    return series


def _linear_regression_forecast(series: pd.Series, horizon_days: int) -> pd.DataFrame:
    """
    Simple linear trend forecast: returns date, forecast, lower, upper.
    """
    s = series.dropna()
    if len(s) < 3:
        raise ValueError("Not enough historical points for linear regression")

    x = np.arange(len(s)).reshape(-1, 1)
    y = s.values.astype(float)

    # y = a + b*t using manual OLS
    X = np.c_[np.ones_like(x), x]
    beta = np.linalg.pinv(X.T @ X) @ (X.T @ y)
    intercept, slope = beta

    last_idx = len(s) - 1
    future_idx = np.arange(last_idx + 1, last_idx + 1 + horizon_days)
    future_dates = s.index[-1] + pd.to_timedelta(np.arange(1, horizon_days + 1), "D")

    forecast_vals = intercept + slope * future_idx

    # Rough CI using residual stddev
    y_hat = intercept + slope * np.arange(len(s))
    resid = y - y_hat
    sigma = float(np.std(resid)) if len(resid) > 1 else 0.0
    ci = 1.96 * sigma

    lower = forecast_vals - ci
    upper = forecast_vals + ci

    return pd.DataFrame(
        {
            "date": future_dates,
            "forecast": forecast_vals,
            "lower": lower,
            "upper": upper,
        }
    )


def _arima_forecast(series: pd.Series, horizon_days: int) -> pd.DataFrame:
    """
    ARIMA(1,1,1) forecast with 95% CI.
    """
    s = series.dropna()
    if len(s) < 5:
        raise ValueError("Not enough historical points for ARIMA")

    model = ARIMA(s, order=(1, 1, 1))
    fit = model.fit()

    fc = fit.get_forecast(steps=horizon_days)
    fc_mean = fc.predicted_mean
    ci = fc.conf_int(alpha=0.05)

    future_dates = s.index[-1] + pd.to_timedelta(np.arange(1, horizon_days + 1), "D")

    return pd.DataFrame(
        {
            "date": future_dates,
            "forecast": fc_mean.values,
            "lower": ci.iloc[:, 0].values,
            "upper": ci.iloc[:, 1].values,
        }
    )


def _lgbm_quantile_forecast(series: pd.Series, horizon_days: int) -> pd.DataFrame:
    """
    LightGBM quantile regression (p10/p50/p90) with simple recursive rollout.
    Falls back to ValueError if not enough points or LightGBM not installed.
    """
    try:
        from lightgbm import LGBMRegressor  # type: ignore
    except Exception:
        raise ValueError("LightGBM not available; install lightgbm to use model='lgbm'")

    s = series.dropna().sort_index()
    if len(s) < 20:  # allow short series; still reject extremely small samples
        raise ValueError("Not enough historical points for LGBM (need >=6)")

    # Build lag/rolling features
    def build_features(vals: list[float], dates: list[pd.Timestamp]) -> pd.DataFrame:
        data = []
        for i in range(len(vals)):
            if i < 7:
                continue  # need enough history for lags/rolls
            window = np.array(vals[: i + 1], dtype=float)
            feat = {
                "lag1": window[-1],
                "lag2": window[-2],
                "lag3": window[-3],
                "lag7": window[-7],
                "roll_mean3": window[-3:].mean(),
                "roll_std3": window[-3:].std(ddof=0),
                "roll_mean7": window[-7:].mean(),
                "roll_std7": window[-7:].std(ddof=0),
                "dow_sin": np.sin(2 * np.pi * dates[i].dayofweek / 7.0),
                "dow_cos": np.cos(2 * np.pi * dates[i].dayofweek / 7.0),
            }
            data.append(feat)
        return pd.DataFrame(data)

    vals = s.values.tolist()
    dates = list(s.index)
    if not all(isinstance(d, pd.Timestamp) for d in dates):
        dates = list(pd.to_datetime(dates))

    X = build_features(vals, dates)
    y = np.array(vals[7:], dtype=float)  # align to feature rows

    def fit_quantile(alpha: float):
        model = LGBMRegressor(
            objective="quantile",
            alpha=alpha,
            n_estimators=200,
            learning_rate=0.05,
            max_depth=-1,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=5,
        )
        model.fit(X, y)
        return model

    m_p10 = fit_quantile(0.10)
    m_p50 = fit_quantile(0.50)
    m_p90 = fit_quantile(0.90)

    def predict_next(ctx_vals: list[float], date: pd.Timestamp):
        if len(ctx_vals) < 7:
            raise ValueError("Not enough context for prediction")
        window = np.array(ctx_vals[-7:], dtype=float)
        feat = pd.DataFrame(
            [
                {
                    "lag1": window[-1],
                    "lag2": window[-2],
                    "lag3": window[-3],
                    "lag7": window[0],
                    "roll_mean3": window[-3:].mean(),
                    "roll_std3": window[-3:].std(ddof=0),
                    "roll_mean7": window.mean(),
                    "roll_std7": window.std(ddof=0),
                    "dow_sin": np.sin(2 * np.pi * date.dayofweek / 7.0),
                    "dow_cos": np.cos(2 * np.pi * date.dayofweek / 7.0),
                }
            ]
        )
        p10 = float(m_p10.predict(feat)[0])
        p50 = float(m_p50.predict(feat)[0])
        p90 = float(m_p90.predict(feat)[0])
        return p10, p50, p90

    fc_rows = []
    ctx_vals = vals.copy()
    last_date = dates[-1]
    for i in range(1, horizon_days + 1):
        d = last_date + pd.Timedelta(days=i)
        p10, p50, p90 = predict_next(ctx_vals, d)
        ctx_vals.append(p50)  # recursive rollout uses median as next value
        fc_rows.append({"date": d, "forecast": p50, "lower": p10, "upper": p90})

    return pd.DataFrame(fc_rows)


def build_forecast(history_df: pd.DataFrame, req: ForecastRequest) -> Dict[str, Any]:
    """
    Main entry point used by EnhancedVibExtractor.
    """
    col_map = {
        ("rms_vel", "d1"): "rms_vel_d1",
        ("rms_vel", "d2"): "rms_vel_d2",
        ("rms_vel", "d3"): "rms_vel_d3",
        ("rms_acc", "d1"): "rms_acc_d1",
        ("rms_acc", "d2"): "rms_acc_d2",
        ("rms_acc", "d3"): "rms_acc_d3",
    }

    key = (req.metric, req.direction)
    if key not in col_map:
        raise ValueError(f"Unsupported metric/direction {key}")

    value_col = col_map[key]
    series = _prepare_daily_series(history_df, value_col)

    fallback_reason: str | None = None
    if req.model == "lgbm":
        try:
            fc_df = _lgbm_quantile_forecast(series, req.horizon_days)
            model_used = "lgbm"
        except Exception as e:
            # Fallback gracefully to linear if LGBM unavailable or insufficient data
            fc_df = _linear_regression_forecast(series, req.horizon_days)
            model_used = "linear_fallback"
            fallback_reason = str(e)
    elif req.model == "linear":
        fc_df = _linear_regression_forecast(series, req.horizon_days)
        model_used = "linear"
    elif req.model == "arima":
        fc_df = _arima_forecast(series, req.horizon_days)
        model_used = "arima"
    else:
        raise ValueError(f"Unknown model: {req.model}")

    history = [
        {"date": d.strftime("%Y-%m-%d"), "value": (float(v) if pd.notna(v) else None)}
        for d, v in series.items()
    ]

    forecast = [
        {
            "date": row.date.strftime("%Y-%m-%d"),
            "forecast": float(row.forecast),
            "lower": float(row.lower),
            "upper": float(row.upper),
        }
        for _, row in fc_df.iterrows()
    ]

    result = {
        "sn": req.sn,
        "metric": req.metric,
        "direction": req.direction,
        "model": model_used,
        "horizon_days": req.horizon_days,
        "history": history,
        "forecast": forecast,
    }
    if fallback_reason:
        result["fallback_reason"] = fallback_reason
    return result
