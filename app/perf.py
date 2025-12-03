# backend/app/perf.py
import time, math
from functools import wraps
from typing import Any, Tuple, Dict, Callable, Optional, List

# --- TTL memo --------------------------------------------------
_cache: Dict[Tuple[str, Tuple, Tuple], Tuple[float, Any]] = {}
def ttl_cache(seconds: int):
    def deco(fn: Callable):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            key = (fn.__name__, tuple(args), tuple(sorted(kwargs.items())))
            now = time.time()
            hit = _cache.get(key)
            if hit and (now - hit[0] < seconds):
                return hit[1]
            out = fn(*args, **kwargs)
            _cache[key] = (now, out)
            return out
        return wrapper
    return deco

# --- Fast downsamplers -----------------------------------------
def _downsample_even(x: List[float], y: List[float], max_n: int) -> Tuple[List[float], List[float]]:
    n = min(len(x), len(y))
    if n <= max_n: return x[:n], y[:n]
    step = n / max_n
    xs, ys = [], []
    i = 0.0
    while len(xs) < max_n and int(i) < n:
        idx = int(i)
        xs.append(x[idx]); ys.append(y[idx])
        i += step
    return xs, ys

def _lttb(y: List[float], max_n: int) -> List[int]:
    # Largest-Triangle-Three-Buckets: return indices to keep
    n = len(y)
    if max_n >= n or max_n < 3: return list(range(n))
    bucket = (n - 2) / (max_n - 2)
    idx = [0]
    a = 0
    for i in range(0, max_n - 2):
        start = int(math.floor((i + 1) * bucket)) + 1
        end   = int(math.floor((i + 2) * bucket)) + 1
        end = min(end, n)
        avg_x = (start + end - 1) / 2
        avg_y = sum(y[start:end]) / max(1, (end - start))
        seg_start = int(math.floor(i * bucket)) + 1
        seg_end   = int(math.floor((i + 1) * bucket)) + 1
        seg_end = min(seg_end, n-1)
        best = seg_start
        best_area = -1.0
        for j in range(seg_start, seg_end):
            area = abs((a - avg_x) * (y[j] - y[a]) - (a - j) * (avg_y - y[a]))
            if area > best_area:
                best_area = area
                best = j
        idx.append(best); a = best
    idx.append(n - 1)
    return idx

def downsample_waveform(signal: list, max_n: int) -> list:
    # LTTB keeps peaks/transients; great for time series
    keep = _lttb(signal, max_n)
    return [signal[i] for i in keep]

def downsample_xy(freqs: list, amps: list, max_n: int) -> Tuple[list, list]:
    # spectra are dense; even sampling is fine (UI uses lines)
    return _downsample_even(freqs, amps, max_n)
