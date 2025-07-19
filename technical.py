# technical.py

import numpy as np
from sklearn.cluster import DBSCAN


def detect_pivot_points(highs, lows, lookback=7):
    """
    highs, lows: numpy array
    戻り値: numpy 配列（None / 'PIVOT_HIGH' / 'PIVOT_LOW'）
    """
    n = len(highs)
    pivot = np.full(n, np.nan)

    for i in range(lookback, n - lookback):
        window_highs = highs[i - lookback:i + lookback + 1]
        window_lows = lows[i - lookback:i + lookback + 1]
        center = lookback

        if np.all(highs[i] > window_highs[np.arange(len(window_highs)) != center]):
            pivot[i] = 1
        elif np.all(lows[i] < window_lows[np.arange(len(window_lows)) != center]):
            pivot[i] = -1

    return pivot

def detect_dow_reversal(highs, lows, pivots):
    """
    pivots: detect_center_pivots()の出力を使う
    高値・安値の切り上げ・切り下げによって転換点を判定
    """
    n = len(highs)
    dow_trend = np.full(n, np.nan)
    swing_points = []

    # ピボット点だけ抽出
    for i, p in enumerate(pivots):
        if p in ['PIVOT_HIGH', 'PIVOT_LOW']:
            swing_points.append((i, p))

    if len(swing_points) < 3:
        return dow_trend

    for i in range(2, len(swing_points)):
        idx1, type1 = swing_points[i - 2]
        idx2, type2 = swing_points[i - 1]
        idx3, type3 = swing_points[i]

        # HIGH - LOW - HIGH → 高値＆安値ともに切り上げで UP_TURN
        if type1 == 'PIVOT_HIGH' and type2 == 'PIVOT_LOW' and type3 == 'PIVOT_HIGH':
            if highs[idx3] > highs[idx1] and lows[idx2] > lows[idx1]:
                dow_trend[idx3] = 1

        # LOW - HIGH - LOW → 高値＆安値ともに切り下げで DOWN_TURN
        elif type1 == 'PIVOT_LOW' and type2 == 'PIVOT_HIGH' and type3 == 'PIVOT_LOW':
            if highs[idx2] < highs[idx1] and lows[idx3] < lows[idx1]:
                dow_trend[idx3] = -1

    return dow_trend



def detect_pivot_tick(timestamps, prices, slide_term_sec=60, center_sec=5):
    pivots = detect_pivot_tick_points(timestamps, prices, slide_term_sec, center_sec)
    pivot_times, pivot_prices, pivot_types = pivots
    return extract_representative_pivots(pivot_times, pivot_prices, pivot_types, timestamps, cluster_eps_sec=5, min_cluster_size=5)

def detect_pivot_tick_points(timestamps, prices, slide_term_sec=60, center_sec=5):
    window_radius = int(slide_term_sec / 2)
 
    pivot_times = []
    pivot_prices = []
    pivot_types = []

    # === ピボット検出ロジック ===
    for i in range(window_radius, len(prices) - window_radius):
        full_window = prices[i - window_radius:i + window_radius + 1]
        center_range = prices[i - center_sec:i + center_sec + 1]

        if np.max(center_range) == np.max(full_window):
            pivot_times.append(timestamps[i].replace(microsecond=0))
            pivot_prices.append(prices[i])
            pivot_types.append(1)

        elif np.min(center_range) == np.min(full_window):
            pivot_times.append(timestamps[i].replace(microsecond=0))
            pivot_prices.append(prices[i])
            pivot_types.append(-1)
    return pivot_times, pivot_prices, pivot_types

def extract_representative_pivots(pivot_times, pivot_prices, pivot_types, timestamps, cluster_eps_sec=5, min_cluster_size=5):
    #timestamp_sec = np.array(timestamps.astype("datetime64[s]")).astype(np.int64)
    timestamp_sec = np.array([t.timestamp() for t in pivot_times])
    db = DBSCAN(eps=cluster_eps_sec, min_samples=1).fit(timestamp_sec.reshape(-1, 1))
    reps_time, reps_price, reps_type = [], [], []
    for cluster_id in np.unique(db.labels_):
        mask = db.labels_ == cluster_id
        if np.sum(mask) < min_cluster_size:
            continue

        group_prices = np.array(pivot_prices)[mask]
        group_types = np.array(pivot_types)[mask]
        group_times = np.array(pivot_times)[mask]

        if group_types[0] == 1:
            idx = np.argmax(group_prices)
        elif group_types[0] == -1:
            idx = np.argmin(group_prices)
        else:
            continue

        reps_time.append(group_times[idx])
        reps_price.append(group_prices[idx])
        reps_type.append(group_types[0])

    return reps_time, reps_price, reps_type