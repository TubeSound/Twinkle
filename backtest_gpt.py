import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN


def read_data(path):
    df = pd.read_csv(path)
    df['jst'] = pd.to_datetime(df['jst'], format='ISO8601')
    return df




# === データ読み込み ===
df = read_data("./data/nsdq_tick.csv")
df = df[['jst', 'bid']].rename(columns={'bid': 'price'})
df = df.dropna().sort_values('jst')
df.set_index('jst', inplace=True)

# === 時間範囲を限定（最初の10分間） ===
start_time = df.index.min()
end_time = start_time + pd.Timedelta(minutes=10)
df_small = df.loc[start_time:end_time]

# === ピボット検出：スライディング60秒 & 中心11点で山/谷を検出 ===
timestamps = df_small.index
prices = df_small['price'].values
window_radius = 30
center_radius = 5

pivot_times = []
pivot_prices = []
pivot_types = []

for i in range(window_radius, len(prices) - window_radius):
    full_window = prices[i - window_radius:i + window_radius + 1]
    center_range = prices[i - center_radius:i + center_radius + 1]

    if np.max(center_range) == np.max(full_window):
        pivot_times.append(timestamps[i].replace(microsecond=0))
        pivot_prices.append(prices[i])
        pivot_types.append('HIGH')
    elif np.min(center_range) == np.min(full_window):
        pivot_times.append(timestamps[i].replace(microsecond=0))
        pivot_prices.append(prices[i])
        pivot_types.append('LOW')

# === 横軸（秒）でクラスタリング（DBSCAN）===
pivot_df = pd.DataFrame({
    'time': pivot_times,
    'price': pivot_prices,
    'type': pivot_types
})
pivot_df['timestamp_sec'] = pivot_df['time'].astype('int64') // 10**9

db = DBSCAN(eps=5, min_samples=1).fit(pivot_df[['timestamp_sec']])
pivot_df['cluster'] = db.labels_

# === 各クラスタから代表点を抽出 ===
rep_times = []
rep_prices = []
rep_types = []

for cluster_id in pivot_df['cluster'].unique():
    group = pivot_df[pivot_df['cluster'] == cluster_id]
    typ = group['type'].iloc[0]

    if typ == 'HIGH':
        idx = group['price'].idxmax()
    elif typ == 'LOW':
        idx = group['price'].idxmin()
    else:
        continue

    rep_times.append(pivot_df.loc[idx, 'time'])
    rep_prices.append(pivot_df.loc[idx, 'price'])
    rep_types.append(typ)

# === プロット ===
plt.figure(figsize=(14, 6))
plt.plot(timestamps, prices, color='black', linewidth=0.8, label='Tick Price (bid)')

for t, p, typ in zip(rep_times, rep_prices, rep_types):
    if typ == 'HIGH':
        plt.scatter(t, p, color='orange', marker='^', s=70)
    elif typ == 'LOW':
        plt.scatter(t, p, color='blue', marker='v', s=70)

plt.title("Clustered Pivots (Time-Axis DBSCAN, 5s) on Tick Chart")
plt.xlabel("Time")
plt.ylabel("Price")
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
