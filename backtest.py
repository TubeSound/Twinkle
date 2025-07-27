# backtest.py
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dateutil import tz
JST = tz.gettz('Asia/Tokyo')
UTC = tz.gettz('utc')

from technical import detect_ema_cross, detect_pivots, detect_pivot_points, volatility, sma_sec


def read_data(path):
    df = pd.read_csv(path)
    df['jst'] = pd.to_datetime(df['jst'], format='ISO8601')
    return df

    
def slice(df, t0, t1):
    df1 = df[df['jst'] > t0]
    df2 = df1[df1['jst'] < t1]
    return df2
    
def plot1(ax, timestamps, prices, ema_quick, ema_fast, ema_slow, golden_cross_idx, dead_cross_idx):
    ax.plot(timestamps, prices, alpha=0.6, color='blue') 
    ax.plot(timestamps, ema_quick, color='red') 
    ax.plot(timestamps, ema_fast, color='green') 
    ax.plot(timestamps, ema_slow, color='orange')
    for index in golden_cross_idx: 
         ax.scatter(timestamps[index], ema_fast[index], color='red', marker='o', alpha=0.3, s=200)
    for index in dead_cross_idx: 
         ax.scatter(timestamps[index], ema_fast[index], color='green', marker='o', alpha=0.3, s=200)
    
def plot2(ax, pivots):
    times = pivots[0]
    prices = pivots[1]
    status = pivots[2]
    for t, price, s in zip(times, prices, status):
        if s == 1:
            ax.scatter(t, price, color='red', marker='v', alpha=0.4, s=80)
        elif s == -1:
            ax.scatter(t, price, color='green', marker='^', alpha=0.4, s=80)

def analyze_tick(png_path, csv_path):
    df = read_data(csv_path)

    # NumPy配列として取り出す
    timestamps_np = df['jst'].tolist()
    
    timestamps = df["jst"].values
    prices = df['bid'].to_numpy()
    ema_fast, ema_mid, ema_slow, golden_cross_idx, dead_cross_idx = detect_ema_cross(timestamps_np, prices, period_fast_sec=60 * 5, period_mid_sec=60 * 13, period_slow_sec=60 * 30)


    fig, axes = plt.subplots(2, 1, figsize=(14, 6))
    plot1(axes[0], timestamps_np, prices, ema_fast, ema_mid, ema_slow, golden_cross_idx, dead_cross_idx)

    
    # ピボット検出
    pivots = detect_pivot_points(timestamps_np, ema_fast, slide_term_sec=60 * 5)
    
    plot2(axes[0], pivots)
    
    
    volt_times, volt_values = volatility(timestamps, prices)
    axes[1].plot(timestamps_np, volt_values, color='blue')

    ma = sma_sec(timestamps, volt_values, window_sec=60 * 15)
    axes[1].plot(timestamps_np, ma, color='red')
    
    axes[1].set_ylim(0, 0.1)
    axes[1].axhline(y=0.02, color='yellow')
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(png_path)
    plt.close()
    
    """
    timestamp_sec = np.array([t.timestamp() for t in timestamps])
    df = pd.DataFrame({'timestamp': timestamps, 'time': timestamp_sec,'bid': prices, 'ema_fast': ema_fast, 'ema_mid': ema_mid, 'ema_slow': ema_slow})
    path = png_path[:-3] + 'csv'
    df.to_csv(path, index=False)
    """
    
def main():
    symbol='NSDQ'
    year = 2025
    month =  4
    mstr = str(month).zfill(2)
    files = glob.glob(f"../tickdata/{symbol}/{year}-{mstr}/*")
    for file in files:
        analyze(file, symbol, year, month)
    
def analyze(csv_path, symbol, year, month): 
    mstr = str(month).zfill(2)
    dirpath = f'./chart/{symbol}/{year}/{mstr}'
    os.makedirs(dirpath, exist_ok=True)
    _, filename = os.path.split(csv_path)
    name, ext = os.path.splitext(filename)
    analyze_tick(os.path.join(dirpath, name + '.png'), csv_path)
    
    

if __name__ == "__main__":
    #os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()