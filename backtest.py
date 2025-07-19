# backtest.py
import os
import pandas as pd
import matplotlib.pyplot as plt
from dateutil import tz
JST = tz.gettz('Asia/Tokyo')
UTC = tz.gettz('utc')

from technical import detect_dow_reversal, detect_pivot_points, detect_pivot_tick



def read_data(path):
    df = pd.read_csv(path)
    df['jst'] = pd.to_datetime(df['jst'], format='ISO8601')
    return df

    
def slice(df, t0, t1):
    df1 = df[df['jst'] > t0]
    df2 = df1[df1['jst'] < t1]
    return df2
    
    
def plot(title, df, pivots):
    pivot_times, pivot_prices, pivot_types = pivots
    n = len(pivot_times)
    size = int(n / 20)
    i0 = 0
    for i in range(size):
        i0 = i * size
        i1 = i0 + size
        times = pivot_times[i0:i1]
        prices = pivot_prices[i0:i1]
        types = pivot_types[i0:i1]
        df2 = slice(df, times[0], times[-1])
        
        fig, ax = plt.subplots(1, 1, figsize=(14, 6))
        ax.plot(df2['jst'], df2['bid'], color='blue')
        
        for t, p, typ in zip(times, prices, types):
            if typ == 1:
                ax.scatter(t, p, color='red', marker='v', alpha=0.4, s=80)
            elif typ == -1:
                ax.scatter(t, p, color='green', marker='^', alpha=0.4, s=80)

        ax.set_title(f"Clustered Pivots (Time-Axis DBSCAN, 5s) - {i}")
        ax.set_xlabel("Time")
        ax.set_ylabel("Price")
        ax.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()



def main():
    df = read_data('./data/nsdq_tick.csv')

    # NumPy配列として取り出す
    timestamps = df['jst'].tolist()
    prices = df['bid'].to_numpy()

    # ピボット検出
    pivots = detect_pivot_tick(timestamps, prices, slide_term_sec=120)
    plot("Pivot", df, pivots)
    

if __name__ == "__main__":
    #os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()