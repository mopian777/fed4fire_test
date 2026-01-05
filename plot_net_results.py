#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
读取 net_results_client*.csv，并按 round 画出 baseline vs fedavg 的对比曲线。

CSV 格式假定为：
mode,client_id,server_host,round,sent,received,loss,rtt_min_ms,rtt_avg_ms,rtt_max_ms,delay_avg_ms
"""

import csv
import sys
import os
from collections import defaultdict

import matplotlib.pyplot as plt


def load_results(csv_path, metric_col):
    """
    从 CSV 加载数据，按 mode 分组，返回:
    data[mode] = (round_list, metric_list)
    """
    data_raw = defaultdict(list)

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mode = row.get("mode", "").strip()
            if not mode:
                continue
            try:
                rnd = int(row["round"])
            except (KeyError, ValueError):
                continue

            try:
                val = float(row[metric_col])
            except (KeyError, ValueError):
                # 该列不存在或无法转成 float，跳过
                continue

            data_raw[mode].append((rnd, val))

    # 按轮次排序
    data = {}
    for mode, lst in data_raw.items():
        lst_sorted = sorted(lst, key=lambda x: x[0])
        rounds = [x[0] for x in lst_sorted]
        vals = [x[1] for x in lst_sorted]
        data[mode] = (rounds, vals)

    return data


def plot_baseline_vs_fedavg(csv_path, metric_col="rtt_avg_ms", save_path=None):
    """
    画一张图：横轴 round，纵轴 metric_col，一条 baseline，一条 fedavg。
    """
    if not os.path.exists(csv_path):
        print("CSV file not found:", csv_path)
        return

    data = load_results(csv_path, metric_col)

    if not data:
        print("No data loaded from:", csv_path)
        return

    plt.figure()

    # 优先画 baseline 和 fedavg，其它 mode（如果有）可以一并画出
    for mode in sorted(data.keys()):
        rounds, vals = data[mode]
        label = mode
        plt.plot(rounds, vals, marker="o", label=label)

    plt.xlabel("Round")
    plt.ylabel(metric_col)
    plt.title("Network metric: {} (baseline vs fedavg)".format(metric_col))
    plt.grid(True)
    plt.legend()

    if save_path is None:
        # 默认保存到同目录，文件名加后缀
        base, ext = os.path.splitext(csv_path)
        save_path = base + "_{}.png".format(metric_col)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print("Figure saved to:", save_path)


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 {} <csv_path> [metric_col]".format(sys.argv[0]))
        print("  default metric_col = rtt_avg_ms")
        print("  other options: delay_avg_ms, loss, rtt_min_ms, rtt_max_ms")
        sys.exit(1)

    csv_path = sys.argv[1]
    if len(sys.argv) >= 3:
        metric_col = sys.argv[2]
    else:
        metric_col = "rtt_avg_ms"

    plot_baseline_vs_fedavg(csv_path, metric_col=metric_col)


if __name__ == "__main__":
    main()
