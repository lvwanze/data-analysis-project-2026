# -*- coding: utf-8 -*-
"""
1 项目启动与数据接入（出租车类比版）

分析对象：原始 CSV 的接入与整体画像
    步骤 1  环境自检（版本 / 开源字体 / 数据文件）
    步骤 2  加载原始数据（100,000 行 × 9 列）
    步骤 3  数据画像 profile()：每个字段的类型、缺失、唯一值
    步骤 4  时间戳换算为波尔图本地时间（UTC -> Europe/Lisbon，夏令时）
    步骤 5  按天统计趟次量 -> ads_daily_trips
    步骤 6  按小时统计趟次量 / 活跃车辆 -> ads_hourly
    步骤 7  出图：D1_01 各日趟次分布、D1_02 小时双面板
"""
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

import taxi_common as tc

SCRIPT = "项目启动与数据接入"
tc.AUTHOR = "高帅龙 2428324009"
SOURCE = tc.source_line(SCRIPT)

# --------------------------------------------------------------- 步骤 1 环境自检
print("=" * 60, "\n步骤 1 环境自检\n", "=" * 60)
tc.env_self_check()
tc.apply_theme()

# --------------------------------------------------------------- 步骤 2 加载原始数据
print("\n步骤 2 加载原始数据")
raw = tc.load_raw()
print(f"数量核对：{len(raw):,} 行 × {raw.shape[1]} 列（含解析列）")
print("原始 9 列：", [c for c in raw.columns if c in
      ["TRIP_ID", "CALL_TYPE", "ORIGIN_CALL", "ORIGIN_STAND", "TAXI_ID",
       "TIMESTAMP", "DAY_TYPE", "MISSING_DATA", "POLYLINE"]])

# --------------------------------------------------------------- 步骤 3 数据画像
print("\n步骤 3 数据画像")
orig_cols = ["TRIP_ID", "CALL_TYPE", "ORIGIN_CALL", "ORIGIN_STAND", "TAXI_ID",
             "TIMESTAMP", "DAY_TYPE", "MISSING_DATA", "POLYLINE"]
prof = tc.profile(raw[orig_cols])
print(prof.to_string(index=False))
# 对账要点（类比信令版 tmp0/tmp1 全空、imsi 缺 15）
print("\n对账：ORIGIN_CALL 结构性缺失（仅 A 型有）；ORIGIN_STAND 仅 B/C 型有；"
      "DAY_TYPE / MISSING_DATA 在本 100k 子集为单值")

# --------------------------------------------------------------- 步骤 4 时间戳换算
print("\n步骤 4 时间戳换算")
print("UTC   ：", raw["ts_utc"].min(), "~", raw["ts_utc"].max())
print("里斯本：", raw["ts_local"].min(), "~", raw["ts_local"].max())
print("（7 月为夏令时 UTC+1；类比信令版 +8 小时）")

# --------------------------------------------------------------- 步骤 5 按天统计
print("\n步骤 5 按天统计")
daily = raw.groupby("date").agg(
    趟次数=("TRIP_ID", "size"),
    活跃车辆=("TAXI_ID", "nunique"),
).reset_index()
daily["趟次占比"] = daily["趟次数"] / daily["趟次数"].sum()
daily["车均趟次"] = (daily["趟次数"] / daily["活跃车辆"]).round(1)
print(daily.to_string(index=False))

top = daily.sort_values("趟次数", ascending=False).iloc[0]
first, last = daily.iloc[0], daily.iloc[-1]
print(f"\n最高峰 {top['date']}：{top['趟次数']:,} 趟（{top['趟次占比']:.1%}）；"
      f"首尾日 {first['趟次数']:,}/{last['趟次数']:,}，末日为不完整天")

# --------------------------------------------------------------- 步骤 6 按小时统计
print("\n步骤 6 按小时统计")
hourly = raw.groupby("hour").agg(
    趟次数=("TRIP_ID", "size"),
    活跃车辆=("TAXI_ID", "nunique"),
).reindex(range(24), fill_value=0)
print(hourly.to_string())
peak_h = int(hourly["趟次数"].idxmax())
low_h = int(hourly["趟次数"].idxmin())
print(f"\n高峰 {peak_h} 点 {hourly.loc[peak_h, '趟次数']:,} 趟；"
      f"低谷 {low_h} 点 {hourly.loc[low_h, '趟次数']:,} 趟")

# =============================================================== 图 D1_01 各日趟次分布
# 横向条形（按天），类比信令版三面板共用日期轴
fig, ax = plt.subplots(figsize=(10, 7), dpi=100)
colors = [tc.GRAY if d == daily["date"].iloc[-1] else tc.MAIN for d in daily["date"]]
bars = ax.barh(daily["date"], daily["趟次数"], color=colors, height=0.62)
vmax = daily["趟次数"].max()
for i, (_, r) in enumerate(daily.iterrows()):
    ax.text(r["趟次数"] + vmax * 0.01, i,
            f"{r['趟次数']:,}（{r['趟次占比']:.1%}）",
            va="center", fontsize=8.5, color=tc.INK)
ax.set_xlim(0, vmax * 1.18)
ax.set_xlabel("趟次数")
ax.grid(axis="x", color=tc.GRAY, lw=0.6)
ax.set_yticks(range(len(daily)))
ax.set_yticklabels(list(daily["date"]), fontsize=8.5)
tc.finish(
    fig,
    f"日均 {daily['趟次数'].mean():,.0f} 趟，末日（{daily['date'].iloc[-1]}）不完整",
    "2013-07-01~21 按天统计发车趟次；灰色 = 当天数据不完整（仅到 20 点）；趟次=一条 POLYLINE",
    SOURCE, "D1_01_各日趟次分布.png")

# =============================================================== 图 D1_02 小时双面板
# 上下两个面板共用横轴（不用双 Y 轴，规范第 5 节）
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6.6), sharex=True)
fig.subplots_adjust(left=0.09, right=0.97, top=0.80, bottom=0.10, hspace=0.30)
for ax in (ax1, ax2):
    ax.axvspan(0, 5, color=tc.GRAY, alpha=0.30, lw=0, zorder=0)   # 夜间
ax1.plot(hourly.index, hourly["活跃车辆"], color=tc.MAIN, lw=2.2)
ax1.fill_between(hourly.index, hourly["活跃车辆"], color=tc.MAIN_LIGHT, alpha=0.5)
ax1.set_ylabel("活跃车辆（辆）", fontsize=10)
ax1.annotate(f"{peak_h} 点 {hourly.loc[peak_h,'活跃车辆']:,} 辆",
             (peak_h, hourly.loc[peak_h, "活跃车辆"]),
             xytext=(-30, 12), textcoords="offset points", color=tc.MAIN_DARK)

ax2.plot(hourly.index, hourly["趟次数"], color=tc.GRAY_DARK, lw=2)
ax2.plot(hourly.loc[4:peak_h].index, hourly.loc[4:peak_h, "趟次数"],
         color=tc.MAIN, lw=2.6)
ax2.set_ylabel("趟次数（趟）", fontsize=10)
ax2.set_xlabel("波尔图本地小时（0–23 点）", fontsize=10)
ax2.set_xticks(range(0, 24, 2))
ax2.annotate(f"{peak_h} 点 {hourly.loc[peak_h,'趟次数']:,} 趟",
             (peak_h, hourly.loc[peak_h, "趟次数"]),
             xytext=(-30, 12), textcoords="offset points", color=tc.MAIN_DARK)
ax2.annotate(f"{low_h} 点低谷 {hourly.loc[low_h,'趟次数']:,} 趟",
             (low_h, hourly.loc[low_h, "趟次数"]),
             xytext=(8, -6), textcoords="offset points", color=tc.MUTED)
tc.finish(
    fig,
    f"发车趟次 {peak_h} 点达峰 {hourly.loc[peak_h,'趟次数']:,} 趟，{low_h} 点最少",
    "21 天全部数据按本地小时聚合；上=活跃车辆，下=发车趟次；灰底=0–5 点夜间；主色段=早间爬升",
    SOURCE, "D1_02_小时趟次分布.png")

print("\nD1 完成：图片已保存至", tc.IMAGE_DIR)
