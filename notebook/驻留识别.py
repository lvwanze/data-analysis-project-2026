# -*- coding: utf-8 -*-
"""
6 驻留识别（出租车类比版）

分析对象：从 GPS 轨迹里识别候客停留（类比信令驻留）
    1 从轨迹到停留点：锚点法（距离切段 + 最短时长）
    2 一辆车的一天：轨迹 + 停留点（类比"一个人的一天"）
    3 GPS 漂移/乒乓：车没动、点在小范围跳（类比基站乒乓切换）
    4 停留识别封装
    5 敏感性分析：半径 100–500m、最短时长 5–30min（类比信令 300–1000m/15–60min）
    6 停留表
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import taxi_common as tc

SCRIPT = "驻留识别"
tc.AUTHOR = "薛宏博 2428324276"
SOURCE = tc.source_line(SCRIPT)
tc.apply_theme()

dwd = pd.read_pickle(tc.CACHE_DIR / "dwd.pkl")

# 点间 haversine 距离（米）：返回 n-1 个
def seg_dist(poly):
    arr = np.asarray(poly, dtype=float)
    if len(arr) < 2:
        return np.array([])
    lng, lat = np.radians(arr[:,0]), np.radians(arr[:,1])
    dl, dp = np.diff(lng), np.diff(lat)
    a = np.sin(dp/2)**2 + np.cos(lat[:-1])*np.cos(lat[1:])*np.sin(dl/2)**2
    return 6371000*2*np.arcsin(np.sqrt(a))

# ---------------------------------------------------------------- 1 锚点法停留识别
def detect_stops(poly, radius_m=200, min_minutes=10):
    """
    锚点法：从 i 起，把离锚点 < radius 的连续点并入；
    段时长（点数*15s）≥ min_minutes 才记为停留；否则锚点前移 1。
    类比信令版"按距离切段 + 最短停留时长"。
    """
    arr = np.asarray(poly, dtype=float)
    n = len(arr)
    stops = []
    i = 0
    while i < n:
        anchor = arr[i]
        # 向量化：后续点到锚点距离
        dlng = np.radians(arr[:,0] - anchor[0])
        dlat = np.radians(arr[:,1] - anchor[1])
        a = np.sin(dlat/2)**2 + np.cos(np.radians(anchor[1]))*np.cos(
            np.radians(arr[:,1]))*np.sin(dlng/2)**2
        d = 6371000*2*np.arcsin(np.sqrt(a))
        beyond = np.flatnonzero(d[i:] >= radius_m)
        j = (i + beyond[0]) if len(beyond) else n
        dur_min = (j - i) * tc.GPS_INTERVAL_SEC / 60
        if dur_min >= min_minutes:
            seg = arr[i:j]
            stops.append({
                "lng": float(np.median(seg[:,0])),
                "lat": float(np.median(seg[:,1])),
                "点数": j - i, "时长分": round(dur_min, 1),
            })
            i = j
        else:
            i += 1
    return stops

# ---------------------------------------------------------------- 2 一辆车的一天
# 选一趟点数较多、且能识别出停留的车（找轨迹内有停滞段的车）
example_taxi = None; example_day = None
# 先在样本里找：一辆车某天有多趟且某趟含停留
cand = dwd[dwd["n_pts"] >= 60].sample(2000, random_state=1)
for _, r in cand.iterrows():
    st = detect_stops(r["poly"], 200, 10)
    if st:
        example_taxi = r["TAXI_ID"]
        example_day = r["date"]
        example_trip = r
        break
day_trips = dwd[(dwd["TAXI_ID"]==example_taxi) & (dwd["date"]==example_day)]
print(f"示例车辆 {example_taxi} 在 {example_day} 共 {len(day_trips)} 趟")
all_stops = []
for _, r in day_trips.iterrows():
    for s in detect_stops(r["poly"], 200, 10):
        s["TRIP_ID"] = r["TRIP_ID"]
        all_stops.append(s)
stops_df = pd.DataFrame(all_stops)
print(f"当天识别停留 {len(stops_df)} 处")
if len(stops_df):
    print(stops_df[["lng","lat","时长分","点数"]].to_string(index=False))

# ---------------------------------------------------------------- 3 GPS 漂移：停留点内散布
print("\n" + "=" * 60, "\nGPS 漂移：停滞段内点到中心的散布\n", "=" * 60)
# 对有停留的趟次，算停留段内点到中心距离；对比正常行驶段
def spread_inside(poly, radius_m=200, min_minutes=10):
    arr = np.asarray(poly, dtype=float)
    out = []
    for s in detect_stops(poly, radius_m, min_minutes):
        # 找属于该停留的点（简化：离中心近的点）
        d = seg_dist(poly)
        dlng = np.radians(arr[:,0]-s["lng"])
        dlat = np.radians(arr[:,1]-s["lat"])
        a = np.sin(dlat/2)**2 + np.cos(np.radians(s["lat"]))*np.cos(
            np.radians(arr[:,1]))*np.sin(dlng/2)**2
        dd = 6371000*2*np.arcsin(np.sqrt(a))
        members = dd < radius_m
        out.append(dd[members])
    return out

sp = []
for _, r in cand.head(400).iterrows():
    sp += spread_inside(r["poly"])
if sp:
    drift = np.concatenate(sp)
    print(f"停留段 GPS 漂移：中位 {np.median(drift):.0f} m，P95 {np.quantile(drift,.95):.0f} m")

# 正常点间距离（对比）
normal = np.concatenate([seg_dist(p) for p in cand.head(200)["poly"]])
print(f"正常行驶相邻点距离：中位 {np.median(normal):.0f} m")

# ---------------------------------------------------------------- 5 敏感性分析
print("\n" + "=" * 60, "\n敏感性分析（采样 15,000 趟）\n", "=" * 60)
sens_sample = dwd.sample(15000, random_state=2)
radii = [100, 200, 300, 500]
min_mins = [5, 10, 15, 30]
# 固定时长 10min，变半径
by_radius = {}
for r in radii:
    cnt = sum(len(detect_stops(p, r, 10)) for p in sens_sample["poly"])
    by_radius[r] = cnt / sens_sample["TAXI_ID"].nunique()
# 固定半径 200m，变时长
by_min = {}
for m in min_mins:
    cnt = sum(len(detect_stops(p, 200, m)) for p in sens_sample["poly"])
    by_min[m] = cnt / sens_sample["TAXI_ID"].nunique()
print("人均停留点（变半径，时长10min）：", {k: round(v,2) for k,v in by_radius.items()})
print("人均停留点（变时长，半径200）：", {k: round(v,2) for k,v in by_min.items()})
# 写停留表
if len(stops_df):
    stops_df.to_pickle(tc.CACHE_DIR / "dws_stops.pkl")

# =============================================================== 图 D6_01 一辆车的一天
fig, ax = plt.subplots(figsize=(9, 9), dpi=100)
for _, r in day_trips.iterrows():
    arr = np.asarray(r["poly"])
    ax.plot(arr[:,0], arr[:,1], lw=1.3, color=tc.GRAY_DARK, alpha=.7, zorder=2)
    ax.scatter(arr[0,0], arr[0,1], s=22, color=tc.MAIN, zorder=3)   # 起点
    ax.scatter(arr[-1,0], arr[-1,1], s=40, marker="^", color=tc.GREEN, zorder=3)
if len(stops_df):
    ax.scatter(stops_df["lng"], stops_df["lat"], s=130, marker="*",
               facecolor=tc.RED, edgecolor="white", zorder=5, label="停留点")
ax.set_aspect(1/np.cos(np.radians(41.15)))
ax.set_xlabel("经度"); ax.set_ylabel("纬度")
ax.legend(fontsize=9, loc="best")
ax.text(0.01, 0.01, "蓝圆=各趟起点；绿三角=终点；红星=停留点",
        transform=ax.transAxes, fontsize=8.5, color=tc.MUTED)
tc.finish(
    fig,
    f"车辆 {example_taxi}（{example_day}）{len(day_trips)} 趟、识别 {len(stops_df)} 处停留",
    "当天该车辆全部载客轨迹；停留=轨迹内连续点 200m 内、时长≥10min；轨迹间等待不在 POLYLINE 内",
    SOURCE, "D6_01_一辆车的一天.png")

# =============================================================== 图 D6_02 GPS 漂移 vs 行驶
fig, ax = plt.subplots(figsize=(10, 5.2), dpi=100)
ax.hist(normal.clip(0,800), bins=40, color=tc.MAIN_MID, alpha=.7,
        label="正常行驶相邻点", edgecolor="white")
if sp:
    ax.hist(drift.clip(0,800), bins=40, color=tc.RED, alpha=.6,
            label="停留段 GPS 漂移", edgecolor="white")
ax.set_xlabel("点间 / 点到停留中心距离（米）")
ax.set_ylabel("频次")
ax.legend(fontsize=9)
ax.grid(axis="y", color=tc.GRAY, lw=.6)
tc.finish(
    fig,
    f"停留段 GPS 漂移中位 {np.median(drift):.0f}m，远小于行驶点间距中位 {np.median(normal):.0f}m",
    "车未动时 GPS 在小范围跳变（漂移），类似信令基站乒乓；半径 200m 可吸收漂移",
    SOURCE, "D6_02_GPS漂移.png")

# =============================================================== 图 D6_03 敏感性
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8), dpi=100, sharey=True)
ax1.plot(radii, [by_radius[r] for r in radii], "-o",
         color=tc.MAIN, lw=2.2, ms=6)
ax1.axvline(200, color=tc.RED, lw=1.2, ls="--")
ax1.set_xlabel("半径（米，固定时长 10min）")
ax1.set_ylabel("人均停留点数")
ax1.set_xticks(radii)
ax1.grid(color=tc.GRAY, lw=.6)
ax2.plot(min_mins, [by_min[m] for m in min_mins], "-o",
         color=tc.AMBER, lw=2.2, ms=6)
ax2.axvline(10, color=tc.RED, lw=1.2, ls="--")
ax2.set_xlabel("最短时长（分钟，固定半径 200m）")
ax2.set_xticks(min_mins)
ax2.grid(color=tc.GRAY, lw=.6)
tc.finish(
    fig,
    f"半径 100→500m 人均停留 {by_radius[100]:.1f}→{by_radius[500]:.1f}，"
    f"时长越严停留越少",
    "左=变半径、右变最短时长；红虚线=选用口径（200m/10min）；停留点对半径、时长的敏感度",
    SOURCE, "D6_03_参数敏感性.png")


print("\nD6 完成：图片已保存至", tc.IMAGE_DIR)