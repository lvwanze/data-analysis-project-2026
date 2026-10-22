# -*- coding: utf-8 -*-
"""
compute_dq.py —— 出租车数据质量稽核：一次性计算 16 项 DQ 的全部关键数字
作为《出租车数据质量分析报告》的事实来源（数字由脚本计算，可复核）。
输出：
  _cache/dq_facts.json   结构化事实
  _tmp/dq_facts.txt      可读摘要
"""
from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np
import pandas as pd
import taxi_common as tc

# ---------------- 清洗框 / 区域口径（与 D2-D4 一致）
LNG_MIN, LNG_MAX = -9.0, -8.0
LAT_MIN, LAT_MAX = 40.8, 41.6
CITY = (-8.610, 41.150)          # 市中心 (lng, lat)
AIRPORT = (-8.686, 41.236)       # 机场
R_CORE, R_INNER, R_OUTER = 1500, 4000, 8000   # 米

def haversine(lng1, lat1, lng2, lat2):
    R = 6371000.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = np.radians(lat2 - lat1)
    dl = np.radians(lng2 - lng1)
    a = np.sin(dp/2)**2 + np.cos(p1)*np.cos(p2)*np.sin(dl/2)**2
    return 2*R*np.arcsin(np.sqrt(a))

facts = {}

df = tc.load_raw()                      # 已解析的原始数据
raw = pd.read_csv(tc.DATA_PATH)         # 原始 9 列（用于重复/画像）
N = len(raw)
facts["规模"] = {"总行数": int(N), "总列数": int(raw.shape[1])}

# ---------- DQ-01 时间覆盖
dates = df["date"].value_counts().sort_index()
facts["DQ-01"] = {
    "时间范围": f'{df["date"].min()} 至 {df["date"].max()}',
    "天数": int(df["date"].nunique()),
    "各日趟次": {k: int(v) for k, v in dates.items()},
    "末日": str(df["date"].max()),
    "末日趟次": int(dates.iloc[-1]),
    "完整日": int(len(dates) - 1),
    "完整日中位趟次": int(dates.iloc[:-1].median()),
}

# ---------- 字段画像 / DQ-02 结构性缺失
prof = tc.profile(raw)
facts["字段画像"] = prof.to_dict("records")
facts["DQ-02"] = {
    "ORIGIN_CALL缺失率": round(float(raw["ORIGIN_CALL"].isna().mean())*100, 1),
    "ORIGIN_STAND缺失率": round(float(raw["ORIGIN_STAND"].isna().mean())*100, 1),
}

# ---------- DQ-03 空轨迹
empty = int((df["n_pts"] == 0).sum())
facts["DQ-03"] = {"空轨迹": empty, "占比": round(empty/N*100, 2)}

# ---------- DQ-04 关键列缺失
key_cols = ["TRIP_ID", "TAXI_ID", "TIMESTAMP", "POLYLINE"]
facts["DQ-04"] = {c: int(raw[c].isna().sum()) for c in key_cols}

# ---------- DQ-05 枚举取值
facts["DQ-05"] = {
    "CALL_TYPE": {str(k): int(v) for k, v in raw["CALL_TYPE"].value_counts().sort_index().items()},
    "DAY_TYPE取值": sorted(raw["DAY_TYPE"].dropna().unique().tolist()),
    "MISSING_DATA取值": sorted(raw["MISSING_DATA"].dropna().unique().tolist()),
}

# ---------- DQ-06 格式（POLYLINE 可解析 / TIMESTAMP 合法）
facts["DQ-06"] = {
    "POLYLINE可解析": int(df["n_pts"].notna().sum()),
    "TIMESTAMP正整数": int((raw["TIMESTAMP"] > 0).sum()),
    "TIMESTAMP最小": int(raw["TIMESTAMP"].min()),
    "TIMESTAMP最大": int(raw["TIMESTAMP"].max()),
}

# ---------- DQ-07 坐标范围
has_pt = df[df["n_pts"] > 0]
out_box = int(((has_pt["lng0"] < LNG_MIN) | (has_pt["lng0"] > LNG_MAX) |
               (has_pt["lat0"] < LAT_MIN) | (has_pt["lat0"] > LAT_MAX)).sum())
facts["DQ-07"] = {
    "lng0最小": round(float(has_pt["lng0"].min()), 6),
    "lng0最大": round(float(has_pt["lng0"].max()), 6),
    "lat0最小": round(float(has_pt["lat0"].min()), 6),
    "lat0最大": round(float(has_pt["lat0"].max()), 6),
    "首点越界": out_box,
}

# ---------- DQ-08 完全重复（原始 9 列）
dup_full = int(raw.duplicated().sum())
facts["DQ-08"] = {"完全重复行": dup_full}

# ---------- DQ-09 TRIP_ID 主键
dup_id = int(raw["TRIP_ID"].duplicated(keep=False).sum())
nuniq_id = int(raw["TRIP_ID"].nunique())
facts["DQ-09"] = {"TRIP_ID唯一数": nuniq_id, "重复ID涉及行数": dup_id,
                  "重复ID对数": int((N - nuniq_id))}

# ---------- DQ-10 站台与叫车类型关联
b = raw[raw["CALL_TYPE"] == "B"]
c = raw[raw["CALL_TYPE"] == "C"]
a = raw[raw["CALL_TYPE"] == "A"]
b_stand = int(b["ORIGIN_STAND"].notna().sum())
facts["DQ-10"] = {
    "B型趟次": int(len(b)), "B型有站台": b_stand,
    "B型关联率": round(b_stand/len(b)*100, 2),
    "C型趟次": int(len(c)), "C型站台缺失": int(c["ORIGIN_STAND"].isna().sum()),
    "A型趟次": int(len(a)),
    "站台数": int(raw["ORIGIN_STAND"].nunique()),
}

# ---------- DQ-11 GPS 采样间隔 / 跳点
seg_d = []
for p in df["poly"]:
    if len(p) < 2:
        continue
    arr = np.asarray(p, dtype=float)
    seg_d.append(haversine(arr[:-1, 0], arr[:-1, 1], arr[1:, 0], arr[1:, 1]))
seg = np.concatenate(seg_d)
facts["DQ-11"] = {
    "相邻点距离中位": round(float(np.median(seg)), 1),
    "相邻点距离P99": round(float(np.quantile(seg, 0.99)), 1),
    "跳点占比": round(float((seg > 420).mean())*100, 2),
    "标称采样间隔秒": tc.GPS_INTERVAL_SEC,
}

# ---------- DQ-12 站台定位准确性
def _stand_disp(frame):
    out = {}
    for sid, g in frame[frame["ORIGIN_STAND"].notna()].groupby("ORIGIN_STAND"):
        lng_m, lat_m = g["lng0"].median(), g["lat0"].median()
        dd = haversine(g["lng0"].values, g["lat0"].values, lng_m, lat_m)
        out[int(sid)] = {"中位": round(float(np.median(dd)), 1),
                         "最大": round(float(dd.max()), 1), "趟次": int(len(g))}
    return out
disp_raw = _stand_disp(has_pt)
inbox_stand = has_pt[(has_pt["lng0"].between(LNG_MIN, LNG_MAX)) &
                     (has_pt["lat0"].between(LAT_MIN, LAT_MAX))]
disp = _stand_disp(inbox_stand)
dmed = [v["中位"] for v in disp.values()]
dmax = [v["最大"] for v in disp.values()]
facts["DQ-12"] = {
    "站台数": len(disp),
    "离散度中位": round(float(np.median(dmed)), 1),
    "离散度P90": round(float(np.quantile(dmed, 0.9)), 1),
    "清洗后最大离散": round(float(np.max(dmax)), 1),
    "原始最大离散": round(float(np.max([v["最大"] for v in disp_raw.values()])), 1),
}

# ---------- DQ-13 / DQ-14 网格与区域
KM_LAT = 1/111.0
KM_LNG = 1/(111.32*math.cos(math.radians(41.15)))
inbox = has_pt[(has_pt["lng0"].between(LNG_MIN, LNG_MAX)) &
               (has_pt["lat0"].between(LAT_MIN, LAT_MAX))]
gx = ((inbox["lng0"] - LNG_MIN)/KM_LNG).astype(int)
gy = ((inbox["lat0"] - LAT_MIN)/KM_LAT).astype(int)
ncell = int(pd.concat([gx, gy], axis=1).drop_duplicates().shape[0])
top = pd.DataFrame({"gx": gx, "gy": gy}).value_counts().iloc[0]
# 区域
dcity = haversine(inbox["lng0"].values, inbox["lat0"].values, CITY[0], CITY[1])
dair = haversine(inbox["lng0"].values, inbox["lat0"].values, AIRPORT[0], AIRPORT[1])
region = np.where(dair < 1000, "机场",
          np.where(dcity < R_CORE, "核心",
           np.where(dcity < R_INNER, "内环",
            np.where(dcity < R_OUTER, "外环", "远郊"))))
rc = pd.Series(region).value_counts()
facts["DQ-13"] = {"网格数": ncell,
                  "区域占比": {k: round(float(rc.get(k, 0))/len(inbox)*100, 1)
                              for k in ["核心", "内环", "外环", "机场", "远郊"]}}
facts["DQ-14"] = {"网格数": ncell,
                  "最密网格趟次": int(top),
                  "最密网格占比": round(int(top)/len(inbox)*100, 1),
                  "都市区框内趟次": int(len(inbox))}

# ---------- DQ-15 样本口径
per_taxi = N / df["TAXI_ID"].nunique()
facts["DQ-15"] = {"车辆数": int(df["TAXI_ID"].nunique()),
                  "车均趟次": round(per_taxi, 1),
                  "单车最多趟次": int(raw.groupby("TAXI_ID").size().max()),
                  "单车最少趟次": int(raw.groupby("TAXI_ID").size().min())}

# ---------- DQ-16 机场长尾
air_n = int((dair < 1000).sum())
facts["DQ-16"] = {"机场1000m内趟次": air_n,
                  "占比": round(air_n/len(inbox)*100, 3)}

# ---------- 同车相邻趟次间隔（D2 结果，报告引用）
d = df.sort_values(["TAXI_ID", "ts_utc"])
d["gap_min"] = d.groupby("TAXI_ID")["ts_utc"].diff().dt.total_seconds()/60
g = d["gap_min"].dropna()
facts["相邻趟次间隔"] = {"中位": round(float(g.median()), 1),
                       "P90": round(float(g.quantile(0.9)), 1),
                       "P99": round(float(g.quantile(0.99)), 1)}

# ---------------- 保存
cache_dir = tc.CACHE_DIR; cache_dir.mkdir(exist_ok=True)
with open(cache_dir / "dq_facts.json", "w", encoding="utf-8") as f:
    json.dump(facts, f, ensure_ascii=False, indent=2, default=str)

tmp = tc.CACHE_DIR / "_tmp"; tmp.mkdir(exist_ok=True)
with open(tmp / "dq_facts.txt", "w", encoding="utf-8") as f:
    f.write(json.dumps(facts, ensure_ascii=False, indent=2, default=str))

print("DQL 事实已计算：")
print("  规模", facts["规模"])
print("  DQ-01 末日", facts["DQ-01"]["末日"], facts["DQ-01"]["末日趟次"], "趟")
print("  DQ-03 空轨迹", facts["DQ-03"])
print("  DQ-07 首点越界", facts["DQ-07"]["首点越界"])
print("  DQ-08 完全重复", facts["DQ-08"])
print("  DQ-09", facts["DQ-09"])
print("  DQ-10 B型关联率", facts["DQ-10"]["B型关联率"])
print("  DQ-11 跳点", facts["DQ-11"]["跳点占比"], "%")
print("  DQ-12 站台离散中位", facts["DQ-12"]["离散度中位"])
print("  DQ-13", facts["DQ-13"])
print("  DQ-14 最密网格", facts["DQ-14"]["最密网格趟次"], facts["DQ-14"]["最密网格占比"], "%")
print("  DQ-15", facts["DQ-15"])
print("  DQ-16 机场", facts["DQ-16"])
print("完成")
