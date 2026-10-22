# -*- coding: utf-8 -*-

from __future__ import annotations
from pathlib import Path
import sys
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle

# ---------------------------------------------------------------- 路径
# 整个项目自包含：所有路径都基于本文件所在目录（taxi_analysis），
# 整个文件夹拷贝到任意位置 / 任意盘均可运行，不依赖外部项目目录。
THIS_DIR = Path(__file__).resolve().parent
DATA_DIR = THIS_DIR / "data"
DATA_PATH = DATA_DIR / "train_100k.csv"
# 兼容旧布局：数据放在项目上一级的 data/（如 signal-insight 根目录）
if not DATA_PATH.exists():
    _legacy = THIS_DIR.parent / "data" / "train_100k.csv"
    if _legacy.exists():
        DATA_PATH = _legacy
# 图片目录定位到项目根（脚本可能在根目录或 notebook/ 子目录，以 data/ 所在目录为项目根）
_IMG_ROOT = THIS_DIR if (THIS_DIR / "data").exists() else THIS_DIR.parent
IMAGE_DIR = _IMG_ROOT / "image"
IMAGE_DIR.mkdir(parents=True, exist_ok=True)
# 缓存目录同样定位到项目根（避免脚本在子目录运行时缓存漂移、产生双份）
CACHE_DIR = _IMG_ROOT / "_cache"

# 图内署名（各脚本出图前可覆写为分工署名：成员1 负责 D1~D3、成员2 负责 D4~D6）
AUTHOR = "高帅龙 2428324009、薛宏博 2428324276"   # 公共/协作默认
SOURCE_NOTE = "数据来源：Porto 出租车轨迹样本 train_100k（ECML/PKDD 2015，WGS84）"
TIMEZONE = "Europe/Lisbon"          # PORTO 本地时区（含夏令时）
GPS_INTERVAL_SEC = 15               # POLYLINE 相邻点标称间隔

# ---------------------------------------------------------------- 颜色（图表规范第 2 节）
MAIN = "#1D6FA5"
MAIN_LIGHT = "#DCEAF5"
MAIN_MID = "#7FB3D9"
MAIN_DARK = "#15527B"
GRAY = "#C4CAD3"
GRAY_DARK = "#8A929E"
RED = "#D64545"
AMBER = "#B7791F"
GREEN = "#3B8C66"
INK = "#1F2937"
MUTED = "#6B7280"
FAINT = "#9CA3AF"

# 开源中文字体回退链（规范第 0、3 节，只列开源字体）
_CN_FONTS = [
    "Source Han Sans SC", "Noto Sans CJK SC", "Noto Sans SC",
    "Noto Sans CJK JP", "WenQuanYi Zen Hei",
]


def apply_theme() -> list[str]:
    """套用项目视觉规范，返回本机实际可用的字体列表。"""
    installed = {f.name for f in font_manager.fontManager.ttflist}
    cn_available = [f for f in _CN_FONTS if f in installed]
    # 把本机已装字体 + DejaVu 兜底写进 font.family，支持逐字回退
    family = (cn_available + ["DejaVu Sans"])
    matplotlib.rcParams.update({
        "font.family": family,
        "axes.unicode_minus": True,
        "axes.edgecolor": "#C4CAD3",
        "axes.linewidth": 0.8,
        "axes.labelcolor": INK,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.grid": True,
        "grid.color": "#E9EDF2",
        "grid.linewidth": 0.7,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.bbox": "tight",
        "savefig.dpi": 200,
    })
    return cn_available


def finish(fig: plt.Figure, title: str, subtitle: str = "",
           source: str = "", filename: str = ""):
    """
    统一收尾：强调色短条 + 结论标题 + 口径副标题 + 来源行（规范第 1 节）。
    标题里的数字必须由调用方代码算好后传入。
    """
    fig.canvas.draw()
    fig_w_in = fig.get_figwidth()
    # 顶部预留标题区
    fig.text(0.06, 0.955, title, fontsize=15, fontweight="bold", color=INK)
    # 强调色短条（标题上方，宽约 0.38 英寸）
    bar_w = 0.38 / fig_w_in
    fig.add_artist(Rectangle((0.06, 0.972), bar_w, 0.012,
                             facecolor=MAIN, edgecolor="none",
                             transform=fig.transFigure, clip_on=False))
    if subtitle:
        fig.text(0.06, 0.925, subtitle, fontsize=9.5, color=MUTED, wrap=True)
    if source:
        fig.text(0.06, 0.02, source, fontsize=8, color=FAINT)
    if filename:
        out = IMAGE_DIR / filename
        fig.savefig(out)
    return fig


def source_line(script: str) -> str:
    """标准来源行。"""
    return f"{SOURCE_NOTE}；脚本：{script}；制图：{AUTHOR}"


def load_raw(use_cache: bool = True):
    """
    加载原始 CSV，并完成与业务无关的基础解析：
      - POLYLINE -> list（列名 poly）
      - 点数 n_pts
      - TIMESTAMP -> UTC / 里斯本本地时间（ts_local, date, hour, weekday）
      - 首点经纬度（lng0, lat0）
    首次读取后落 parquet 缓存（D5 会对比格式性能），之后直接读缓存。
    """
    import json
    import pandas as pd

    cache = CACHE_DIR / "raw_base.pkl"
    if use_cache and cache.exists():
        return pd.read_pickle(cache)

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"未找到原始数据 {DATA_PATH.name}。请将 train_100k.csv 放到项目内 "
            f"{DATA_DIR} 目录（或旧的 {THIS_DIR.parent / 'data'} 布局），再重新运行。")

    df = pd.read_csv(DATA_PATH)
    df["poly"] = df["POLYLINE"].map(json.loads)
    df["n_pts"] = df["poly"].map(len)
    ts_utc = pd.to_datetime(df["TIMESTAMP"], unit="s", utc=True)
    df["ts_utc"] = ts_utc
    ts_local = ts_utc.dt.tz_convert(TIMEZONE)
    df["ts_local"] = ts_local
    df["date"] = ts_local.dt.date.astype(str)
    df["hour"] = ts_local.dt.hour
    df["weekday"] = ts_local.dt.weekday           # 0=周一
    df["lng0"] = df["poly"].map(lambda x: x[0][0] if x else float("nan"))
    df["lat0"] = df["poly"].map(lambda x: x[0][1] if x else float("nan"))
    df["lng1"] = df["poly"].map(lambda x: x[-1][0] if x else float("nan"))
    df["lat1"] = df["poly"].map(lambda x: x[-1][1] if x else float("nan"))

    if use_cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        df.to_pickle(cache)
    return df


def profile(df) -> "pd.DataFrame":
    """字段画像：类型、缺失数/率、唯一值数、示例（类比 D1 profile）。"""
    import pandas as pd
    rows = []
    for c in df.columns:
        s = df[c]
        ex = s.dropna().iloc[0] if s.notna().any() else None
        rows.append({
            "字段": c,
            "类型": str(s.dtype),
            "缺失数": int(s.isna().sum()),
            "缺失率": round(float(s.isna().mean()), 4),
            "唯一值数": int(s.nunique(dropna=True)),
            "示例": str(ex)[:40],
        })
    return pd.DataFrame(rows)


def env_self_check() -> None:
    """D1 环境自检：版本、字体、数据文件。"""
    import pandas as pd
    import numpy as np
    cn = apply_theme()
    print(f"Python      {sys.version.split()[0]}")
    print(f"pandas      {pd.__version__}")
    print(f"numpy       {np.__version__}")
    print(f"matplotlib  {matplotlib.__version__}")
    print(f"开源中文字体 {cn if cn else '无（将回退 DejaVu，中文可能变方框）'}")
    print(f"数据文件    {DATA_PATH.name}（{DATA_PATH.stat().st_size / 1e6:.0f} MB）")
