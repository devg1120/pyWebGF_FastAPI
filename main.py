import io
import glob
import os
import platform
import json
import traceback
from typing import Optional
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import matplotlib.dates as mdates
from natsort import natsorted

# FastAPI関連
from fastapi import FastAPI, Request, Form, status
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.templating import Jinja2Templates

# Merlion 関連のインポート
#from merlion.utils import TimeSeries
#from merlion.models.anomaly.default import DefaultDetector, DefaultDetectorConfig

# ✖ 既存のものを削除
# from merlion.models.anomaly.default import DefaultDetector, DefaultDetectorConfig

# ⭕ 以下に書き換え
from merlion.utils import TimeSeries
from merlion.models.anomaly.isolation_forest import IsolationForest, IsolationForestConfig

# 動的しきい値（直近の急変重視）モデルのインポートを追加
from merlion.models.anomaly.stat_threshold import StatThreshold, StatThresholdConfig

# バースト検出（ウィンドウ統計型）モデルのインポートを追加
#from merlion.models.anomaly.windower import Windower,WindowerConfig
# ✖ エラーになった既存の行を削除
# from merlion.models.anomaly.windower import Windower, WindowerConfig

# ⭕ 以下に書き換え
#from merlion.models.anomaly.spectral_anomaly import SpectralAnomalies, SpectralAnomaliesConfig

from merlion.post_process.threshold import AggregateAlarms


# OS自動判定による日本語文字化け対策
matplotlib.use("Agg")
current_os = platform.system()
if current_os == "Windows":
    matplotlib.rcParams["font.family"] = "Yu Gothic"
elif current_os == "Linux":
    matplotlib.rcParams["font.family"] = "IPAexGothic"
else:
    matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["axes.unicode_minus"] = False

app = FastAPI()
templates = Jinja2Templates(directory="templates")
BASE_DIR = "./gen_csv/month_data_build/test"
DASHBOARD_REGISTRY = []

# 共通ディレクトリツリー取得
def get_date_tree():
    date_tree = {}
    yyyymm_dirs = sorted([d for d in glob.glob(os.path.join(BASE_DIR, "*")) if os.path.isdir(d)])
    for ymd in yyyymm_dirs:
        ym_name = os.path.basename(ymd)
        dd_dirs = sorted([d for d in glob.glob(os.path.join(ymd, "*")) if os.path.isdir(d)])
        dd_list = [os.path.basename(d) for d in dd_dirs]
        if dd_list:
            date_tree[ym_name] = dd_list
    return date_tree
@app.get("/", response_class=HTMLResponse)
async def index(
    request: Request,
    mode: str = "single",
    yyyymm: str = "",
    dd: str = "",
    start_dd: str = "",
    end_dd: str = "",
    switch: str = "",
    cols: int = 2,
    start_hour: int = 8,
    end_hour: int = 18,
):
    date_tree = get_date_tree()
    if not date_tree:
        return HTMLResponse(content=f"<h1>エラー: ディレクトリが見つかりません。</h1>", status_code=500)

    available_yyyymm = list(date_tree.keys())
    if not yyyymm or yyyymm not in date_tree:
        yyyymm = available_yyyymm[0] if available_yyyymm else ""

    available_dds = date_tree.get(yyyymm, [])
    if not available_dds:
        return HTMLResponse(content=f"<h1>エラー: 有効な日がありません。</h1>", status_code=500)

    if not dd or dd not in available_dds:
        dd = available_dds[0] if available_dds else ""

    selected_date = f"{yyyymm}/{dd}"
    date_dir = os.path.join(BASE_DIR, selected_date)
    available_switches = sorted([os.path.basename(d) for d in glob.glob(os.path.join(date_dir, "*")) if os.path.isdir(d)])

    if not switch or switch not in available_switches:
        switch = available_switches[0] if available_switches else ""

    csv_filenames = []
    info_dic = {}
    if switch:
        target_dir = os.path.join(BASE_DIR, selected_date, switch)
        csv_files = sorted(glob.glob(os.path.join(target_dir, "*.csv")))
        csv_filenames = natsorted([os.path.basename(f) for f in csv_files])
        info_json_path = os.path.join(target_dir, "info.json")
        if os.path.exists(info_json_path):
            with open(info_json_path, 'r', encoding='utf-8') as f:
                info_dic = json.load(f)

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "date_tree": date_tree, "current_mode": mode,
            "selected_start_dd": start_dd, "selected_end_dd": end_dd,
            "selected_yyyymm": yyyymm, "selected_dd": dd, "available_dds": available_dds,
            "switches": available_switches, "selected_switch": switch, "selected_cols": cols,
            "start_hour": start_hour, "end_hour": end_hour, "csv_filenames": csv_filenames,
            "info_dic": info_dic, "registry": DASHBOARD_REGISTRY,
        }
    )

# ※もし上部のインポートになければ、関数の外か中にこれを追加してください
from fastapi.concurrency import run_in_threadpool

@app.get("/plot")
async def plot_graph(
    mode: str = "single", yyyymm: Optional[str] = None, dd: Optional[str] = None,
    switch: Optional[str] = None, filename: Optional[str] = None, cols: int = 2,
    start_hour: int = 8, end_hour: int = 18, desc: str = "", start_dd: str = "", end_dd: str = "",
):
    fig_size = (14, 4.0) if cols == 1 else (6, 2.8) if cols == 4 else (8, 3.0)
    fig = Figure(figsize=fig_size)
    ax = fig.subplots()

    try:
        df_list = []
        if mode == "range" and start_dd and end_dd:
            dr = pd.date_range(start=start_dd, end=end_dd)
            for d in dr:
                file_path = os.path.join(BASE_DIR, d.strftime("%Y%m"), d.strftime("%d"), switch, filename)
                if os.path.exists(file_path):
                    # 💡 【非同期化】スレッドプールでPandasのCSV読み込みを実行
                    _df = await run_in_threadpool(pd.read_csv, file_path, engine="c")
                    if not _df.empty:
                        df_list.append(_df)
            if df_list:
                df = pd.concat(df_list, ignore_index=True)
            else:
                raise FileNotFoundError("データファイルが見つかりません。")
            start_time = pd.to_datetime(start_dd) + pd.Timedelta(hours=start_hour)
            end_time = pd.to_datetime(end_dd) + pd.Timedelta(hours=end_hour, minutes=55)
            x_label_text = f"{start_dd} ～ {end_dd}"
            x_tick_format = "%m/%d %H:%M"
        else:
            file_path = os.path.join(BASE_DIR, yyyymm, dd, switch, filename)
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"ファイルなし: {yyyymm}/{dd}")

            # 💡 【非同期化】スレッドプールでPandasのCSV読み込みを実行
            df = await run_in_threadpool(pd.read_csv, file_path, engine="c")

            base_date = pd.to_datetime(f"{yyyymm[:4]}-{yyyymm[4:]}-{dd}").normalize()
            start_time = base_date + pd.Timedelta(hours=start_hour)
            end_time = base_date + pd.Timedelta(hours=end_hour, minutes=55)
            x_label_text = f"{yyyymm[:4]}-{yyyymm[4:6]}-{dd}"
            x_tick_format = "%H:%M"

        if len(df.columns) >= 6:
            df.columns = ["日時", "INDEX", "NAME", "STATUS", "項目A", "項目B"] + list(df.columns[6:])
        elif len(df.columns) >= 3:
            df.columns = ["日時", "項目A", "項目B"] + list(df.columns[3:])

        df["日時"] = pd.to_datetime(df["日時"], format="mixed", errors='coerce')
        df = df.dropna(subset=["日時"]).sort_values("日時").reset_index(drop=True)
        df["差分A"] = df["項目A"].diff().clip(lower=0)
        df["差分B"] = df["項目B"].diff().clip(lower=0) if "項目B" in df.columns else 0
        df["bps_A"] = (df["差分A"] * 8) / 300
        df["bps_B"] = (df["差分B"] * 8) / 300

        df_filtered = df[(df["日時"] >= start_time) & (df["日時"] <= end_time)]
        fig.set_facecolor("#f0f0f0")
        ax.set_facecolor("#ffffff")

        if not df_filtered.empty:
            ax.fill_between(df_filtered["日時"], df_filtered["bps_A"], color="#00eb0c", alpha=0.9, label="In (bps)")
            ax.plot(df_filtered["日時"], df_filtered["bps_A"], color="#006600", linewidth=0.8)
            if "bps_B" in df_filtered.columns:
                ax.plot(df_filtered["日時"], df_filtered["bps_B"], color="#1000ff", linewidth=1.8, label="Out (bps)")
        else:
            ax.text(0.5, 0.5, "データがありません", ha="center", va="center", color="red")

        ax.grid(True, which="both", color="#cccccc", linestyle="-", linewidth=0.5)
        ax.set_xlim(start_time, end_time)
        ax.xaxis.set_major_formatter(mdates.DateFormatter(x_tick_format))
        for label in ax.get_xticklabels():
            label.set_rotation(30)
            label.set_horizontalalignment("right")

        max_val = max(df_filtered["bps_A"].max(), df_filtered["bps_B"].max()) if not df_filtered.empty else 0
        if max_val >= 1_000_000:
            ax.get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: f"{x / 1_000_000:,.1f} M"))
            ax.set_ylabel("転送速度 (Mbps)")
        elif max_val >= 1_000:
            ax.get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: f"{x / 1_000:,.0f} K"))
            ax.set_ylabel("転送速度 (Kbps)")
        else:
            ax.get_yaxis().set_major_formatter(matplotlib.ticker.StrMethodFormatter("{x:,.0f}"))
            ax.set_ylabel("転送速度 (bps)")

        ax.set_title(f"{switch}: {filename.replace('.csv', '')}", fontsize=11, loc="left")
        ax.set_title(desc, fontsize=11, loc="right")
        ax.set_xlabel(x_label_text)
    except Exception as e:
        traceback.print_exc()
        ax.text(0.5, 0.5, f"Error: {type(e).__name__}", ha="center", va="center", color="red")

    fig.legend(bbox_to_anchor=(0.13, -0.04), loc='lower left', ncol=2, frameon=True, facecolor="#ffffff", edgecolor="#cccccc")
    plt.subplots_adjust(bottom=0.25)
    fig.tight_layout()

    img = io.BytesIO()
    # 💡 【参考】画像保存（fig.savefig）もスレッドプールに逃がすことでさらに安定します
    await run_in_threadpool(fig.savefig, img, format="png", bbox_inches="tight", dpi=120)
    img.seek(0)
    fig.clear()
    return StreamingResponse(img, media_type="image/png")

"""
@app.get("/plot")
async def plot_graph(
    mode: str = "single", yyyymm: Optional[str] = None, dd: Optional[str] = None,
    switch: Optional[str] = None, filename: Optional[str] = None, cols: int = 2,
    start_hour: int = 8, end_hour: int = 18, desc: str = "", start_dd: str = "", end_dd: str = "",
):
    fig_size = (14, 4.0) if cols == 1 else (6, 2.8) if cols == 4 else (8, 3.0)
    fig = Figure(figsize=fig_size)
    ax = fig.subplots()

    try:
        df_list = []
        if mode == "range" and start_dd and end_dd:
            dr = pd.date_range(start=start_dd, end=end_dd)
            for d in dr:
                file_path = os.path.join(BASE_DIR, d.strftime("%Y%m"), d.strftime("%d"), switch, filename)
                if os.path.exists(file_path):
                    _df = pd.read_csv(file_path, engine="c")
                    if not _df.empty:
                        df_list.append(_df)
            if df_list:
                df = pd.concat(df_list, ignore_index=True)
            else:
                raise FileNotFoundError("データファイルが見つかりません。")
            start_time = pd.to_datetime(start_dd) + pd.Timedelta(hours=start_hour)
            end_time = pd.to_datetime(end_dd) + pd.Timedelta(hours=end_hour, minutes=55)
            x_label_text = f"{start_dd} ～ {end_dd}"
            x_tick_format = "%m/%d %H:%M"
        else:
            file_path = os.path.join(BASE_DIR, yyyymm, dd, switch, filename)
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"ファイルなし: {yyyymm}/{dd}")
            df = pd.read_csv(file_path, engine="c")
            base_date = pd.to_datetime(f"{yyyymm[:4]}-{yyyymm[4:]}-{dd}").normalize()
            start_time = base_date + pd.Timedelta(hours=start_hour)
            end_time = base_date + pd.Timedelta(hours=end_hour, minutes=55)
            x_label_text = f"{yyyymm[:4]}-{yyyymm[4:6]}-{dd}"
            x_tick_format = "%H:%M"

        if len(df.columns) >= 6:
            df.columns = ["日時", "INDEX", "NAME", "STATUS", "項目A", "項目B"] + list(df.columns[6:])
        elif len(df.columns) >= 3:
            df.columns = ["日時", "項目A", "項目B"] + list(df.columns[3:])

        df["日時"] = pd.to_datetime(df["日時"], format="mixed", errors='coerce')
        df = df.dropna(subset=["日時"]).sort_values("日時").reset_index(drop=True)
        df["差分A"] = df["項目A"].diff().clip(lower=0)
        df["差分B"] = df["項目B"].diff().clip(lower=0) if "項目B" in df.columns else 0
        df["bps_A"] = (df["差分A"] * 8) / 300
        df["bps_B"] = (df["差分B"] * 8) / 300

        df_filtered = df[(df["日時"] >= start_time) & (df["日時"] <= end_time)]
        fig.set_facecolor("#f0f0f0")
        ax.set_facecolor("#ffffff")

        if not df_filtered.empty:
            ax.fill_between(df_filtered["日時"], df_filtered["bps_A"], color="#00eb0c", alpha=0.9, label="In (bps)")
            ax.plot(df_filtered["日時"], df_filtered["bps_A"], color="#006600", linewidth=0.8)
            if "bps_B" in df_filtered.columns:
                ax.plot(df_filtered["日時"], df_filtered["bps_B"], color="#1000ff", linewidth=1.8, label="Out (bps)")
        else:
            ax.text(0.5, 0.5, "データがありません", ha="center", va="center", color="red")

        ax.grid(True, which="both", color="#cccccc", linestyle="-", linewidth=0.5)
        ax.set_xlim(start_time, end_time)
        ax.xaxis.set_major_formatter(mdates.DateFormatter(x_tick_format))
        for label in ax.get_xticklabels():
            label.set_rotation(30)
            label.set_horizontalalignment("right")

        max_val = max(df_filtered["bps_A"].max(), df_filtered["bps_B"].max()) if not df_filtered.empty else 0
        if max_val >= 1_000_000:
            ax.get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: f"{x / 1_000_000:,.1f} M"))
            ax.set_ylabel("転送速度 (Mbps)")
        elif max_val >= 1_000:
            ax.get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: f"{x / 1_000:,.0f} K"))
            ax.set_ylabel("転送速度 (Kbps)")
        else:
            ax.get_yaxis().set_major_formatter(matplotlib.ticker.StrMethodFormatter("{x:,.0f}"))
            ax.set_ylabel("転送速度 (bps)")

        ax.set_title(f"{switch}: {filename.replace('.csv', '')}", fontsize=11, loc="left")
        ax.set_title(desc, fontsize=11, loc="right")
        ax.set_xlabel(x_label_text)
    except Exception as e:
        traceback.print_exc()
        ax.text(0.5, 0.5, f"Error: {type(e).__name__}", ha="center", va="center", color="red")

    fig.legend(bbox_to_anchor=(0.13, -0.04), loc='lower left', ncol=2, frameon=True, facecolor="#ffffff", edgecolor="#cccccc")
    plt.subplots_adjust(bottom=0.25)
    fig.tight_layout()
    
    img = io.BytesIO()
    fig.savefig(img, format="png", bbox_inches="tight", dpi=120)
    img.seek(0)
    fig.clear()
    return StreamingResponse(img, media_type="image/png")
"""

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(
    request: Request, mode: str = "single", yyyymm: str = "", dd: str = "",
    start_dd: str = "", end_dd: str = "", cols: int = 2, start_hour: int = 8, end_hour: int = 18,
):
    date_tree = get_date_tree()
    available_yyyymm = list(date_tree.keys())
    if not yyyymm or yyyymm not in date_tree:
        yyyymm = available_yyyymm[0] if available_yyyymm else ""
    available_dds = date_tree.get(yyyymm, [])
    if not dd or dd not in available_dds:
        dd = available_dds[0] if available_dds else ""

    return templates.TemplateResponse(
        request=request,
        name="dashboard.html",
        context={
            "registry": DASHBOARD_REGISTRY, "date_tree": date_tree,
            "current_mode": mode, "selected_start_dd": start_dd, "selected_end_dd": end_dd,
            "selected_yyyymm": yyyymm, "selected_dd": dd, "available_dds": available_dds,
            "selected_cols": cols, "start_hour": start_hour, "end_hour": end_hour,
        }
    )


@app.post("/dashboard/add")
async def dashboard_add(
    yyyymm: str = Form(...), dd: str = Form(...), switch: str = Form(...), filename: str = Form(...), desc: str = Form(""),
):
    if yyyymm and dd and switch and filename:
        exists = any(i["yyyymm"] == yyyymm and i["dd"] == dd and i["switch"] == switch and i["filename"] == filename and i["desc"] == desc for i in DASHBOARD_REGISTRY)
        if not exists:
            DASHBOARD_REGISTRY.append({"yyyymm": yyyymm, "dd": dd, "switch": switch, "filename": filename, "desc": desc})
    return RedirectResponse(url=f"/?mode=single&yyyymm={yyyymm}&dd={dd}&switch={switch}", status_code=status.HTTP_303_SEE_OTHER)

@app.post("/dashboard/delete")
async def dashboard_delete(index: int = Form(...)):
    if 0 <= index < len(DASHBOARD_REGISTRY):
        DASHBOARD_REGISTRY.pop(index)
    return RedirectResponse(url="/dashboard", status_code=status.HTTP_303_SEE_OTHER)



@app.get("/anomaly_dashboard", response_class=HTMLResponse)
async def anomaly_dashboard(
    request: Request,
    start_dd: str = "",
    end_dd: str = "",
    switch: str = "",
    logic: str = "stat_threshold",  # 💡 初期値を「直近の急変重視」に設定
    start_hour: int = 8,
    end_hour: int = 18,
):
    """異常検知結果を列挙する画面のメインルート"""
    date_tree = get_date_tree()
    csv_filenames = []

    if switch and start_dd:
        try:
            parsed_date = pd.to_datetime(start_dd)
            ym_dir = parsed_date.strftime("%Y%m")
            d_single = str(parsed_date.day)
            d_double = parsed_date.strftime("%d")
            
            candidate_dirs = [
                os.path.join(BASE_DIR, ym_dir, d_single, switch),
                os.path.join(BASE_DIR, ym_dir, d_double, switch),
                os.path.join(BASE_DIR, ym_dir, d_single, switch.upper()),
                os.path.join(BASE_DIR, ym_dir, d_double, switch.upper()),
            ]
            
            target_dir = None
            for c_dir in candidate_dirs:
                if os.path.exists(c_dir):
                    target_dir = c_dir
                    break
                    
            if target_dir:
                switch = os.path.basename(target_dir)
                csv_files = sorted(glob.glob(os.path.join(target_dir, "*.csv")))
                csv_filenames = natsorted([os.path.basename(f) for f in csv_files])
        except Exception as e:
            print(f"[エラー] パス解決処理中に障害が発生しました: {e}")

    return templates.TemplateResponse(
        request=request,
        name="anomaly_dashboard.html",
        context={
            "date_tree": date_tree,
            "selected_start_dd": start_dd,
            "selected_end_dd": end_dd,
            "selected_switch": switch,
            "selected_logic": logic,  # 💡 選択されたロジックをHTMLへ引き渡す
            "start_hour": start_hour,
            "end_hour": end_hour,
            "csv_filenames": csv_filenames,
        }
    )

"""
@app.get("/anomaly_dashboard", response_class=HTMLResponse)
async def anomaly_dashboard(
    request: Request,
    start_dd: str = "",
    end_dd: str = "",
    switch: str = "",
    start_hour: int = 8,
    end_hour: int = 18,
):
    date_tree = get_date_tree()
    csv_filenames = []

    if not start_dd and date_tree:
        first_ym = list(date_tree.keys())[0]       # 例: "202606"
        first_dd = date_tree[first_ym][0]           # 例: "01" または "1"
        # "202606" と "01" から "2026-06-01" を組み立てる
        formatted_d = first_dd.zfill(2)
        start_dd = f"{first_ym[:4]}-{first_ym[4:]}-{formatted_d}"
        if not end_dd:
            end_dd = start_dd
    # 💡 入力された条件から実在するフォルダパスを総当たりで確実に捕捉する
    if switch and start_dd:
        try:
            parsed_date = pd.to_datetime(start_dd)
            ym_dir = parsed_date.strftime("%Y%m")
            d_single = str(parsed_date.day)       # "1"
            d_double = parsed_date.strftime("%d")  # "01"

            # 探索候補のディレクトリパターン（大文字・小文字・ゼロ埋め有無）
            candidate_dirs = [
                os.path.join(BASE_DIR, ym_dir, d_single, switch),
                os.path.join(BASE_DIR, ym_dir, d_double, switch),
                os.path.join(BASE_DIR, ym_dir, d_single, switch.upper()),
                os.path.join(BASE_DIR, ym_dir, d_double, switch.upper()),
                os.path.join(BASE_DIR, ym_dir, d_single, switch.lower()),
                os.path.join(BASE_DIR, ym_dir, d_double, switch.lower()),
            ]

            # 最初に見つかった有効なディレクトリからCSVを取得
            target_dir = None
            for c_dir in candidate_dirs:
                if os.path.exists(c_dir):
                    target_dir = c_dir
                    break

            if target_dir:
                # 💡 ヒットした実際のフォルダから実在するスイッチ名を再取得して同期する
                switch = os.path.basename(target_dir)
                csv_files = sorted(glob.glob(os.path.join(target_dir, "*.csv")))
                csv_filenames = natsorted([os.path.basename(f) for f in csv_files])
            else:
                print(f"[警告] 以下の候補のいずれにもCSVディレクトリが見つかりません:")
                for c_dir in candidate_dirs:
                    print(f"  -> {c_dir}")

        except Exception as e:
            print(f"[エラー] パス解決処理中に障害が発生しました: {e}")

    return templates.TemplateResponse(
        request=request,
        name="anomaly_dashboard.html",
        context={
            "date_tree": date_tree,
            "selected_start_dd": start_dd,
            "selected_end_dd": end_dd,
            "selected_switch": switch,  # フォーム側へ正確に引き継ぐ
            "start_hour": start_hour,
            "end_hour": end_hour,
            "csv_filenames": csv_filenames,
        }
    )
"""
@app.get("/plot_anomaly")
async def plot_anomaly(
    mode: str = "range", yyyymm: Optional[str] = None, dd: Optional[str] = None,
    switch: Optional[str] = None, filename: Optional[str] = None, cols: int = 2,
    start_hour: int = 8, end_hour: int = 18, start_dd: str = "", end_dd: str = "",
    logic: str = "stat_threshold",
):
    """Merlionで異常検知を行い、異常箇所を赤くハイライトしたグラフを返すルート（変数定義安全版）"""
    fig_size = (14, 4.0) if cols == 1 else (6, 2.8) if cols == 4 else (8, 3.0)
    fig = Figure(figsize=fig_size)
    ax = fig.subplots()

    # 初期化（エラー時対策）
    df_filtered = pd.DataFrame()

    try:
        df_list = []
        if start_dd and end_dd:
            dr = pd.date_range(start=start_dd, end=end_dd)
            for d in dr:
                file_path = os.path.join(BASE_DIR, d.strftime("%Y%m"), d.strftime("%d"), switch, filename)
                if os.path.exists(file_path):
                    _df = await run_in_threadpool(pd.read_csv, file_path, engine="c")
                    if not _df.empty:
                        df_list.append(_df)

            if df_list:
                df = pd.concat(df_list, ignore_index=True)
            else:
                raise FileNotFoundError("データファイルが見つかりません。")

            start_time = pd.to_datetime(start_dd) + pd.Timedelta(hours=start_hour)
            end_time = pd.to_datetime(end_dd) + pd.Timedelta(hours=end_hour, minutes=55)
            x_label_text = f"{start_dd} ～ {end_dd}"
            x_tick_format = "%m/%d %H:%M"
        else:
            raise ValueError("期間指定(start_dd, end_dd)が必要です。")

        # ─── 共通データ整形処理 ───
        if len(df.columns) >= 6:
            df.columns = ["日時", "INDEX", "NAME", "STATUS", "項目A", "項目B"] + list(df.columns[6:])
        elif len(df.columns) >= 3:
            df.columns = ["日時", "項目A", "項目B"] + list(df.columns[3:])

        df["日時"] = pd.to_datetime(df["日時"], format="mixed", errors='coerce')
        df = df.dropna(subset=["日時"])
        df = df[(df["日時"] >= start_time) & (df["日時"] <= end_time)]
        df = df.sort_values("日時").reset_index(drop=True)

        df["差分A"] = df["項目A"].diff().clip(lower=0)
        df["bps_A"] = (df["差分A"] * 8) / 300
        df = df.dropna(subset=["bps_A"]).reset_index(drop=True)

        fig.set_facecolor("#f0f0f0")
        ax.set_facecolor("#ffffff")

        if not df.empty:
            # 💡 【重要修正】ここで確実に `df_filtered` を定義します
            max_points = 1000
            if len(df) > max_points:
                step = len(df) // max_points
                df_filtered = df.iloc[::step].copy().reset_index(drop=True)
            else:
                df_filtered = df.copy()

            # 重複の削除
            df_filtered = df_filtered[~df_filtered["日時"].duplicated(keep='first')].reset_index(drop=True)

            # 通常のトラフィック線を描画
            ax.plot(df_filtered.index, df_filtered["bps_A"], color="#006600", linewidth=0.8, label="In (bps)")
            ax.fill_between(df_filtered.index, df_filtered["bps_A"], color="#00eb0c", alpha=0.4)

            # ─── 🧠 Merlion による異常検知処理 ───
            merlion_df = df_filtered[["日時", "bps_A"]].set_index("日時")

            if len(merlion_df) > 2:
                time_series = TimeSeries.from_pd(merlion_df)
                
                # 指定された値に応じてモデルを選択
                """
                if logic == "isolation_forest":
                    config = IsolationForestConfig()
                    model = IsolationForest(config)
                    title_prefix = "【異常検知: 総合外れ値】"
                else:
                    config = StatThresholdConfig(alpha=4.0)
                    model = StatThreshold(config)
                    title_prefix = "【異常検知: 直近急変重視】"
                """
                

            # ─── 🧠 Merlion による異常検知処理 ───
            # 💡 【バースト重視のデータ前処理】
            # 選択されたロジックが「バースト検出」の場合、あらかじめ直近10分間（2サンプル分）の移動平均を計算して突発トゲを除去

            """
            異常検知モード

            1. 直近の急変重視 (StatThreshold)
                	• 一瞬のトラフィック急増（突発スパイク、DDoSの初期兆候など）を敏感に捉えます。
            2. 連続バースト重視 (Pandas Rolling + Isolation Forest)
                	• 短時間のトゲを無視し、大容量データの流出や長時間のバックアップなど「高負荷が持続している塊」を狙い撃ちで捉えます。
            3. 総合的な外れ値重視 (Isolation Forest)
                	• 指定期間全体の中で、絶対的な通信量が最も高くなっているピークポイントを捉えます。
            """

            if logic == "burst_detector":
                df_filtered["bps_for_ai"] = df_filtered["bps_A"].rolling(window=2, min_periods=1).mean()
                title_prefix = "【異常検知: 連続バースト重視】"
            else:
                df_filtered["bps_for_ai"] = df_filtered["bps_A"]
                if logic == "isolation_forest":
                    title_prefix = "【異常検知: 総合外れ値】"
                else:
                    title_prefix = "【異常検知: 直近急変重視】"

            merlion_df = df_filtered[["日時", "bps_for_ai"]].set_index("日時")

            if len(merlion_df) > 2:
                time_series = TimeSeries.from_pd(merlion_df)
                
                # 💡 インポートエラーを防ぐため、既に確実に動くことが実証されている2大モデルを再利用
                if logic == "isolation_forest":
                    config = IsolationForestConfig()
                    model = IsolationForest(config)
                elif logic == "burst_detector":
                    # バースト検出時も、移動平均データに対して IsolationForest を適用（塊を捉えるのが得意）
                    config = IsolationForestConfig()
                    model = IsolationForest(config)
                else:
                    # stat_threshold (直近の急変重視)
                    config = StatThresholdConfig(alpha=4.0)
                    model = StatThreshold(config)
                
                # 計算を実行
                anom_scores = await run_in_threadpool(model.train, time_series)
                anom_scores_df = anom_scores.to_pd()

                scores_map = pd.DataFrame(anom_scores_df)
                scores_map.columns = ["score"]
                
                df_filtered = df_filtered.merge(scores_map, left_on="日時", right_index=True, how="left")
                df_filtered["score"] = df_filtered["score"].fillna(0)
                
                # バースト検出時は、継続的な負荷に対してスコアが集中するため、上位3%を赤くプロット
                percentile_val = 0.97 if logic == "burst_detector" else 0.98
                threshold = df_filtered["score"].quantile(percentile_val)
                anomalies = df_filtered[df_filtered["score"] >= threshold]

                if not anomalies.empty:
                    ax.scatter(anomalies.index, anomalies["bps_A"], color="red", s=15, zorder=5, label="Anomaly")
            else:
                print("[情報] 異常検知を行うための十分なデータ数がありません。")
                title_prefix = "【異常検知】"



            # X軸の目盛りマッピング
            ax.set_xlim(0, len(df_filtered) - 1)
            tick_indices = [0, len(df_filtered)//4, len(df_filtered)//2, (len(df_filtered)*3)//4, len(df_filtered)-1]
            tick_labels = [df_filtered["日時"].iloc[i].strftime(x_tick_format) for i in tick_indices if i < len(df_filtered)]
            ax.set_xticks(tick_indices)
            ax.set_xticklabels(tick_labels, rotation=30, ha="right")
        else:
            ax.text(0.5, 0.5, "指定された時間帯に\nデータがありません", ha="center", va="center", color="red")
            title_prefix = "【異常検知】"

        # Y軸の単位パース
        max_val = df_filtered["bps_A"].max() if not df_filtered.empty else 0
        if max_val >= 1_000_000:
            ax.get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: f"{x / 1_000_000:,.1f} M"))
            ax.set_ylabel("転送速度 (Mbps)")
        elif max_val >= 1_000:
            ax.get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: f"{x / 1_000:,.0f} K"))
            ax.set_ylabel("転送速度 (Kbps)")
        else:
            ax.get_yaxis().set_major_formatter(matplotlib.ticker.StrMethodFormatter("{x:,.0f}"))
            ax.set_ylabel("転送速度 (bps)")

        ax.set_title(f"{title_prefix} {switch}: {filename.replace('.csv', '')}", fontsize=11, loc="left", color="darkred")
        ax.set_xlabel(x_label_text)
    except Exception as e:
        traceback.print_exc()
        ax.text(0.5, 0.5, f"Error: {type(e).__name__}", ha="center", va="center", color="red")

    fig.legend(bbox_to_anchor=(0.13, -0.04), loc='lower left', ncol=2, frameon=True)
    plt.subplots_adjust(bottom=0.25)
    fig.tight_layout()

    img = io.BytesIO()
    await run_in_threadpool(fig.savefig, img, format="png", bbox_inches="tight", dpi=120)
    img.seek(0)
    fig.clear()
    return StreamingResponse(img, media_type="image/png")






if __name__ == "__main__":
    import uvicorn
    if not os.path.exists(BASE_DIR):
        os.makedirs(BASE_DIR)
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
