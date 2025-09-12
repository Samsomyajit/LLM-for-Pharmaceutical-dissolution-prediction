# visualize_runs_basic.py
# Build a static run gallery with seaborn plots (no gridlines).
# Usage:
#   python visualize_runs_basic.py \
#       --excel RAG_database.xlsx \
#       --log mcp_runs.jsonl \
#       --out dashboards_basic

import os, re, json, argparse, math
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="white")  # clean look, no grid by default

# --------------------------- Utils ---------------------------

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def _no_grid(ax):
    ax.grid(False)
    ax.set_axisbelow(False)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

def _safe_rel(from_dir: str, path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    return os.path.relpath(os.path.abspath(path), os.path.abspath(from_dir))

# ---------------------- Profiles & Metrics ----------------------

def interp_profile(profile: List[Dict[str, float]], grid: List[float]) -> List[Dict[str, float]]:
    if not profile:
        return []
    xs = np.array([p["time"] for p in profile], float)
    ys = np.array([p["dissolved"] for p in profile], float)
    ys = np.clip(np.maximum.accumulate(ys), 0.0, 100.0)  # monotone + bounds
    gy = np.interp(grid, xs, ys, left=ys[0], right=ys[-1])
    return [{"time": float(t), "dissolved": float(v)} for t, v in zip(grid, gy)]

def f2_on_grid(ref: List[Dict[str, float]], tst: List[Dict[str, float]]) -> Optional[float]:
    n = len(ref)
    if n < 3 or n != len(tst):
        return None
    diffsq = sum((ref[i]["dissolved"] - tst[i]["dissolved"]) ** 2 for i in range(n)) / n
    try:
        return 50 * math.log10((1 + diffsq) ** -0.5 * 100)
    except Exception:
        return None

def diff_profile(pred: List[Dict[str, float]], ref: List[Dict[str, float]]) -> Tuple[np.ndarray, np.ndarray]:
    """Return grid and (pred - ref) values on a common grid."""
    grid = sorted(set([p["time"] for p in pred] + [p["time"] for p in ref]))
    if len(grid) < 5:
        grid = [0, 5, 10, 15, 30, 45, 60]
    r = interp_profile(ref, grid)
    z = interp_profile(pred, grid)
    err = np.array([z[i]["dissolved"] - r[i]["dissolved"] for i in range(len(grid))], float)
    return np.array(grid, float), err

def rate_profile(profile: List[Dict[str, float]]) -> Tuple[np.ndarray, np.ndarray]:
    """Finite-difference rate d(%)/dt over midpoints."""
    t = np.array([p["time"] for p in profile], float)
    y = np.array([p["dissolved"] for p in profile], float)
    if len(t) < 2:
        return t, np.zeros_like(t)
    dt = np.diff(t)
    dy = np.diff(y)
    # midpoints for plotting
    mids = t[:-1] + dt / 2.0
    rate = np.divide(dy, dt, out=np.zeros_like(dy), where=dt != 0)
    return mids, rate

# ---------------------- Excel experimental curve ----------------------

def _to_float(s) -> Optional[float]:
    s = str(s)
    s = s.replace("µ", "μ").replace("%", "")
    m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
    return float(m.group(0)) if m else None

def parse_sheet_timecurve(xls_path: str, sheet: str) -> Optional[List[Dict[str, float]]]:
    """Robustly extract a (time, dissolved) curve from a sheet, or None if not found."""
    try:
        df = pd.read_excel(xls_path, sheet_name=sheet, header=None, dtype=str).fillna("")
    except Exception:
        return None
    n_rows, n_cols = df.shape
    data_num = [[_to_float(df.iat[r, c]) for c in range(n_cols)] for r in range(n_rows)]

    def mono_increasing(xs):
        xs = [x for x in xs if x is not None]
        return len(xs) >= 4 and all(b >= a for a, b in zip(xs, xs[1:]))

    # Header-ish time columns
    header_like = re.compile(r"(time|t\s*\(.*min.*\)|min)", re.IGNORECASE)
    time_cols = [c for c in range(n_cols) if any(header_like.search(str(df.iat[r, c])) for r in range(min(n_rows, 5)))]

    best = None
    if time_cols:
        for tc in time_cols:
            for vc in range(n_cols):
                if vc == tc:
                    continue
                times, vals = [], []
                for r in range(n_rows):
                    t = data_num[r][tc]
                    v = data_num[r][vc]
                    if t is None or v is None:
                        continue
                    times.append(t)
                    vals.append(v)
                if len(times) >= 4 and mono_increasing(times):
                    in_rng = np.mean([(0.0 <= y <= 110.0) for y in vals]) if vals else 0
                    if in_rng >= 0.8:
                        score = len(times) + 0.5 * in_rng
                        best = {"times": times, "vals": vals, "score": score}
                        break

    # Fallback: brute-force
    if best is None:
        cand_cols = []
        for c in range(n_cols):
            col = [data_num[r][c] for r in range(n_rows)]
            vals = [v for v in col if v is not None]
            if len(vals) >= 4:
                cand_cols.append(c)
        for tc in cand_cols:
            times = [data_num[r][tc] for r in range(n_rows)]
            if not mono_increasing(times):
                continue
            for vc in cand_cols:
                if vc == tc:
                    continue
                vals = [data_num[r][vc] for r in range(n_rows)]
                pairs = [(t, v) for t, v in zip(times, vals) if t is not None and v is not None]
                if len(pairs) < 4:
                    continue
                T = [p[0] for p in pairs]
                Y = [p[1] for p in pairs]
                in_rng = np.mean([(0.0 <= y <= 110.0) for y in Y])
                if in_rng < 0.8:
                    continue
                try:
                    corr = np.corrcoef(T, Y)[0, 1]
                except Exception:
                    corr = 0.0
                if corr < -0.1:
                    continue
                score = len(T) + corr
                if best is None or score > best["score"]:
                    best = {"times": T, "vals": Y, "score": score}

    if not best:
        return None
    arr = sorted(zip(best["times"], best["vals"]), key=lambda x: x[0])
    out = []
    last = -1e9
    for t, y in arr:
        if t is None or y is None:
            continue
        if t < last:
            continue
        last = t
        out.append({"time": float(t), "dissolved": float(np.clip(y, 0.0, 100.0))})
    return out if len(out) >= 4 else None

# ---------------------- Plotting ----------------------

def plot_overlay(run_id: str, pred: List[Dict[str, float]], exp: Optional[List[Dict[str, float]]], outdir: str) -> str:
    ensure_dir(outdir)
    fig, ax = plt.subplots(figsize=(6, 3.6), dpi=160)
    _no_grid(ax)

    if pred:
        t = [p["time"] for p in pred]
        y = [p["dissolved"] for p in pred]
        sns.lineplot(x=t, y=y, ax=ax, marker="o", linewidth=2, label="Predicted")
    if exp:
        te = [p["time"] for p in exp]
        ye = [p["dissolved"] for p in exp]
        sns.lineplot(x=te, y=ye, ax=ax, marker="o", linewidth=2, label="Experimental")

    ax.set_title("Dissolution Profile")
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Dissolved (%)")
    ax.set_ylim(0, 105)
    ax.legend(frameon=False)
    fig.tight_layout()
    path = os.path.abspath(os.path.join(outdir, f"{run_id}_overlay.png"))
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path

def plot_error(run_id: str, pred: List[Dict[str, float]], exp: Optional[List[Dict[str, float]]], outdir: str) -> Optional[str]:
    if not exp or not pred:
        return None
    ensure_dir(outdir)
    grid, err = diff_profile(pred, exp)
    fig, ax = plt.subplots(figsize=(6, 3.6), dpi=160)
    _no_grid(ax)
    sns.lineplot(x=grid, y=err, ax=ax, marker="o", linewidth=2)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_title("Error (Pred - Exp)")
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Δ%")
    fig.tight_layout()
    path = os.path.abspath(os.path.join(outdir, f"{run_id}_error.png"))
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path

def plot_rate(run_id: str, pred: List[Dict[str, float]], outdir: str) -> Optional[str]:
    if not pred or len(pred) < 2:
        return None
    ensure_dir(outdir)
    t, r = rate_profile(pred)
    fig, ax = plt.subplots(figsize=(6, 3.6), dpi=160)
    _no_grid(ax)
    sns.lineplot(x=t, y=r, ax=ax, marker="o", linewidth=2)
    ax.set_title("Release Rate (d%/dt)")
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Rate (%/min)")
    fig.tight_layout()
    path = os.path.abspath(os.path.join(outdir, f"{run_id}_rate.png"))
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path

def plot_qc_bars(run_id: str, metrics: Dict[str, Any], outdir: str) -> str:
    ensure_dir(outdir)
    # Select a few keys if present
    keys = []
    for k in ["mae", "rmse", "T50", "T90", "smoothness_abs_ddiff"]:
        if metrics.get(k) is not None:
            keys.append(k)
    if not keys:
        # fallback single bar showing monotonicity fraction
        keys = ["monotonicity_fraction"]
    vals = [metrics.get(k) for k in keys]
    labels = [k.replace("_", " ").upper() for k in keys]

    fig, ax = plt.subplots(figsize=(6, 3.6), dpi=160)
    _no_grid(ax)
    df = pd.DataFrame({"metric": labels, "value": vals})
    sns.barplot(data=df, x="metric", y="value", ax=ax)
    ax.set_title("QC Summary")
    ax.set_xlabel("")
    ax.set_ylabel("")
    for tick in ax.get_xticklabels():
        tick.set_rotation(20)
        tick.set_ha("right")
    fig.tight_layout()
    path = os.path.abspath(os.path.join(outdir, f"{run_id}_qc.png"))
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path

# ---------------------- HTML ----------------------

def write_index_html(cards: List[Dict[str, Any]], outdir: str) -> str:
    html = [
        "<!doctype html><meta charset='utf-8'/>",
        "<title>PharmaDissolve-MCP — Run Gallery (no contour)</title>",
        "<style>",
        "body{font-family:Inter,system-ui,Arial,sans-serif;padding:24px;}",
        ".grid{display:flex;flex-wrap:wrap;gap:18px}",
        ".card{border:1px solid #e5e7eb;border-radius:10px;padding:12px;width:460px;background:#fff;box-shadow:0 1px 2px rgba(0,0,0,.05)}",
        ".card img{width:100%;display:block;margin:6px 0;border-radius:6px}",
        ".links a{color:#2563eb;text-decoration:none;margin-right:10px}",
        ".links a:hover{text-decoration:underline}",
        "h1{font-weight:800;margin:0 0 12px}",
        "</style>",
        "<h1>PharmaDissolve-MCP — Run Gallery (no contour)</h1>",
        "<div class='grid'>",
    ]

    for c in cards:
        html.append("<div class='card'>")
        html.append(f"<div><b>Run:</b> {c['run_id']}</div>")

        for k in ["overlay", "error", "rate", "qc"]:
            rp = _safe_rel(outdir, c.get(k))
            if rp:
                html.append(f"<img src='{rp}' alt='{k}'>")

        # links
        links = []
        rp = _safe_rel(outdir, c.get("report"))
        pj = _safe_rel(outdir, c.get("profile"))
        if rp: links.append(f"<a href='{rp}'>report.md</a>")
        if pj: links.append(f"<a href='{pj}'>profile.json</a>")
        if links:
            html.append("<div class='links'>" + " | ".join(links) + "</div>")

        html.append("</div>")

    html.append("</div>")
    ensure_dir(outdir)
    out_path = os.path.join(outdir, "index.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(html))
    return out_path

# ---------------------- Run loading ----------------------

def load_runs(log_path: str) -> Dict[str, Dict[str, Any]]:
    """Return {run_id: final_record_payload} for latest final per run."""
    runs: Dict[str, Dict[str, Any]] = {}
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if rec.get("stage") == "final":
                rid = rec.get("run_id")
                runs[rid] = rec.get("payload", {})
    return runs

def load_profile(path: str) -> Optional[List[Dict[str, float]]]:
    try:
        obj = json.load(open(path, "r", encoding="utf-8"))
    except Exception:
        return None
    if isinstance(obj, list):
        prof = obj
    else:
        prof = obj.get("profile") or obj
    out = []
    for p in prof:
        if isinstance(p, dict) and "time" in p and "dissolved" in p:
            out.append({"time": float(p["time"]), "dissolved": float(p["dissolved"])})
    return out if out else None

# ---------------------- Main build ----------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--excel", required=True, help="Path to RAG_database.xlsx")
    ap.add_argument("--log", required=True, help="Path to mcp_runs.jsonl")
    ap.add_argument("--out", default="dashboards_basic", help="Output directory for gallery")
    args = ap.parse_args()

    gallery_dir = os.path.abspath(args.out)
    runs_dir = os.path.join(gallery_dir, "runs")
    ensure_dir(runs_dir)

    finals = load_runs(args.log)
    cards: List[Dict[str, Any]] = []

    for rid, payload in finals.items():
        report = payload.get("report_path")
        profile = payload.get("profile_json")
        metrics = payload.get("metrics") or {}
        sources = payload.get("sources") or []
        pred = load_profile(profile) if profile and os.path.exists(profile) else None

        # Try to locate an experimental curve from the first source with a real sheet
        exp = None
        for s in sources:
            sheet = s.get("sheet")
            if sheet:
                exp = parse_sheet_timecurve(args.excel, sheet)
                if exp:
                    break

        # Plots
        overlay = plot_overlay(rid, pred or [], exp, runs_dir)
        error = plot_error(rid, pred or [], exp, runs_dir)
        rate = plot_rate(rid, pred or [], runs_dir)
        qc = plot_qc_bars(rid, metrics, runs_dir)

        cards.append({
            "run_id": rid,
            "overlay": overlay,
            "error": error,
            "rate": rate,
            "qc": qc,
            "report": report if report and os.path.exists(report) else None,
            "profile": profile if profile and os.path.exists(profile) else None,
        })

    index_path = write_index_html(cards, gallery_dir)
    print("Gallery written to:", index_path)
    print("Tip: serve locally with")
    print(f"  python -m http.server -d {gallery_dir} 8000")

if __name__ == "__main__":
    main()
