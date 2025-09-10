# dashboard_gallery.py
# -*- coding: utf-8 -*-

import os, json, glob, math, base64, io
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import seaborn as sns

ARTIFACTS_DIR = Path("artifacts")
RUN_GLOB      = "run_*"
LOG_FILE      = Path("mcp_runs.jsonl")
EXCEL_FILE    = Path("RAG_database.xlsx")  # for experimental fallback

sns.set_theme(style="white")  # clean, no gridlines

# ---------- helpers ----------
def _load_jsonl(path: Path):
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if not line: continue
            try: rows.append(json.loads(line))
            except: pass
    return rows

def _collect_runs(rows):
    """Map run_id -> {final, prompt, ...}"""
    by_id = {}
    for r in rows:
        rid = r.get("run_id")
        if not rid: continue
        by_id.setdefault(rid, {})
        stage = r.get("stage")
        by_id[rid][stage] = r.get("payload", {})
    # keep only those with final artifacts
    out = {}
    for rid, stages in by_id.items():
        if "final" in stages and "report_path" in stages["final"]:
            out[rid] = stages
    return out

def _read_profile_json(path):
    try:
        return json.load(open(path, "r", encoding="utf-8"))
    except Exception:
        return []

def _mk_rel(p: Path):
    try:
        return str(p.relative_to(ARTIFACTS_DIR))
    except Exception:
        return str(p)

# --- very light experimental fallback: first curve in Excel ---
def parse_first_exp_profile(xlsx: Path):
    if not xlsx.exists(): return []
    try:
        xl = pd.ExcelFile(xlsx)
        for sh in xl.sheet_names:
            df = pd.read_excel(xlsx, sheet_name=sh, header=None, dtype=str).fillna("")
            n_rows, n_cols = df.shape
            def to_float(s):
                s = str(s).replace("µ","μ").replace("%","")
                m = pd.to_numeric(pd.Series([s]), errors="coerce").iloc[0]
                if pd.isna(m):
                    import re
                    mm = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(s))
                    return float(mm.group(0)) if mm else None
                return float(m)
            data = df.applymap(to_float).values
            # pick two numeric columns with monotone time and % in [0,110]
            cand = [c for c in range(n_cols) if np.sum(~pd.isna(data[:,c]))>=4]
            for tc in cand:
                times = data[:, tc]
                times = [t for t in times if t is not None]
                if len(times)<4 or any(b<a for a,b in zip(times, times[1:])): continue
                for vc in cand:
                    if vc==tc: continue
                    vals = data[:, vc]
                    pairs = [(t,v) for t,v in zip(data[:,tc], vals) if t is not None and v is not None]
                    if len(pairs)<4: continue
                    T = [p[0] for p in pairs]; Y = [p[1] for p in pairs]
                    if np.mean([(0<=y<=110) for y in Y])<0.8: continue
                    arr = sorted(zip(T,Y), key=lambda x:x[0])
                    return [{"time": float(t), "dissolved": float(min(max(y,0),100))} for t,y in arr]
    except Exception:
        return []
    return []

EXP_FALLBACK = parse_first_exp_profile(EXCEL_FILE)

def _interp(profile, grid):
    xs = np.array([p["time"] for p in profile], float)
    ys = np.array([p["dissolved"] for p in profile], float)
    ys = np.clip(np.maximum.accumulate(ys), 0, 100)
    return np.interp(grid, xs, ys, left=ys[0], right=ys[-1])

def _rate(times, values):
    t = np.array(times, float); y = np.array(values, float)
    dt = np.diff(t); dy = np.diff(y)
    rate = np.divide(dy, dt, out=np.zeros_like(dy), where=dt>0)
    tt = (t[:-1]+t[1:])/2.0
    return tt, rate

def _f2(ref, tst):
    if len(ref)<3 or len(ref)!=len(tst): return None
    diffsq = np.mean((np.array(ref)-np.array(tst))**2)
    try:
        return 50*np.log10((1+diffsq)**-0.5*100)
    except Exception:
        return None

def _no_grid(ax):
    ax.grid(False)
    sns.despine(ax=ax, trim=True)

# ---------- main render ----------
def build_gallery():
    rows = _load_jsonl(LOG_FILE)
    runs = _collect_runs(rows)
    if not runs:
        print("No runs found.")
        return

    # map: run_id -> prompt_filename (from prompt or final)
    prompt_names = {}
    for rid, st in runs.items():
        pf = (st.get("final", {}).get("prompt_filename")
              or st.get("prompt", {}).get("prompt_filename"))
        if not pf:
            # fallback: first *.txt in the run dir
            rdir = ARTIFACTS_DIR / f"run_{rid}"
            txts = list(rdir.glob("*.txt"))
            if txts: pf = txts[0].name
        prompt_names[rid] = pf or "unknown.txt"

    # build HTML + plots
    cards_html = []

    for rid, st in runs.items():
        final = st["final"]
        report_path  = Path(final["report_path"])
        profile_path = Path(final["profile_json"])
        rdir = report_path.parent

        # load predicted profile
        prof = _read_profile_json(profile_path)
        if not prof:
            # skip this card if no profile
            continue
        t_pred = [p["time"] for p in prof]
        y_pred = [p["dissolved"] for p in prof]

        # experimental: fallback (same for all) – OK for demo
        exp = EXP_FALLBACK
        t_exp = [p["time"] for p in exp] if exp else []
        y_exp = [p["dissolved"] for p in exp] if exp else []

        # define a comparison grid
        grid = sorted(set(t_pred + (t_exp if t_exp else [])))
        if len(grid) < 5:
            grid = [0,5,10,15,30,45,60,90,120,180,360][:max(5,len(t_pred))]
        z_pred = _interp(prof, grid)
        z_exp  = _interp(exp, grid) if exp else None

        # compute QC numbers (optional)
        f2 = _f2(z_exp, z_pred) if z_exp is not None else None
        mae = float(np.mean(np.abs(z_pred - z_exp))) if z_exp is not None else None
        rmse = float(np.sqrt(np.mean((z_pred - z_exp)**2))) if z_exp is not None else None

        # ---------- plots ----------
        fig_w, fig_h = 6, 6.6

        # 1) Dissolution overlay
        fig1, ax1 = plt.subplots(figsize=(fig_w, 3))
        ax1.plot(grid, z_pred, marker="o", linewidth=1.8, label="Predicted")
        if z_exp is not None:
            ax1.plot(grid, z_exp, marker="o", linewidth=1.8, label="Experimental")
        ax1.set_title("Dissolution Profile")
        ax1.set_xlabel("Time (min)")
        ax1.set_ylabel("Dissolved (%)")
        ax1.legend(frameon=False)
        _no_grid(ax1)
        p1 = rdir / f"dash_{rid}_overlay.png"
        fig1.tight_layout(); fig1.savefig(p1, dpi=140); plt.close(fig1)

        # 2) Error (Pred-Exp)
        fig2, ax2 = plt.subplots(figsize=(fig_w, 2.6))
        if z_exp is not None:
            err = z_pred - z_exp
            ax2.plot(grid, err, marker="o", linewidth=1.4)
        else:
            ax2.text(0.5,0.5,"No experimental curve", ha="center", va="center", transform=ax2.transAxes)
        ax2.set_title("Error (Pred - Exp)")
        ax2.set_xlabel("Time (min)")
        ax2.set_ylabel("Δ%")
        _no_grid(ax2)
        p2 = rdir / f"dash_{rid}_error.png"
        fig2.tight_layout(); fig2.savefig(p2, dpi=140); plt.close(fig2)

        # 3) Release rate d%/dt
        fig3, ax3 = plt.subplots(figsize=(fig_w, 2.6))
        tt, rr = _rate(grid, z_pred)
        ax3.plot(tt, rr, marker="o", linewidth=1.4)
        ax3.set_title("Release Rate (d%/dt)")
        ax3.set_xlabel("Time (min)")
        ax3.set_ylabel("Rate (%/min)")
        _no_grid(ax3)
        p3 = rdir / f"dash_{rid}_rate.png"
        fig3.tight_layout(); fig3.savefig(p3, dpi=140); plt.close(fig3)

        # 4) QC summary bars (T50/T90 from predicted + smoothness proxy)
        t_pred_arr = np.array(grid); y_pred_arr = np.array(z_pred)
        def _tx(x):
            if y_pred_arr[0] >= x: return float(t_pred_arr[0])
            if y_pred_arr[-1] < x: return float("nan")
            return float(np.interp(x, y_pred_arr, t_pred_arr))
        T50 = _tx(50); T90 = _tx(90)
        smooth = float(np.mean(np.abs(np.diff(y_pred_arr, n=2)))) if len(y_pred_arr)>=3 else 0.0

        fig4, ax4 = plt.subplots(figsize=(fig_w, 2.8))
        bars = [("T50", T50 if not math.isnan(T50) else 0.0),
                ("T90", T90 if not math.isnan(T90) else 0.0),
                ("SMOOTHNESS ABS DDIFF", smooth)]
        ax4.bar([b[0] for b in bars], [b[1] for b in bars])
        ax4.set_title("QC Summary")
        for spine in ["top","right"]: ax4.spines[spine].set_visible(False)
        _no_grid(ax4)
        # f2/MAE/RMSE annotation
        if f2 is not None and mae is not None and rmse is not None:
            ax4.text(0.5, -0.35, f"f2: {f2:.1f} | MAE: {mae:.2f}% | RMSE: {rmse:.2f}%",
                     ha="center", va="top", transform=ax4.transAxes, fontsize=9)
        p4 = rdir / f"dash_{rid}_qc.png"
        fig4.tight_layout(); fig4.savefig(p4, dpi=140, bbox_inches="tight"); plt.close(fig4)

        # ---------- card HTML ----------
        rel1 = _mk_rel(p1); rel2 = _mk_rel(p2); rel3 = _mk_rel(p3); rel4 = _mk_rel(p4)
        rel_report = _mk_rel(report_path)
        rel_profile = _mk_rel(profile_path)
        prompt_name = prompt_names.get(rid, "unknown.txt")

        card = f"""
        <div class="card">
          <div class="run-title"><b>Run:</b> {rid}</div>
          <div class="meta">prompt: <code>{prompt_name}</code></div>

          <img src="{rel1}" />
          <img src="{rel2}" />
          <img src="{rel3}" />
          <img src="{rel4}" />

          <div class="links">
            <a href="{rel_report}" target="_blank">report.md</a> |
            <a href="{rel_profile}" target="_blank">profile.json</a>
          </div>

          <details class="md-preview">
            <summary>Preview report.md</summary>
            <iframe src="{rel_report}" loading="lazy"></iframe>
          </details>
        </div>
        """
        cards_html.append(card)

    # write index.html
    html = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8"/>
<title>PharmaDissolve-MCP — Run Gallery (no contour)</title>
<style>
body {{ font-family: -apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Helvetica,Arial,sans-serif; margin: 28px; }}
h1 {{ margin: 0 0 18px 0; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(380px, 1fr)); grid-gap: 18px; }}
.card {{
  border: 1px solid #eaeaea; border-radius: 10px; padding: 14px; box-shadow: 0 1px 2px rgba(0,0,0,.03);
}}
.card img {{ width: 100%; height: auto; display: block; margin: 10px 0; border-radius: 6px; }}
.run-title {{ margin-bottom: 6px; }}
.meta {{ color: #666; font-size: 12px; margin-bottom: 8px; }}
.links a {{ text-decoration: none; }}
.md-preview summary {{ cursor: pointer; margin-top: 6px; }}
.md-preview iframe {{
  width: 100%; height: 420px; border: 1px solid #eee; border-radius: 6px; margin-top: 8px;
}}
footer {{ margin-top: 22px; font-size: 12px; color: #666; }}
</style>
</head>
<body>
  <h1>PharmaDissolve-MCP — Run Gallery (no contour)</h1>
  <div class="grid">
    {"".join(cards_html)}
  </div>
  <footer>Data source: <code>{LOG_FILE}</code> · Excel fallback: <code>{EXCEL_FILE.name}</code></footer>
</body>
</html>"""
    (ARTIFACTS_DIR / "index.html").write_text(html, encoding="utf-8")
    print(f"✅ Gallery written to: {ARTIFACTS_DIR / 'index.html'}")

if __name__ == "__main__":
    build_gallery()
    print("Serve with:  python -m http.server 8000 -d artifacts")
