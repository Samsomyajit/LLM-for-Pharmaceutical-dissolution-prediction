# dashboard_gallary.py
# FINAL VERSION: 2x2 grid, large fonts, rendered Markdown, fixed paths, compact + expanded plots

import os, re, json, argparse, math, datetime, shutil
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
import plotly
import plotly.io as pio
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import markdown  # For rendering Markdown reports

pio.templates.default = "simple_white"

# --------------------------- Utils ---------------------------

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def _safe_rel(from_dir: str, path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    return os.path.relpath(os.path.abspath(path), os.path.abspath(from_dir))

def _parse_ts(ts: str) -> Optional[datetime.datetime]:
    try:
        return datetime.datetime.fromisoformat(ts)
    except Exception:
        return None

def plotly_div(fig: go.Figure, div_id: str) -> str:
    """Return an embeddable div for the figure."""
    return plotly.io.to_html(
        fig,
        include_plotlyjs=False,
        full_html=False,
        div_id=div_id,
        config={
            "displayModeBar": True,
            "displaylogo": False,
            "modeBarButtonsToAdd": ["zoom2d", "pan2d", "resetScale2d"],
            "modeBarButtonsToRemove": ["select2d", "lasso2d"],
            "responsive": True
        }
    )

# ---------------------- Profiles & Metrics ----------------------

def interp_profile(profile: List[Dict[str, float]], grid: List[float]) -> List[Dict[str, float]]:
    if not profile:
        return []
    xs = np.array([p["time"] for p in profile], float)
    ys = np.array([p["dissolved"] for p in profile], float)
    ys = np.clip(np.maximum.accumulate(ys), 0.0, 100.0)
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
    grid = sorted(set([p["time"] for p in pred] + [p["time"] for p in ref]))
    if len(grid) < 5:
        grid = [0, 5, 10, 15, 30, 45, 60]
    r = interp_profile(ref, grid)
    z = interp_profile(pred, grid)
    err = np.array([z[i]["dissolved"] - r[i]["dissolved"] for i in range(len(grid))], float)
    return np.array(grid, float), err

def rate_profile(profile: List[Dict[str, float]]) -> Tuple[np.ndarray, np.ndarray]:
    t = np.array([p["time"] for p in profile], float)
    y = np.array([p["dissolved"] for p in profile], float)
    if len(t) < 2:
        return t, np.zeros_like(t)
    dt = np.diff(t)
    dy = np.diff(y)
    mids = t[:-1] + dt / 2.0
    rate = np.divide(dy, dt, out=np.zeros_like(dy), where=dt != 0)
    return mids, rate

# ---------------------- Excel experimental curve ----------------------

def _to_float(s) -> Optional[float]:
    s = str(s).replace("µ", "μ").replace("%", "")
    m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
    return float(m.group(0)) if m else None

def parse_sheet_timecurve(xls_path: str, sheet: str) -> Optional[List[Dict[str, float]]]:
    try:
        df = pd.read_excel(xls_path, sheet_name=sheet, header=None, dtype=str).fillna("")
    except Exception:
        return None
    n_rows, n_cols = df.shape
    data_num = [[_to_float(df.iat[r, c]) for c in range(n_cols)] for r in range(n_rows)]
    def mono_increasing(xs):
        xs = [x for x in xs if x is not None]
        return len(xs) >= 4 and all(b >= a for a, b in zip(xs, xs[1:]))
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

# ---------------------- Plotly builders (per-run) ----------------------

def overlay_div(run_id: str, pred: List[Dict[str, float]], exp: Optional[List[Dict[str, float]]]) -> str:
    fig = go.Figure()
    if pred:
        fig.add_trace(go.Scatter(
            x=[p["time"] for p in pred], 
            y=[p["dissolved"] for p in pred],
            mode="lines+markers",
            name="Predicted",
            line=dict(color='#1f77b4', width=3),
            marker=dict(size=8, symbol='circle'),
            hovertemplate="Time: %{x:.1f} min<br>Dissolved: %{y:.1f}%<extra></extra>"
        ))
    if exp:
        fig.add_trace(go.Scatter(
            x=[p["time"] for p in exp], 
            y=[p["dissolved"] for p in exp],
            mode="lines+markers",
            name="Experimental",
            line=dict(color='#ff7f0e', width=3, dash='dash'),
            marker=dict(size=8, symbol='square'),
            hovertemplate="Time: %{x:.1f} min<br>Dissolved: %{y:.1f}%<extra></extra>"
        ))
    if not pred and not exp:
        fig.add_annotation(
            text="<i>No profile data</i>",
            showarrow=False,
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            font=dict(size=18, color="#666")
        )
    fig.update_layout(
        height=450,
        width=600,
        margin=dict(l=50, r=30, t=30, b=100),
        font=dict(family="Inter, sans-serif", size=14),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.12,
            xanchor="left",
            x=-0.1,
            bgcolor="rgba(255,255,255,0.8)",
            borderwidth=1,
            font=dict(size=14)
        ),
        xaxis=dict(
            title="Time (minutes)",
            title_font=dict(size=16),
            showgrid=True,
            gridcolor='#eee',
            tickfont=dict(size=12)
        ),
        yaxis=dict(
            title="Dissolved (%)",
            title_font=dict(size=16),
            range=[0, 105],
            showgrid=True,
            gridcolor='#eee',
            tickfont=dict(size=12)
        )
    )

    fig.add_annotation(
        text=f"Run {run_id[:8]} — Dissolution Profile",
        xref="paper", yref="paper",
        x=0.5, y=-0.23,
        showarrow=False,
        font=dict(size=18, color="#111", family="Inter, sans-serif"),
        xanchor="center", yanchor="top"
    )

    return plotly_div(fig, f"ov-{run_id}")

def error_div(run_id: str, pred: List[Dict[str, float]], exp: Optional[List[Dict[str, float]]]) -> Optional[str]:
    if not pred or not exp:
        return None
    grid, err = diff_profile(pred, exp)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=grid, y=err, 
        mode="lines+markers",
        line=dict(color='#d62728', width=3),
        marker=dict(size=8),
        hovertemplate="Error: %{y:.2f}%<extra></extra>"
    ))
    fig.add_hline(y=0, line_color="black", line_width=1, line_dash="dash")

    fig.update_layout(
        height=450,
        width=600,
        margin=dict(l=50, r=30, t=40, b=100),
        font=dict(family="Inter, sans-serif", size=14),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(
            title="Time (min)",
            title_font=dict(size=16),
            showgrid=True,
            gridcolor='#eee',
            tickfont=dict(size=12)
        ),
        yaxis=dict(
            title="Δ% (Pred - Exp)",
            title_font=dict(size=16),
            showgrid=True,
            gridcolor='#eee',
            tickfont=dict(size=12)
        )
    )

    fig.add_annotation(
        text="Prediction Error (Δ%)",
        xref="paper", yref="paper",
        x=0.5, y=-0.2,
        showarrow=False,
        font=dict(size=18, color="#111"),
        xanchor="center", yanchor="top"
    )

    return plotly_div(fig, f"err-{run_id}")

def residual_hist_div(run_id: str, pred: List[Dict[str, float]], exp: Optional[List[Dict[str, float]]]) -> Optional[str]:
    if not pred or not exp:
        return None
    grid, err = diff_profile(pred, exp)
    fig = px.histogram(x=err, nbins=min(20, max(5, len(err)//2)))
    fig.update_traces(hovertemplate="Error: %{x:.2f}%<br>Count: %{y}<extra></extra>")

    fig.update_layout(
        height=450,
        width=600,
        margin=dict(l=50, r=30, t=40, b=100),
        font=dict(family="Inter, sans-serif", size=14),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(
            title="Δ% (Pred - Exp)",
            title_font=dict(size=16),
            showgrid=True,
            gridcolor='#eee',
            tickfont=dict(size=12)
        ),
        yaxis=dict(
            title="Count",
            title_font=dict(size=16),
            showgrid=True,
            gridcolor='#eee',
            tickfont=dict(size=12)
        )
    )

    fig.add_annotation(
        text="Residual Histogram",
        xref="paper", yref="paper",
        x=0.5, y=-0.2,
        showarrow=False,
        font=dict(size=18, color="#111"),
        xanchor="center", yanchor="top"
    )

    return plotly_div(fig, f"hist-{run_id}")

def rate_div(run_id: str, pred: List[Dict[str, float]]) -> Optional[str]:
    if not pred or len(pred) < 2:
        return None
    t, r = rate_profile(pred)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=t, y=r, 
        mode="lines+markers",
        line=dict(color='#2ca02c', width=3),
        marker=dict(size=8),
        hovertemplate="Rate: %{y:.2f}%/min<extra></extra>"
    ))

    fig.update_layout(
        height=450,
        width=600,
        margin=dict(l=50, r=30, t=40, b=100),
        font=dict(family="Inter, sans-serif", size=14),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(
            title="Time (min)",
            title_font=dict(size=16),
            showgrid=True,
            gridcolor='#eee',
            tickfont=dict(size=12)
        ),
        yaxis=dict(
            title="Rate (%/min)",
            title_font=dict(size=16),
            showgrid=True,
            gridcolor='#eee',
            tickfont=dict(size=12)
        )
    )

    fig.add_annotation(
        text="Release Rate (d%/dt)",
        xref="paper", yref="paper",
        x=0.5, y=-0.2,
        showarrow=False,
        font=dict(size=18, color="#111"),
        xanchor="center", yanchor="top"
    )

    return plotly_div(fig, f"rate-{run_id}")

def qc_bars_div(run_id: str, metrics: Dict[str, Any]) -> str:
    keys = ["mae", "rmse", "T50", "T90", "smoothness_abs_ddiff"]
    valid_keys = [k for k in keys if metrics.get(k) is not None]
    if not valid_keys:
        valid_keys = ["monotonicity_fraction"]
    vals = [metrics.get(k) for k in valid_keys]
    labels = [k.replace("_", " ").title() for k in valid_keys]

    fig = px.bar(pd.DataFrame({"Metric": labels, "Value": vals}), x="Metric", y="Value", text_auto=True)
    fig.update_traces(
        textfont_size=14,
        textposition="outside",
        cliponaxis=False,
        hovertemplate="%{x}: %{y:.3f}<extra></extra>"
    )

    fig.update_layout(
        height=450,
        width=600,
        margin=dict(l=50, r=30, t=40, b=100),
        font=dict(family="Inter, sans-serif", size=14),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(
            title="",
            title_font=dict(size=16),
            showgrid=False,
            tickfont=dict(size=12)
        ),
        yaxis=dict(
            title="",
            title_font=dict(size=16),
            showgrid=True,
            gridcolor='#eee',
            tickfont=dict(size=12)
        )
    )

    fig.add_annotation(
        text="QC Summary Metrics",
        xref="paper", yref="paper",
        x=0.5, y=-0.2,
        showarrow=False,
        font=dict(size=18, color="#111"),
        xanchor="center", yanchor="top"
    )

    return plotly_div(fig, f"qc-{run_id}")

# ----- TOP SUMMARY PLOTS -----

def f2_timeline_expanded(timeline: List[Dict[str, Any]]) -> Optional[str]:
    df = pd.DataFrame([{"run_id": x["run_id"], "ts": x["ts"], "f2": x.get("f2")} for x in timeline if x.get("f2") is not None])
    if df.empty:
        return None
    df = df.sort_values("ts")
    fig = px.line(df, x="ts", y="f2", markers=True, color_discrete_sequence=['#1f77b4'])
    fig.update_traces(
        hovertemplate="Time: %{x}<br>f₂: %{y:.2f}<extra></extra>",
        marker=dict(size=8),
        line=dict(width=3)
    )

    fig.update_layout(
        height=450,
        width=600,
        margin=dict(l=50, r=30, t=40, b=100),
        font=dict(family="Inter, sans-serif", size=14),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        showlegend=False,
        xaxis=dict(
            title="Run Time",
            title_font=dict(size=16),
            showgrid=True,
            gridcolor='#eee',
            tickfont=dict(size=12),
            tickangle=25
        ),
        yaxis=dict(
            title="f₂ Score (higher is better)",
            title_font=dict(size=16),
            showgrid=True,
            gridcolor='#eee',
            tickfont=dict(size=12)
        )
    )

    fig.add_annotation(
        text="f₂ Score Timeline Across Runs",
        xref="paper", yref="paper",
        x=0.5, y=-0.29,
        showarrow=False,
        font=dict(size=18, color="#111", weight='bold'),
        xanchor="center", yanchor="top"
    )

    return plotly_div(fig, "top-f2-expanded")



def mae_timeline_div(timeline: List[Dict[str, Any]]) -> Optional[str]:
    df = pd.DataFrame([{"run_id": x["run_id"], "ts": x["ts"], "mae": x.get("mae")} for x in timeline if x.get("mae") is not None])
    if df.empty:
        return None
    df = df.sort_values("ts")
    fig = px.line(df, x="ts", y="mae", markers=True, color_discrete_sequence=['#d62728'])
    fig.update_traces(hovertemplate="Time: %{x}<br>MAE: %{y:.2f}%<extra></extra>")

    fig.update_layout(
        height=450,
        width=600,
        margin=dict(l=50, r=30, t=40, b=100),
        font=dict(family="Inter, sans-serif", size=14),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(
            title="Run Time",
            title_font=dict(size=16),
            showgrid=True,
            gridcolor='#eee',
            tickfont=dict(size=12)
        ),
        yaxis=dict(
            title="MAE (%)",
            title_font=dict(size=16),
            showgrid=True,
            gridcolor='#eee',
            tickfont=dict(size=12)
        )
    )

    fig.add_annotation(
        text="MAE Timeline (lower is better)",
        xref="paper", yref="paper",
        x=0.5, y=-0.2,
        showarrow=False,
        font=dict(size=18, color="#111"),
        xanchor="center", yanchor="top"
    )

    return plotly_div(fig, "top-mae")

def t50_t90_scatter_div(rows: List[Dict[str, Any]]) -> Optional[str]:
    df = pd.DataFrame([r for r in rows if r.get("T50") is not None and r.get("T90") is not None])
    if df.empty:
        return None
    fig = px.scatter(
        df, x="T50", y="T90", color="f2",
        size=df["f2"].fillna(df["f2"].mean() if df["f2"].notna().any() else 40),
        color_continuous_scale="Viridis",
        hover_data=["run_id"]
    )
    fig.update_traces(hovertemplate="T50: %{x:.1f} min<br>T90: %{y:.1f} min<br>f₂: %{marker.color:.2f}<extra></extra>")

    fig.update_layout(
        height=450,
        width=600,
        margin=dict(l=50, r=30, t=40, b=100),
        font=dict(family="Inter, sans-serif", size=14),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(
            title="T50 (min)",
            title_font=dict(size=16),
            showgrid=True,
            gridcolor='#eee',
            tickfont=dict(size=12)
        ),
        yaxis=dict(
            title="T90 (min)",
            title_font=dict(size=16),
            showgrid=True,
            gridcolor='#eee',
            tickfont=dict(size=12)
        ),
        coloraxis_colorbar=dict(title="f₂")
    )

    fig.add_annotation(
        text="T50 vs T90 Across Runs",
        xref="paper", yref="paper",
        x=0.5, y=-0.2,
        showarrow=False,
        font=dict(size=18, color="#111"),
        xanchor="center", yanchor="top"
    )

    return plotly_div(fig, "top-t50t90")

def t50_t90_gauge_compact(rows: List[Dict[str, Any]]) -> Optional[str]:
    """Compact gauge-style horizontal bars for T50/T90."""
    df = pd.DataFrame([r for r in rows if r.get("T50") is not None and r.get("T90") is not None])
    if df.empty:
        return None

    # Take top 4 runs
    df = df.sort_values("f2", ascending=False).head(4).copy()
    df["run_id_short"] = df["run_id"].str[:8]

    # Normalize to max T90 for visual consistency
    max_t90 = df["T90"].max() * 1.1

    fig = go.Figure()

    for i, row in df.iterrows():
        # T50 bar
        fig.add_trace(go.Bar(
            y=[f"{row['run_id_short']} (T50)"],
            x=[row["T50"]],
            name="T50",
            orientation='h',
            marker=dict(color='#636efa'),
            hovertemplate="Run: %{y}<br>T50: %{x:.1f} min<extra></extra>"
        ))
        # T90 bar
        fig.add_trace(go.Bar(
            y=[f"{row['run_id_short']} (T90)"],
            x=[row["T90"]],
            name="T90",
            orientation='h',
            marker=dict(color='#ef553b'),
            hovertemplate="Run: %{y}<br>T90: %{x:.1f} min<extra></extra>"
        ))

    fig.update_layout(
        height=350,
        width=600,
        margin=dict(l=150, r=20, t=30, b=80),
        font=dict(size=11),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        barmode='group',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        xaxis=dict(
            title="Time (minutes)",
            title_font=dict(size=13),
            tickfont=dict(size=11),
            range=[0, max_t90],
            showgrid=False,
            gridcolor='#eee'
        ),
        yaxis=dict(
            title="",
            tickfont=dict(size=11),
            showgrid=False
        )
    )

    fig.add_annotation(
        text="T50 & T90 Comparison (Top 4 Runs)",
        xref="paper", yref="paper",
        x=0.5, y=-0.25,
        showarrow=False,
        font=dict(size=14, color="#111", weight='bold'),
        xanchor="center", yanchor="top"
    )

    return plotly_div(fig, "top-t50t90-gauge-compact")

def prompt_leaderboard_div(rows: List[Dict[str, Any]]) -> Optional[str]:
    df = pd.DataFrame([{"prompt": r.get("prompt_name") or "unknown", "f2": r.get("f2")} for r in rows if r.get("f2") is not None])
    if df.empty:
        return None
    agg = df.groupby("prompt", as_index=False)["f2"].mean().sort_values("f2", ascending=False).head(10)
    fig = px.bar(agg, x="prompt", y="f2", color_discrete_sequence=['#ff7f0e'])
    fig.update_traces(hovertemplate="Prompt: %{x}<br>Mean f₂: %{y:.2f}<extra></extra>")

    fig.update_layout(
        height=450,
        width=600,
        margin=dict(l=50, r=30, t=40, b=100),
        font=dict(family="Inter, sans-serif", size=14),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(
            title="",
            title_font=dict(size=16),
            tickangle=30,
            showgrid=False,
            tickfont=dict(size=12)
        ),
        yaxis=dict(
            title="Mean f₂",
            title_font=dict(size=16),
            showgrid=True,
            gridcolor='#eee',
            tickfont=dict(size=12)
        )
    )

    fig.add_annotation(
        text="Top 10 Prompts by Mean f₂",
        xref="paper", yref="paper",
        x=0.5, y=-0.25,
        showarrow=False,
        font=dict(size=18, color="#111"),
        xanchor="center", yanchor="top"
    )

    return plotly_div(fig, "top-prompt")

def qc_donut_div(cards: List[Dict[str, Any]]) -> Optional[str]:
    oks = sum(1 for c in cards if c.get("metrics", {}).get("ok") is True)
    total = sum(1 for c in cards if c.get("metrics", {}).get("ok") is not None)
    fails = total - oks
    if total == 0:
        return "<div class='section'><p>No QC data available.</p></div>"
    data = [
        {"label": "Pass", "value": oks, "color": "#2ca02c"},
        {"label": "Fail", "value": fails, "color": "#d62728"}
    ]
    d3_script = f"""
    <script>
    document.addEventListener('DOMContentLoaded', function() {{
        const data = {json.dumps(data)};
        const width = 300, height = 300, radius = Math.min(width, height) / 2;
        const svg = d3.select('#qc-donut')
            .append('svg')
            .attr('width', width)
            .attr('height', height)
            .append('g')
            .attr('transform', 'translate(' + width / 2 + ',' + height / 2 + ')');
        const color = d3.scaleOrdinal().domain(data.map(d => d.label)).range(data.map(d => d.color));
        const pie = d3.pie().value(d => d.value);
        const arc = d3.arc().innerRadius(radius * 0.6).outerRadius(radius);
        const arcs = svg.selectAll('arc')
            .data(pie(data))
            .enter()
            .append('g');
        arcs.append('path')
            .attr('d', arc)
            .attr('fill', d => color(d.data.label))
            .on('mouseover', function(event, d) {{
                d3.select(this).transition().duration(200).attr('transform', 'scale(1.05)');
            }})
            .on('mouseout', function() {{
                d3.select(this).transition().duration(200).attr('transform', 'scale(1)');
            }});
        arcs.append('text')
            .attr('transform', d => `translate(${{arc.centroid(d)}})`)
            .attr('text-anchor', 'middle')
            .attr('font-size', '14px')
            .attr('fill', 'white')
            .text(d => d.data.label);
    }});
    </script>
    """
    legend_html = """
    <div style="margin-top: 10px; display: flex; gap: 16px;">
        <div style="display: flex; align-items: center; gap: 8px;"><span style="width: 12px; height: 12px; background: #2ca02c; border-radius: 50%;"></span><span style="font-size: 12px;">Pass</span></div>
        <div style="display: flex; align-items: center; gap: 8px;"><span style="width: 12px; height: 12px; background: #d62728; border-radius: 50%;"></span><span style="font-size: 12px;">Fail</span></div>
    </div>
    """
    return f"""
    <div class='section'>
        <div class='section-title'>QC Pass/Fail Summary</div>
        <div id="qc-donut" style="margin: 0 auto 10px;"></div>
        {d3_script}
        {legend_html}
    </div>
    """

def thumbnails_wall_div(cards: List[Dict[str, Any]]) -> Optional[str]:
    runs = []
    for c in cards:
        pred = c.get("pred") or []
        if pred:
            runs.append({
                "run_id": c["run_id"],
                "profile": [{"x": p["time"], "y": p["dissolved"]} for p in pred],
                "ok": c.get("metrics", {}).get("ok", None)
            })
    if not runs:
        return "<div class='section'><p>No profiles to display.</p></div>"
    d3_script = f"""
    <script>
    document.addEventListener('DOMContentLoaded', function() {{
        const data = {json.dumps(runs)};
        const margin = {{top: 10, right: 10, bottom: 30, left: 30}},
              width = 200 - margin.left - margin.right,
              height = 120 - margin.top - margin.bottom;
        const container = d3.select('#thumbnails-wall')
            .style('display', 'grid')
            .style('grid-template-columns', 'repeat(auto-fill, minmax(240px, 1fr))')
            .style('gap', '16px')
            .style('padding', '16px');
        data.forEach(function(d, i) {{
            const card = container.append('div')
                .style('border', '1px solid #e5e7eb')
                .style('border-radius', '12px')
                .style('padding', '12px')
                .style('background', '#fff')
                .style('box-shadow', '0 2px 8px rgba(0,0,0,0.05)')
                .style('cursor', 'pointer')
                .on('click', function() {{
                    window.location.href = `../runs/${{d.run_id}}.html`;
                }});
            card.append('div')
                .style('font-weight', '600')
                .style('font-size', '12px')
                .style('margin-bottom', '8px')
                .text(d.run_id.substring(0, 12));
            const svg = card.append('svg')
                .attr('width', width + margin.left + margin.right)
                .attr('height', height + margin.top + margin.bottom);
            const g = svg.append('g')
                .attr('transform', 'translate(' + margin.left + ',' + margin.top + ')');
            const x = d3.scaleLinear()
                .domain([0, d3.max(d.profile, p => p.x) || 60])
                .range([0, width]);
            const y = d3.scaleLinear()
                .domain([0, 100])
                .range([height, 0]);
            const line = d3.line()
                .x(p => x(p.x))
                .y(p => y(p.y));
            g.append('path')
                .datum(d.profile)
                .attr('fill', 'none')
                .attr('stroke', d.ok === true ? '#2ca02c' : d.ok === false ? '#d62728' : '#1f77b4')
                .attr('stroke-width', 2)
                .attr('d', line);
            g.append('line')
                .attr('x1', 0).attr('y1', height)
                .attr('x2', width).attr('y2', height)
                .attr('stroke', '#ccc');
            g.append('line')
                .attr('x1', 0).attr('y1', 0)
                .attr('x2', 0).attr('y2', height)
                .attr('stroke', '#ccc');
        }});
    }});
    </script>
    """
    return f"""
    <div class='section'>
        <div class='section-title'>Profile Thumbnails Wall (click to view)</div>
        <div id="thumbnails-wall"></div>
        {d3_script}
    </div>
    """

def provenance_block_div(latest_run: Dict[str, Any], excel_path: str, outdir: str) -> Optional[str]:
    if not latest_run or not latest_run.get("retrieve_payload"):
        return None
    rows = latest_run["retrieve_payload"].get("results") or []
    if not rows:
        return None
    top = sorted(rows, key=lambda r: (r.get("rerank_score") or -1, r.get("faiss_score") or -1), reverse=True)[:5]
    sheets = [r.get("sheet", "N/A") for r in top]
    rer = [r.get("rerank_score", 0) for r in top]
    fai = [r.get("faiss_score", 0) for r in top]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=sheets, y=rer, name="ReRank Score",
        marker_color='#636efa',
        hovertemplate="Sheet: %{x}<br>ReRank: %{y:.3f}<extra></extra>"
    ))
    fig.add_trace(go.Bar(
        x=sheets, y=fai, name="FAISS Score",
        marker_color='#ef553b',
        hovertemplate="Sheet: %{x}<br>FAISS: %{y:.3f}<extra></extra>"
    ))
    fig.update_layout(
        barmode="group",
        height=450,
        width=600,
        margin=dict(l=50, r=30, t=40, b=100),
        font=dict(family="Inter, sans-serif", size=14),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(
            title="",
            title_font=dict(size=16),
            tickangle=-25,
            showgrid=False,
            tickfont=dict(size=12)
        ),
        yaxis=dict(
            title="Similarity Score",
            title_font=dict(size=16),
            showgrid=True,
            gridcolor='#eee',
            tickfont=dict(size=12)
        ),
        legend=dict(orientation="h", yanchor="top", y=-0.15, x=0)
    )
    fig.add_annotation(
        text="Top 5 Retrieved Sheets — Similarity Scores",
        xref="paper", yref="paper",
        x=0.5, y=-0.25,
        showarrow=False,
        font=dict(size=18, color="#111"),
        xanchor="center", yanchor="top"
    )
    bar_div = plotly_div(fig, "top-provenance")
    previews_html = ["<details class='prov'><summary>📄 View Sheet Previews</summary><div class='provgrid'>"]
    for s in sheets:
        curve = parse_sheet_timecurve(excel_path, s)
        sub = go.Figure()
        if curve:
            sub.add_trace(go.Scatter(
                x=[p["time"] for p in curve], y=[p["dissolved"] for p in curve],
                mode="lines+markers",
                line=dict(width=2.5, color='#7f7f7f'),
                hovertemplate="Time: %{x} min<br>%{y:.1f}%<extra></extra>"
            ))
            sub.update_yaxes(range=[0, 105])
        else:
            sub.add_annotation(text="⚠️ No curve found", showarrow=False, x=0.5, y=0.5, xref="paper", yref="paper", font=dict(color="#f00"))
        sub.update_layout(
            height=200,
            margin=dict(l=20, r=10, t=30, b=20),
            showlegend=False,
            font=dict(size=10),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showgrid=True, gridcolor='#eee'),
            yaxis=dict(showgrid=True, gridcolor='#eee')
        )
        sub.add_annotation(
            text=s,
            xref="paper", yref="paper",
            x=0.5, y=-0.1,
            showarrow=False,
            font=dict(size=12, color="#111"),
            xanchor="center", yanchor="top"
        )
        previews_html.append(f"<div class='provcard'>{plotly_div(sub, f'prov-{s}')}</div>")
    previews_html.append("</div></details>")
    return f"<div class='section'>{bar_div}{''.join(previews_html)}</div>"

# ---------------------- Compact Plots for Homepage ----------------------

def f2_timeline_compact(timeline: List[Dict[str, Any]]) -> Optional[str]:
    df = pd.DataFrame([{"run_id": x["run_id"], "ts": x["ts"], "f2": x.get("f2")} for x in timeline if x.get("f2") is not None])
    if df.empty:
        return None
    df = df.sort_values("ts")
    fig = px.line(df, x="ts", y="f2", markers=True, color_discrete_sequence=['#1f77b4'])
    fig.update_traces(
        hovertemplate="Time: %{x}<br>f₂: %{y:.2f}<extra></extra>",
        marker=dict(size=4),
        line=dict(width=2)
    )
    fig.update_layout(
        height=350,
        width=600,
        # margin=dict(l=60, r=10, t=20, b=90),
        font=dict(family="Inter, sans-serif", size=10),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        showlegend=False,
        xaxis=dict(title="", showgrid=False, showticklabels=False),
        yaxis=dict(title="", showgrid=False, showticklabels=True, tickfont=dict(size=10))
    )
    fig.add_annotation(
        text="f₂ Timeline",
        xref="paper", yref="paper",
        x=0.5, y=-0.3,
        showarrow=False,
        font=dict(size=12, color="#111"),
        xanchor="center", yanchor="top"
    )
    return plotly_div(fig, "top-f2-compact")

def t50_t90_contour_compact(rows: List[Dict[str, Any]]) -> Optional[str]:
    """Compact contour plot for homepage: T50 vs T90 density with f2 color."""
    df = pd.DataFrame([r for r in rows if r.get("T50") is not None and r.get("T90") is not None and r.get("f2") is not None])
    if df.empty:
        return None

    # Take top 50 runs for smoother contour
    df = df.sort_values("f2", ascending=False).head(50).copy()

    # Create contour plot
    fig = go.Figure(go.Histogram2dContour(
        x=df["T50"],
        y=df["T90"],
        colorscale="RdYlBu",
        showscale=True,
        hovertemplate="T50: %{x:.1f} min<br>T90: %{y:.1f} min<br>Density: %{z}<extra></extra>",
        colorbar=dict(title="Density")
    ))

    # Add scatter points on top for reference
    fig.add_trace(go.Scatter(
        x=df["T50"],
        y=df["T90"],
        mode="markers",
        marker=dict(
            size=6,
            color=df["f2"],
            colorscale="Viridis",
            showscale=False,
            line=dict(width=1, color='white')
        ),
        hovertemplate="T50: %{x:.1f} min<br>T90: %{y:.1f} min<br>f₂: %{marker.color:.2f}<extra></extra>",
        name="Runs"
    ))

    fig.update_layout(
        height=350,
        width=600,
        margin=dict(l=50, r=20, t=30, b=80),
        font=dict(size=11),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        showlegend=False,
        xaxis=dict(
            title="T50 (minutes)",
            title_font=dict(size=13),
            tickfont=dict(size=11),
            showgrid=True,
            gridcolor='#eee'
        ),
        yaxis=dict(
            title="T90 (minutes)",
            title_font=dict(size=13),
            tickfont=dict(size=11),
            showgrid=True,
            gridcolor='#eee'
        ),
        coloraxis_colorbar=dict(title="f₂")
    )

    fig.add_annotation(
        text="T50 vs T90 Density Contour (Top 50 Runs)",
        xref="paper", yref="paper",
        x=0.5, y=-0.18,
        showarrow=False,
        font=dict(size=14, color="#111", weight='bold'),
        xanchor="center", yanchor="top"
    )

    return plotly_div(fig, "top-t50t90-contour-compact")

def t50_t90_scatter_compact(rows: List[Dict[str, Any]]) -> Optional[str]:
    df = pd.DataFrame([r for r in rows if r.get("T50") is not None and r.get("T90") is not None])
    if df.empty:
        return None
    fig = px.scatter(
        df, x="T50", y="T90",
        size=[10]*len(df),
        color_discrete_sequence=['#1f77b4']
    )
    fig.update_traces(
        hovertemplate="T50: %{x:.1f}<br>T90: %{y:.1f}<extra></extra>",
        marker=dict(sizemode='diameter', sizeref=0.1)
    )
    fig.update_layout(
        height=220,
        width=300,
        margin=dict(l=20, r=10, t=20, b=50),
        font=dict(size=10),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        showlegend=False,
        xaxis=dict(title="", showgrid=False, showticklabels=False),
        yaxis=dict(title="", showgrid=False, showticklabels=False)
    )
    fig.add_annotation(
        text="T50 vs T90",
        xref="paper", yref="paper",
        x=0.5, y=-0.3,
        showarrow=False,
        font=dict(size=12, color="#111"),
        xanchor="center", yanchor="top"
    )
    return plotly_div(fig, "top-t50t90-compact")

def prompt_leaderboard_compact(rows: List[Dict[str, Any]]) -> Optional[str]:
    df = pd.DataFrame([{"prompt": r.get("prompt_name") or "unknown", "f2": r.get("f2")} for r in rows if r.get("f2") is not None])
    if df.empty:
        return None
    agg = df.groupby("prompt", as_index=False)["f2"].mean().sort_values("f2", ascending=False).head(5)
    fig = px.bar(agg, x="prompt", y="f2", color_discrete_sequence=['#ff7f0e'])
    fig.update_traces(
        hovertemplate="Prompt: %{x}<br>Mean f₂: %{y:.2f}<extra></extra>",
        texttemplate='%{y:.1f}',
        textposition='outside',
        textfont_size=10
    )
    fig.update_layout(
        height=350,
        width=400,
        margin=dict(l=30, r=10, t=20, b=60),
        font=dict(size=10),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        showlegend=False,
        xaxis=dict(title="", tickangle=30, showgrid=False, tickfont=dict(size=9)),
        yaxis=dict(title="", showgrid=True, gridcolor='#eee', tickfont=dict(size=9))
    )
    fig.add_annotation(
        text="Top Prompts",
        xref="paper", yref="paper",
        x=0.5, y=-0.3,
        showarrow=False,
        font=dict(size=12, color="#111"),
        xanchor="center", yanchor="top"
    )
    return plotly_div(fig, "top-prompt-compact")

# ---------------------- HTML ----------------------

def _retrieval_table_html(retrieve_payload: Dict[str, Any], outdir: str, max_rows: int = 5) -> str:
    if not retrieve_payload:
        return ""
    rows = retrieve_payload.get("results") or []
    if not rows:
        return ""
    head = "<tr><th>Sheet</th><th>Doc</th><th>FAISS</th><th>ReRank</th><th>Curve?</th></tr>"
    body = []
    for r in rows[:max_rows]:
        faiss_score = r.get("faiss_score")
        rerank_score = r.get("rerank_score")
        faiss_str = "" if faiss_score is None else f"{faiss_score:.3f}"
        rerank_str = "" if rerank_score is None else f"{rerank_score:.3f}"
        body.append(
            "<tr>"
            f"<td>{r.get('sheet','')}</td>"
            f"<td style='font-size:12px'>{r.get('doc_id','')}</td>"
            f"<td>{faiss_str}</td>"
            f"<td>{rerank_str}</td>"
            f"<td>{'✓' if r.get('has_timeseries') else '—'}</td>"
            "</tr>"
        )
    return f"""
    <details class="ret"><summary>🔍 Retrieval Evidence</summary>
      <table class="ret">{head}{''.join(body)}</table>
    </details>
    """

def get_common_head(title: str) -> str:
    return f"""
<!doctype html>
<meta charset='utf-8'/>
<title>{title}</title>
<script src='https://cdn.plot.ly/plotly-2.32.0.min.js'></script>
<script src='https://d3js.org/d3.v7.min.js'></script>
<style>
    body {{
        font-family: Inter, system-ui, Arial, sans-serif;
        padding: 24px;
        background: #fafafa;
        color: #111;
        max-width: 1400px;
        margin: 0 auto;
    }}
    .nav {{
        background: white;
        padding: 16px;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        margin-bottom: 24px;
        display: flex;
        gap: 16px;
        flex-wrap: wrap;
    }}
    .nav a {{
        padding: 8px 16px;
        background: #2563eb;
        color: white;
        text-decoration: none;
        border-radius: 6px;
        font-weight: 500;
    }}
    .nav a:hover {{
        background: #1d4ed8;
    }}
    .section {{
        background: white;
        padding: 24px;
        border-radius: 16px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        margin-bottom: 24px;
    }}
    .section-title {{
        font-size: 1.2rem;
        font-weight: 700;
        margin: 0 0 16px;
        padding-bottom: 8px;
        border-bottom: 2px solid #eee;
    }}
    .run-header {{
        background: white;
        padding: 24px;
        border-radius: 16px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        margin-bottom: 24px;
    }}
    .run-meta {{
        display: flex;
        gap: 24px;
        flex-wrap: wrap;
        margin: 16px 0;
        font-size: 14px;
        color: #555;
    }}
    .plot-container {{
        margin-bottom: 40px;
    }}
    #prompt-filter, #run-jump {{
        padding: 10px 16px;
        width: 300px;
        border: 1px solid #ddd;
        border-radius: 20px;
        font-size: 14px;
    }}
    .run-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
        gap: 16px;
    }}
    .run-card {{
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 16px;
        background: white;
        box-shadow: 0 2px 6px rgba(0,0,0,0.05);
    }}
    .run-card a {{
        display: inline-block;
        margin-top: 8px;
        padding: 6px 12px;
        background: #10b981;
        color: white;
        text-decoration: none;
        border-radius: 6px;
        font-size: 12px;
    }}
    .quick-links {{
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin: 16px 0;
    }}
    .quick-links a {{
        padding: 6px 12px;
        background: #e0e7ff;
        color: #2563eb;
        text-decoration: none;
        border-radius: 6px;
        font-size: 12px;
    }}
</style>
"""

def write_homepage(cards: List[Dict[str, Any]], timeline: List[Dict[str, Any]], t50t90_rows: List[Dict[str, Any]], prompt_rows: List[Dict[str, Any]], outdir: str) -> str:
    html = [get_common_head("💊 Pharma Dissolve Dashboard — Home")]
    html.append("""
    <div class="nav">
        <a href="diagnostics.html">📊 View Diagnostics</a>
        <a href="#runs">🧪 Browse Runs</a>
    </div>
    """)
    html.append("<h1>💊 Pharma Dissolve Dashboard</h1>")
    html.append("<p>Interactive dashboard for dissolution profile analysis. Navigate using buttons above.</p>")
    html.append("""
    <div style="margin: 24px 0; padding: 16px; background: white; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.05);">
        <h3>🔍 Jump to Run ID</h3>
        <input id="run-jump" type="text" placeholder="Enter Run ID" style="margin-right: 8px;">
        <button onclick="jumpToRun()" style="padding: 8px 16px; background: #2563eb; color: white; border: none; border-radius: 6px; cursor: pointer;">Go</button>
    </div>
    <script>
    function jumpToRun() {
        const id = document.getElementById('run-jump').value.trim();
        if (id) {
            window.location.href = `runs/${id}.html`;
        }
    }
    document.addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            jumpToRun();
        }
    });
    </script>
    """)
    compact_f2_block = f2_timeline_compact(timeline)
    compact_t50t90_block = t50_t90_contour_compact(t50t90_rows)
    compact_prompt_block = prompt_leaderboard_compact(prompt_rows)
    top_blocks_preview = [compact_f2_block, compact_t50t90_block, compact_prompt_block]
    html.append("<h2>📈 Diagnostics Preview</h2>")
    html.append("<div style='display: grid; grid-template-columns: repeat(2, 1fr); gap: 24px;'>")
    for block in top_blocks_preview:
        if block:
            html.append(f"<div class='section'>{block}</div>")
    html.append("</div>")
    html.append("<h2 id='runs'>🧪 All Runs</h2>")
    html.append("<div class='run-grid'>")
    for c in cards:
        rid = c['run_id']
        ts = c.get('ts', 'Unknown')
        f2 = c.get('metrics', {}).get('f2', 'N/A')
        f2_display = f"{f2:.2f}" if isinstance(f2, float) else f2
        html.append(f"""
        <div class='run-card'>
            <div style='font-weight: 600; font-size: 14px;'>{rid[:12]}</div>
            <div style='font-size: 12px; color: #555; margin: 4px 0;'>{ts}</div>
            <div style='font-size: 12px; color: #555;'>f₂: {f2_display}</div>
            <a href='runs/{rid}.html'>View Details</a>
        </div>
        """)
    html.append("</div>")
    ensure_dir(outdir)
    out_path = os.path.join(outdir, "index.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(html))
    return out_path

def write_diagnostics_page(cards: List[Dict[str, Any]], top_blocks: List[str], outdir: str) -> str:
    html = [get_common_head("💊 Pharma Dissolve Dashboard — Diagnostics")]
    html.append("""
    <div class="nav">
        <a href="index.html">🏠 Home</a>
        <a href="#runs">🧪 Browse Runs</a>
    </div>
    """)
    html.append("<h1>📊 Global Diagnostics</h1>")
    html.append("<p>Comprehensive metrics across all runs.</p>")
    html.append("<div style='display: grid; grid-template-columns: repeat(2, 1fr); gap: 24px; margin-bottom: 32px;'>")
    for block in top_blocks:
        if block:
            html.append(f"<div class='section'>{block}</div>")
    html.append("</div>")
    html.append("<h2 id='runs'>🧪 Quick Run Access</h2>")
    html.append("<div class='quick-links'>")
    for c in cards[:30]:
        rid = c['run_id']
        html.append(f"<a href='runs/{rid}.html'>{rid[:8]}</a>")
    html.append("</div>")
    ensure_dir(outdir)
    out_path = os.path.join(outdir, "diagnostics.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(html))
    return out_path

def write_run_detail_page(card: Dict[str, Any], outdir: str) -> str:
    rid = card['run_id']
    html = [get_common_head(f"💊 Run {rid} — Details")]
    html.append("""
    <div class="nav">
        <a href="../index.html">🏠 Home</a>
        <a href="../diagnostics.html">📊 Diagnostics</a>
    </div>
    """)
    html.append(f"<div class='run-header'>")
    html.append(f"<h1>🧪 Run: {rid}</h1>")
    html.append(f"<div class='run-meta'>")
    if card.get("ts"):
        html.append(f"<div>🕒 {card['ts']}</div>")
    if card.get("prompt_name"):
        pn = card["prompt_name"]
        if card.get("prompt_path"):
            pr = _safe_rel(outdir, card["prompt_path"])
            if pr:
                html.append(f"<div>📄 Prompt: <a href='{pr}' target='_blank'>{pn}</a></div>")
            else:
                html.append(f"<div>📄 Prompt: {pn}</div>")
        else:
            html.append(f"<div>📄 Prompt: {pn}</div>")
    if card.get("metrics", {}).get("f2") is not None:
        html.append(f"<div>🎯 f₂: {card['metrics']['f2']:.2f}</div>")
    if card.get("metrics", {}).get("mae") is not None:
        html.append(f"<div>📉 MAE: {card['metrics']['mae']:.2f}%</div>")
    html.append("</div>")
    links = []
    if card.get("report"):
        report_filename = os.path.basename(card["report"])
        links.append(f"<a href='{report_filename}' target='_blank'>📝 Report</a>")
    if card.get("profile"):
        profile_filename = os.path.basename(card["profile"])
        links.append(f"<a href='{profile_filename}' target='_blank'>📊 Profile JSON</a>")
    if links:
        html.append("<div>" + " | ".join(links) + "</div>")
    html.append("</div>")
    html.append("<h2>📈 Analysis Plots</h2>")
    html.append("<div style='display: grid; grid-template-columns: repeat(2, 1fr); gap: 24px; margin-bottom: 32px;'>")
    plot_keys = ["overlay_div", "error_div", "hist_div", "rate_div", "qc_div"]
    for key in plot_keys:
        div = card.get(key)
        if div:
            html.append(f"<div class='section' style='height: 450px;'>{div}</div>")
    html.append("</div>")
    if card.get("retrieve_payload"):
        html.append("<div class='section'>")
        html.append("<div class='section-title'>🔍 Retrieval Evidence</div>")
        html.append(_retrieval_table_html(card["retrieve_payload"], outdir))
        html.append("</div>")
    if card.get("report"):
        html.append("<div class='section'>")
        html.append("<div class='section-title'>📄 Report Content</div>")
        try:
            with open(card["report"], 'r', encoding='utf-8') as f:
                report_text = f.read()
            html_content = markdown.markdown(report_text, extensions=[
                'fenced_code',
                'tables',
                'nl2br',
                'extra'
            ])
            html.append(f"<div style='font-family: Inter, sans-serif; line-height: 1.6;'>{html_content}</div>")
        except Exception as e:
            html.append(f"<p style='color: #d32f2f;'>Error reading report: {str(e)}</p>")
        html.append("</div>")
    run_dir = os.path.join(outdir, "runs")
    ensure_dir(run_dir)
    out_path = os.path.join(run_dir, f"{rid}.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(html))
    return out_path

# ---------------------- Run loading ----------------------

def load_runs_all_stages(log_path: str) -> Dict[str, Dict[str, Any]]:
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
            rid = rec.get("run_id")
            if not rid:
                continue
            stage = rec.get("stage")
            payload = rec.get("payload", {})
            runs.setdefault(rid, {})
            if stage == "final":
                runs[rid]["final"] = payload
                runs[rid]["final_ts"] = rec.get("ts")
            elif stage == "prompt":
                runs[rid]["prompt"] = payload
            elif stage == "retrieve":
                runs[rid]["retrieve"] = payload
    return {rid: st for rid, st in runs.items() if "final" in st and "report_path" in st["final"]}

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

def find_prompt_file(run_id: str, run_dir: str, runs_meta: Dict[str, Any]) -> Tuple[str, Optional[str]]:
    prompt_name = (
        runs_meta.get("final", {}).get("prompt_filename") or
        runs_meta.get("prompt", {}).get("prompt_filename")
    )
    if prompt_name:
        candidate = os.path.join(run_dir, prompt_name)
        if os.path.exists(candidate):
            return prompt_name, candidate
        return prompt_name, None
    txts = [f for f in os.listdir(run_dir) if f.lower().endswith(".txt")] if os.path.isdir(run_dir) else []
    if txts:
        return txts[0], os.path.join(run_dir, txts[0])
    return "unknown.txt", None

def copy_artifacts_into_gallery(run_id: str, report: Optional[str], profile: Optional[str], runs_dir: str) -> Tuple[Optional[str], Optional[str]]:
    ensure_dir(runs_dir)
    new_report = None
    new_profile = None
    if report and os.path.exists(report):
        report_filename = os.path.basename(report)
        new_report = os.path.join(runs_dir, report_filename)
        shutil.copyfile(report, new_report)
    if profile and os.path.exists(profile):
        profile_filename = os.path.basename(profile)
        new_profile = os.path.join(runs_dir, profile_filename)
        shutil.copyfile(profile, new_profile)
    return new_report, new_profile

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

    runs_all = load_runs_all_stages(args.log)

    timeline: List[Dict[str, Any]] = []
    t50t90_rows: List[Dict[str, Any]] = []
    prompt_rows: List[Dict[str, Any]] = []
    cards: List[Dict[str, Any]] = []
    latest_rid = None
    latest_ts = datetime.datetime.min

    for rid, stages in runs_all.items():
        final = stages["final"]
        report = final.get("report_path")
        profile = final.get("profile_json")
        metrics = final.get("metrics") or {}
        sources = final.get("sources") or []
        ts_iso = stages.get("final_ts")
        ts = _parse_ts(ts_iso)
        ts_human = ts.isoformat(sep=" ", timespec="seconds") if ts else None

        if ts and ts > latest_ts:
            latest_ts = ts
            latest_rid = rid

        run_dir = os.path.dirname(report) if report else os.path.join("artifacts", f"run_{rid}")
        prompt_name, prompt_path = find_prompt_file(rid, run_dir, stages)

        pred = load_profile(profile) if profile and os.path.exists(profile) else None
        exp = None
        for s in sources:
            sheet = s.get("sheet")
            if sheet:
                exp = parse_sheet_timecurve(args.excel, sheet)
                if exp:
                    break

        report_copy, profile_copy = copy_artifacts_into_gallery(rid, report, profile, runs_dir)

        ov = overlay_div(rid, pred or [], exp)
        err = error_div(rid, pred or [], exp)
        hist = residual_hist_div(rid, pred or [], exp)
        rate = rate_div(rid, pred or [])
        qc = qc_bars_div(rid, metrics)

        f2_val = None
        mae_val = metrics.get("mae")
        if pred and exp:
            grid, _err = diff_profile(pred, exp)
            ref = interp_profile(exp, list(grid))
            tst = interp_profile(pred, list(grid))
            f2_val = f2_on_grid(ref, tst)

        timeline.append({"run_id": rid, "ts": ts or datetime.datetime.min, "f2": f2_val, "mae": mae_val})
        t50t90_rows.append({"run_id": rid, "T50": metrics.get("T50"), "T90": metrics.get("T90"), "f2": f2_val})
        prompt_rows.append({"prompt_name": prompt_name, "f2": f2_val})

        cards.append({
            "run_id": rid,
            "ts": ts_human,
            "overlay_div": ov,
            "error_div": err,
            "hist_div": hist,
            "rate_div": rate,
            "qc_div": qc,
            "report": report_copy,
            "profile": profile_copy,
            "retrieve_payload": stages.get("retrieve"),
            "prompt_name": prompt_name,
            "prompt_path": prompt_path,
            "metrics": metrics,
            "pred": pred,
        })

    f2_block_expanded = f2_timeline_expanded(timeline)
    t50t90_block_expanded = t50_t90_scatter_div(t50t90_rows)
    prompt_block_expanded = prompt_leaderboard_div(prompt_rows)
    qc_donut_block = qc_donut_div(cards)
    thumbs_block = thumbnails_wall_div(cards)

    latest_run_card = next((c for c in cards if c["run_id"] == latest_rid), None)
    provenance_block = provenance_block_div(latest_run_card, args.excel, gallery_dir) if latest_run_card else None

    top_blocks_diagnostics = [
        thumbs_block,
        f2_block_expanded,
        t50t90_block_expanded,
        prompt_block_expanded,
        qc_donut_block,
        provenance_block
    ]

    homepage_path = write_homepage(cards, timeline, t50t90_rows, prompt_rows, gallery_dir)
    diagnostics_path = write_diagnostics_page(cards, top_blocks_diagnostics, gallery_dir)

    for card in cards:
        write_run_detail_page(card, gallery_dir)

    print("✅ Multi-page dashboard generated!")
    print(f"   🏠 Homepage: {homepage_path}")
    print(f"   📊 Diagnostics: {diagnostics_path}")
    print(f"   🧪 Run Pages: {len(cards)} pages in {runs_dir}/")
    print("\n🚀 Serve locally with:")
    print(f"   python -m http.server -d {gallery_dir} 8000")

if __name__ == "__main__":
    main()