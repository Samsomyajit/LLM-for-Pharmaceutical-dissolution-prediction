# 🧪 Pharma Dissolution Dashboard — Diagnosict AI

> **Interactive multi-page dashboard for visualizing pharmaceutical dissolution profiles, QC metrics, and LLM-generated reports.**

Built with Python, Plotly, D3.js, and Markdown — **no backend required**. Perfect for QC teams, formulation scientists, and AI/ML reviewers.

---

## 🚀 Features

✅ **Multi-page navigation** — Homepage, Diagnostics, Individual Run Details  
✅ **Rendered Markdown reports** — No more raw text — full formatting with headers, lists, code blocks  
✅ **Zoom/Pan enabled** on all plots — Explore data interactively  
✅ **Compact + Expanded views** — Homepage cards are small; Diagnostics page is full-width  
✅ **QC Donut with Legend** — Clear Pass/Fail visualization  
✅ **Clickable Run Thumbnails** — Jump to any run with one click  
✅ **Search & Filter** — Jump to Run ID or filter by prompt  
✅ **Static & Portable** — Just run `python -m http.server` — no database or Flask needed

---

## 📦 Prerequisites

You need:

- Python 3.9+
- Git (optional, for version control)
- Terminal / Command Prompt

---

## ⚙️ Installation & Setup

### 1. Clone or Download Project

If using Git:

```bash
git clone https://github.com/yourusername/LLM-for-Pharmaceutical-dissolution-prediction.git
cd LLM-for-Pharmaceutical-dissolution-prediction
```

Or just download the folder and `cd` into it.

---

### 2. Set Up Virtual Environment (Recommended)

```bash
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
```

---

### 3. Install Dependencies

```bash
pip install plotly pandas numpy openpyxl d3py markdown
```

> ✅ `plotly` — for interactive charts  
> ✅ `pandas` — for data handling  
> ✅ `openpyxl` — for reading Excel files  
> ✅ `markdown` — for rendering `.md` reports  
> ✅ `numpy` — for math operations

---

## ▶️ Generate Dashboard

Run the dashboard generator:

```bash
python dashboard_gallary.py \
    --excel RAG_database.xlsx \
    --log mcp_runs.jsonl \
    --out dashboards_basic
```

### Arguments:

| Flag | Description | Required |
|------|-------------|----------|
| `--excel` | Path to Excel file with experimental curves | ✅ Yes |
| `--log` | Path to `mcp_runs.jsonl` log file | ✅ Yes |
| `--out` | Output folder for generated dashboard (default: `dashboards_basic`) | ❌ No |

---

## 🌐 Serve & View Dashboard

After generation, serve the dashboard locally:

```bash
cd dashboards_basic
python -m http.server 8000
```

Then open in your browser:

👉 [http://localhost:8000](http://localhost:8000)

---

## 🖥️ Dashboard Structure

```
dashboards_basic/
├── index.html                  ← Homepage: Run list + compact diagnostics
├── diagnostics.html            ← Full diagnostics: f2 timeline, scatter, leaderboard, QC, thumbnails
└── runs/
    ├── <run_id>.html           ← Individual run detail page (5 plots + rendered report)
    └── <run_id>_report.md      ← Copied report file (rendered in HTML)
```

---

## 📊 What You’ll See

### 🏠 Homepage (`index.html`)
- Navigation bar
- Jump-to-run search box
- Compact previews: f₂ timeline, T50/T90 scatter, prompt leaderboard
- Grid of all runs with f₂ score and “View Details” button

### 📈 Diagnostics Page (`diagnostics.html`)
- Full-width, detailed plots:
  - f₂ Timeline (expanded)
  - T50 vs T90 Scatter
  - Prompt Leaderboard
  - QC Pass/Fail Donut (with legend)
  - Profile Thumbnails Wall (click to navigate)
  - Provenance Bar + Sheet Previews

### 🧪 Run Detail Page (`runs/<run_id>.html`)
- Run metadata: ID, timestamp, prompt, f₂, MAE
- 5 full-width plots:
  - Dissolution Profile (Pred vs Exp)
  - Prediction Error (Δ%)
  - Residual Histogram
  - Release Rate (d%/dt)
  - QC Summary Metrics
- Retrieval evidence table
- **Fully rendered Markdown report** (headers, bold, code, tables)
- Links to download raw `.md` and `.json`

---
