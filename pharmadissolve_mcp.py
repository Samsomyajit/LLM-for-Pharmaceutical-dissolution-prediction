# pharmadissolve_mcp.py
# -*- coding: utf-8 -*-

import os, re, io, json, math, uuid, time, hashlib, warnings, datetime
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# ============================== Logging ==============================

class JSONLLogger:
    def __init__(self, path="mcp_runs.jsonl", project="PharmaDissolve-MCP"):
        self.path = path
        self.project = project
        self.run_id = str(uuid.uuid4())
        self.log("start", {"project": project})

    @staticmethod
    def _ts():
        return datetime.datetime.now().isoformat()

    def log(self, stage: str, payload: Dict[str, Any]):
        rec = {"run_id": self.run_id, "stage": stage, "ts": self._ts(), "payload": payload}
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def initialize_deepseek_client(api_key: str):
    provider = os.getenv("LLM_PROVIDER", "openrouter")
    if provider == "groq":
        from groq import Groq
        provider_client = Groq(api_key=os.getenv("GROQ_API_KEY", api_key))
        base_create = provider_client.chat.completions.create
        def create_fn(**kwargs): return base_create(**kwargs)
    elif provider == "openai":
        from openai import OpenAI
        provider_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", api_key))
        base_create = provider_client.chat.completions.create
        def create_fn(**kwargs): return base_create(**kwargs)
    elif provider == "openrouter":
        from openai import OpenAI
        provider_client = OpenAI(api_key=os.getenv("OPENROUTER_API_KEY", api_key),
                                 base_url="https://openrouter.ai/api/v1")
        base_create = provider_client.chat.completions.create
        default_headers = {
            "HTTP-Referer": os.getenv("OPENROUTER_SITE_URL", "http://localhost"),
            "X-Title": os.getenv("OPENROUTER_APP_TITLE", "PharmaDissolve-MCP"),
        }
        def create_fn(**kwargs):
            headers = kwargs.get("extra_headers") or {}
            kwargs["extra_headers"] = {**default_headers, **headers}
            return base_create(**kwargs)
    elif provider == "ollama":
        from openai import OpenAI
        provider_client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
        base_create = provider_client.chat.completions.create
        def create_fn(**kwargs): return base_create(**kwargs)
    else:
        raise ValueError(f"Unsupported LLM_PROVIDER: {provider}")

    class _Completions:
        def create(self, **kwargs):
            if "model" not in kwargs:
                kwargs["model"] = os.getenv("LLM_MODEL", "deepseek/deepseek-chat-v3.1:free")
            return create_fn(**kwargs)
    class _Chat:
        def __init__(self): self.completions = _Completions()
    class ShimClient:
        def __init__(self): self.chat = _Chat()
    return ShimClient()

# ============================== LLM Client Shim ==============================

def initialize_deepseek_client(api_key: str):
    """
    Returns an object with .chat.completions.create(...)
    Provider via env LLM_PROVIDER in {groq, openai, openrouter, ollama}
    """
    provider = os.getenv("LLM_PROVIDER", "openrouter")

    if provider == "groq":
        from groq import Groq
        provider_client = Groq(api_key=os.getenv("GROQ_API_KEY", api_key))
        base_create = provider_client.chat.completions.create

        def create_fn(**kwargs):
            return base_create(**kwargs)

    elif provider == "openai":
        from openai import OpenAI
        provider_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", api_key))
        base_create = provider_client.chat.completions.create

        def create_fn(**kwargs):
            return base_create(**kwargs)

    elif provider == "openrouter":
        from openai import OpenAI
        provider_client = OpenAI(
            api_key=os.getenv("OPENROUTER_API_KEY", api_key),
            base_url="https://openrouter.ai/api/v1",
        )
        base_create = provider_client.chat.completions.create
        default_headers = {
            "HTTP-Referer": os.getenv("OPENROUTER_SITE_URL", "http://localhost"),
            "X-Title": os.getenv("OPENROUTER_APP_TITLE", "PharmaDissolve-MCP"),
        }

        def create_fn(**kwargs):
            headers = kwargs.get("extra_headers") or {}
            merged = {**default_headers, **headers}
            kwargs["extra_headers"] = merged
            return base_create(**kwargs)

    elif provider == "ollama":
        from openai import OpenAI
        provider_client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
        base_create = provider_client.chat.completions.create

        def create_fn(**kwargs):
            return base_create(**kwargs)

    else:
        raise ValueError(f"Unsupported LLM_PROVIDER: {provider}")

    class _Completions:
        def create(self, **kwargs):
            if "model" not in kwargs:
                kwargs["model"] = os.getenv("LLM_MODEL", "deepseek/deepseek-chat-v3.1:free")
            return create_fn(**kwargs)

    class _Chat:
        def __init__(self): self.completions = _Completions()

    class ShimClient:
        def __init__(self): self.chat = _Chat()

    return ShimClient()

# ============================== Data Processor ==============================

class PharmaDataProcessor:
    @staticmethod
    def _clean_value(raw_value, field_type: str):
        if pd.isna(raw_value) or str(raw_value).strip().lower() in ["", "na", "n/a", "nan"]:
            if field_type == "drug_name":
                return None
            return "Unknown" if field_type in ["model_type", "shape"] else 0.0
        if field_type == "drug_name":
            cleaned = str(raw_value).strip().title()
            return cleaned if len(cleaned) > 2 else None
        if field_type in ["particle_size", "solubility", "diffusion_coeff", "true_density", "surface_area"]:
            try:
                s = str(raw_value).replace("µ", "μ")
                return float(re.sub(r"[^\d.+\-eE]", "", s))
            except:
                return 0.0
        return str(raw_value).strip()

    def parse_excel_sheet(self, file_path: str, sheet_name: str) -> tuple:
        df = pd.read_excel(file_path, sheet_name=sheet_name, header=None, dtype=str).fillna("")
        n_rows, n_cols = df.shape

        def cell(r, c):
            if 0 <= r < n_rows and 0 <= c < n_cols:
                return str(df.iat[r, c]).strip()
            return ""

        def find_header(patterns):
            pat = re.compile("|".join(patterns), re.IGNORECASE)
            for r in range(n_rows):
                for c in range(n_cols):
                    if pat.fullmatch(cell(r, c)):
                        right = cell(r, c+1)
                        down  = cell(r+1, c)
                        return right if right != "" else (down if down != "" else "")
            return ""

        # ---- fixed metadata extraction (if present) ----
        meta = {}
        meta["drug_name"]       = self._clean_value(find_header([r"^\s*(drug|product|api)\s*name\s*$", r"^\s*drug\s*$"]), "drug_name")
        meta["particle_size"]   = self._clean_value(find_header([r"particle\s*size"]), "particle_size")
        meta["solubility"]      = self._clean_value(find_header([r"solubility"]), "solubility")
        meta["model_type"]      = self._clean_value(find_header([r"model\s*type", r"model"]), "model_type")
        meta["diffusion_coeff"] = self._clean_value(find_header([r"diffusion\s*coefficient"]), "diffusion_coeff")
        meta["true_density"]    = self._clean_value(find_header([r"(true\s*)?density"]), "true_density")
        meta["shape"]           = self._clean_value(find_header([r"^shape$"]), "shape")
        meta["surface_area"]    = self._clean_value(find_header([r"surface\s*area"]), "surface_area")
        meta["reference"]       = find_header([r"reference"])

        # ---------- Robust time/percent extraction ----------
        def _to_float(s):
            s = str(s)
            s = s.replace("µ", "μ").replace("%", "")
            m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
            return float(m.group(0)) if m else None

        data_num = [[_to_float(df.iat[r, c]) for c in range(n_cols)] for r in range(n_rows)]

        # candidate numeric columns (≥4 numeric cells)
        cand_cols = []
        for c in range(n_cols):
            col = [data_num[r][c] for r in range(n_rows)]
            vals = [v for v in col if v is not None]
            if len(vals) >= 4:
                cand_cols.append(c)

        def _mono_increasing(xs):
            xs = [x for x in xs if x is not None]
            return len(xs) >= 4 and all(b >= a for a, b in zip(xs, xs[1:]))

        best = None
        # header-led detection first
        header_like = re.compile(r"(time|t\s*\(.*min.*\)|min)", re.IGNORECASE)
        time_cols = [c for c in range(n_cols) if any(header_like.search(cell(r, c)) for r in range(min(n_rows, 5)))]
        if time_cols:
            for tc in time_cols:
                for vc in range(n_cols):
                    if vc == tc: continue
                    times, vals = [], []
                    for r in range(n_rows):
                        t = data_num[r][tc]; v = data_num[r][vc]
                        if t is None or v is None: continue
                        times.append(t); vals.append(v)
                    if len(times) >= 4 and _mono_increasing(times):
                        frac_in_range = np.mean([(0.0 <= y <= 110.0) for y in vals]) if vals else 0
                        if frac_in_range >= 0.8:
                            score = len(times) + 0.5 * in_range
                            best = {"times": times, "vals": vals, "score": score}
                            break

        # fallback: brute-force any numeric pair (monotone time, % in [0,110], non-negative corr)
        if best is None:
            for tc in cand_cols:
                times = [data_num[r][tc] for r in range(n_rows)]
                if not _mono_increasing(times): continue
                for vc in cand_cols:
                    if vc == tc: continue
                    vals = [data_num[r][vc] for r in range(n_rows)]
                    pairs = [(t, v) for t, v in zip(times, vals) if t is not None and v is not None]
                    if len(pairs) < 4: continue
                    T = [p[0] for p in pairs]; Y = [p[1] for p in pairs]
                    in_range = np.mean([(0.0 <= y <= 110.0) for y in Y])
                    if in_range < 0.8: continue
                    try:
                        corr = np.corrcoef(T, Y)[0,1]
                    except Exception:
                        corr = 0.0
                    if corr < -0.1: continue
                    score = len(T) + corr
                    if best is None or score > best["score"]:
                        best = {"times": T, "vals": Y, "score": score}

        time_series = []
        if best:
            arr = sorted(zip(best["times"], best["vals"]), key=lambda x: x[0])
            last = -1e9
            for t, y in arr:
                if t is None or y is None: continue
                if t < last: continue
                last = t
                time_series.append({"time": float(t), "dissolved": float(np.clip(y, 0.0, 100.0))})

        return meta, time_series

    @staticmethod
    def generate_semantic_text(metadata: Dict, time_series: List) -> str:
        meta_text = (
            f"Drug: {metadata.get('drug_name', 'Unknown')}\n"
            f"• Particle Size: {metadata.get('particle_size', 0.0)} μm\n"
            f"• Solubility: {metadata.get('solubility', 0.0)} mg/mL\n"
            f"• Model: {metadata.get('model_type', 'Unknown')}\n"
            f"• Shape: {metadata.get('shape', 'Unknown')}"
        )
        time_text = "Dissolution Profile:\n" + "\n".join(
            [f"{t['time']} min: {t['dissolved']}%" for t in time_series[:5]]
        )
        if len(time_series) > 5:
            time_text += f"\n(...{len(time_series) - 5} more points)"
        return f"{meta_text}\n\n{time_text}"

    def process_pharma_data(self, file_path: str) -> List[Document]:
        excel = pd.ExcelFile(file_path)
        docs = []
        for sheet in excel.sheet_names:
            meta, times = self.parse_excel_sheet(file_path, sheet)
            if not meta.get('drug_name'):
                meta['drug_name'] = f"Unknown-{sheet}"
            doc = Document(
                page_content=self.generate_semantic_text(meta, times),
                metadata={
                    "model_type": meta.get("model_type", "Unknown"),
                    "particle_size_μm": meta.get("particle_size", 0.0),
                    "shape": meta.get("shape", "Unknown"),
                    "sheet": sheet,
                    "drug_name": meta.get("drug_name"),
                    "solubility": meta.get("solubility", 0.0),
                    "diffusion_coeff": meta.get("diffusion_coeff", 0.0),
                    "true_density": meta.get("true_density", 0.0),
                    "surface_area": meta.get("surface_area", 0.0),
                    # carry experimental curve for QC
                    "time_series": times,
                }
            )
            docs.append(doc)
        return docs

# ============================== Retriever ==============================

class AdaptiveWeightedRetriever:
    def __init__(self):
        self.default_weights = {
            'particle_size': 0.9,
            'solubility': 0.3,
            'surfactant': 0.3,
            'polymorph': 0.3,
            'disintegration': 0.3,
            'model_type': 0.3,
            'diffusion_coeff': 0.3,
            'shape': 0.15,
        }
        self.metadata_boost = 0.3
        self.query_weights = self.default_weights.copy()

    def _normalize_query(self, query: str) -> Dict:
        param_patterns = {
            'particle_size': r"(\d+\.?\d*)\s*μm",
            'solubility': r"Solubility:\s*(\d+\.?\d*)",
            'diffusion_coeff': r"Diffusion\s*Coefficient:\s*([\d.e-]+)",
            'true_density': r"Density:\s*(\d+\.?\d*)",
            'surface_area': r"Surface\s*Area:\s*(\d+\.?\d*)",
            'shape': r"Shape:\s*\"([^\"]+)\"",
            'model_type': r"Model:\s*'([^']+)'",
        }
        extracted = {}
        for param, pattern in param_patterns.items():
            m = re.search(pattern, query, re.IGNORECASE)
            if m:
                try:
                    extracted[param] = float(m.group(1)) if param != 'shape' else m.group(1).lower()
                except:
                    extracted[param] = m.group(1).lower() if param == 'shape' else 0.0
        return extracted

    @staticmethod
    def _gaussian_similarity(a: float, b: float, sigma: float) -> float:
        return float(np.exp(-(a - b) ** 2 / (2 * sigma ** 2)))

    def calculate_score(self, doc_meta: Dict, query_params: Dict) -> float:
        score = 0.0
        num_params = ['particle_size', 'solubility', 'diffusion_coeff', 'true_density', 'surface_area']
        for param in num_params:
            if param in query_params and param in doc_meta:
                q_val = query_params[param]
                doc_val = doc_meta.get(f"{param}_μm", 0.0) if param == 'particle_size' else doc_meta.get(param, 0.0)
                decay = {'particle_size':0.3,'solubility':0.25,'diffusion_coeff':0.4,'true_density':0.5,'surface_area':0.3}.get(param,0.25)
                sigma = max(abs(q_val * decay), 1e-3)
                w = self.query_weights.get(param, 0.0)
                score += w * self._gaussian_similarity(doc_val, q_val, sigma)

        if 'model_type' in query_params and 'model_type' in doc_meta:
            if str(doc_meta['model_type']).lower() == str(query_params['model_type']).lower():
                score += self.query_weights.get('model_type',0.0) * 1.2

        shape_synonyms = {
            'spherical': ['round', 'globular'],
            'irregular': ['angular', 'rough'],
            'crystalline': ['needle-like', 'prismatic'],
            'cuboid': ['orthorhombic', 'blocky'],
        }
        if 'shape' in query_params and 'shape' in doc_meta:
            q_shape = str(query_params['shape']).lower()
            doc_shape = str(doc_meta['shape']).lower()
            if q_shape == doc_shape:
                score += self.query_weights.get('shape', 0.0)
            else:
                for key, syns in shape_synonyms.items():
                    if q_shape in syns and doc_shape == key:
                        score += self.query_weights.get('shape', 0.0) * 0.8
                        break

        # prefer docs that actually contain experimental curves
        if doc_meta.get("time_series"):
            score += 0.25

        return float(score)

    def adaptive_retrieval(self, query: str, items: List[Dict], k: int = 3) -> List[Tuple[float, Dict]]:
        query_params = self._normalize_query(query)
        total_params = len(query_params)
        if total_params > 0:
            scale = 1 / total_params
            self.query_weights = {kk: vv * scale for kk, vv in self.default_weights.items()}

        scored: List[Tuple[float, Dict]] = []
        for it in items:
            s = self.calculate_score(it['metadata'], query_params)
            scored.append((s, it))
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:k]

# ============================== Vector Store Wrapper ==============================

class PharmaVectorStore:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        self.store = None
        self.processor = PharmaDataProcessor()

    def build_knowledge_base(self, file_path: str, persist_dir: str = "faiss_index"):
        docs = self.processor.process_pharma_data(file_path)
        if not docs: raise ValueError(f"No documents parsed from {file_path}.")
        chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=75).split_documents(docs)
        if not chunks: raise ValueError("Got 0 chunks from documents; inspect parsing/generation.")
        self.store = FAISS.from_documents(chunks, self.embeddings)
        self.store.save_local(persist_dir)
        return chunks

    def load(self, persist_dir: str = "faiss_index"):
        index_path = os.path.join(persist_dir, "index.faiss")
        if not os.path.exists(index_path): raise FileNotFoundError(index_path)
        self.store = FAISS.load_local(persist_dir, self.embeddings, allow_dangerous_deserialization=True)

    def similarity(self, query: str, k: int = 12) -> List[Tuple[Document, Optional[float]]]:
        if self.store is None: raise ValueError("Vector store not loaded")
        try:
            pairs = self.store.similarity_search_with_score(query, k=k)
            return pairs
        except Exception:
            docs = self.store.similarity_search(query, k=k)
            return [(d, None) for d in docs]

# ============================== QC / Metrics ==============================

def _interp_profile(profile: List[Dict[str, float]], grid: List[float]) -> List[Dict[str, float]]:
    if not profile: return []
    xs = np.array([p["time"] for p in profile], float)
    ys = np.array([p["dissolved"] for p in profile], float)
    ys = np.clip(np.maximum.accumulate(ys), 0.0, 100.0)
    gy = np.interp(grid, xs, ys, left=ys[0], right=ys[-1])
    return [{"time": float(t), "dissolved": float(v)} for t, v in zip(grid, gy)]

def f2_on_grid(ref: List[Dict[str,float]], tst: List[Dict[str,float]]) -> Optional[float]:
    n = len(ref)
    if n < 3 or n != len(tst): return None
    diffsq = sum((ref[i]["dissolved"] - tst[i]["dissolved"])**2 for i in range(n)) / n
    try:
        return 50 * math.log10((1 + diffsq) ** -0.5 * 100)
    except Exception:
        return None

def qc_metrics(profile: List[Dict[str,float]], exp_profile: Optional[List[Dict[str,float]]] = None) -> Dict[str, Any]:
    if not profile:
        return {"ok": False, "reason": "empty_profile"}
    t = np.array([p["time"] for p in profile], float)
    y = np.array([p["dissolved"] for p in profile], float)
    within = float(np.mean((y >= -1e-6) & (y <= 100 + 1e-6)))
    mono = float(np.mean(np.diff(y) >= -1e-6)) if len(y) > 1 else 1.0
    s2 = float(np.mean(np.abs(np.diff(y, n=2)))) if len(y) >= 3 else 0.0

    f2_val = None; mae = None; rmse = None; t50 = None; t90 = None
    grid = None
    if exp_profile:
        grid = sorted(set([p["time"] for p in profile] + [p["time"] for p in exp_profile]))
        if len(grid) < 5:
            grid = [0, 5, 10, 15, 30, 45, 60]
        r = _interp_profile(exp_profile, grid)
        z = _interp_profile(profile, grid)
        f2_val = f2_on_grid(r, z)
        diffs = np.array([z[i]["dissolved"] - r[i]["dissolved"] for i in range(len(grid))], float)
        mae = float(np.mean(np.abs(diffs)))
        rmse = float(np.sqrt(np.mean(diffs**2)))

    def _tx(y_arr, t_arr, x):
        if y_arr[0] >= x: return float(t_arr[0])
        if y_arr[-1] < x: return None
        return float(np.interp(x, y_arr, t_arr))
    t50 = _tx(y, t, 50); t90 = _tx(y, t, 90)

    return {
        "ok": bool(within == 1.0 and mono >= 0.9),
        "within_0_100": within,
        "monotonicity_fraction": mono,
        "smoothness_abs_ddiff": s2,
        "f2": f2_val,
        "mae": mae,
        "rmse": rmse,
        "T50": t50,
        "T90": t90,
        "grid": grid,
    }

# ============================== Agents ==============================

class RetrievalAgent:
    def __init__(self, vs: PharmaVectorStore, logger: JSONLLogger):
        self.vs = vs
        self.logger = logger
        self.reranker = AdaptiveWeightedRetriever()

    def retrieve(self, query: str, k: int = 6) -> List[Document]:
        t0 = time.time()
        # grab a larger pool, then re-rank adaptively
        pairs = self.vs.similarity(query, k=max(12, k*2))
        items = [{"doc": doc, "metadata": doc.metadata, "content": doc.page_content, "faiss_score": float(score) if score is not None else None}
                 for (doc, score) in pairs]
        ranked = self.reranker.adaptive_retrieval(query, items, k=k)  # List[(rerank_score, item)]
        top_docs = [rec["doc"] for (sc, rec) in ranked]

        results = []
        for rerank_score, rec in ranked:
            md = rec["metadata"]
            results.append({
                "sheet": md.get("sheet"),
                "doc_id": f"{md.get('sheet','#')}#{md.get('drug_name','')}",
                "faiss_score": rec.get("faiss_score"),
                "rerank_score": float(rerank_score),
                "has_timeseries": bool(md.get("time_series")),
            })
        self.logger.log("retrieve", {
            "query": query, "k": k, "results": results,
            "duration_ms": int((time.time() - t0)*1000)
        })
        return top_docs

class DraftingAgent:
    def __init__(self, client, model_name: str, logger: JSONLLogger, temperature: float = 0.4, max_tokens: int = 1000):
        self.client = client
        self.model = model_name
        self.logger = logger
        self.temperature = temperature
        self.max_tokens = max_tokens

    def draft(self, prompt: str, n: int = 3) -> Dict[str, Any]:
        t0 = time.time()
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content":
                 "You are a pharmaceutical scientist. Respond in EXACTLY this structure:\n"
                 "### SinkCondition\n...\n\n### Profile\n... include a Markdown table (Time (min), Dissolved (%))\n"
                 "and a JSON code block with key \"profile\" as a list of {time, dissolved}.\n"
                 "### Recommendations\n..."},
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=0.9,
            n=n,
        )
        latency_ms = int((time.time() - t0) * 1000)
        choices = [c.message.content for c in resp.choices]
        usage = getattr(resp, "usage", None)
        self.logger.log("drafts", {
            "n": n,
            "latency_ms": latency_ms,
            "prompt_sha256": sha256_text(prompt),
            "tokens_in": getattr(usage, "prompt_tokens", None) if usage else None,
            "tokens_out": getattr(usage, "completion_tokens", None) if usage else None,
        })
        return {"choices": choices, "usage": getattr(resp, "usage", None)}

class CriticAgent:
    def __init__(self, logger: JSONLLogger):
        self.logger = logger

    @staticmethod
    def _extract_json_profile(text: str) -> List[Dict[str, float]]:
        # Prefer ```json ... ```
        m = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL|re.IGNORECASE)
        if m:
            try:
                obj = json.loads(m.group(1))
                prof = obj.get("profile")
                if isinstance(prof, list):
                    out = []
                    for p in prof:
                        if isinstance(p, dict) and "time" in p and "dissolved" in p:
                            out.append({"time": float(p["time"]), "dissolved": float(p["dissolved"])})
                    if out:
                        return sorted(out, key=lambda x: x["time"])
            except Exception:
                pass
        # Fallback: parse markdown table with "Time (min)" and "Dissolved (%)"
        tbl = re.search(r"\|?\s*Time\s*\(min\)\s*\|\s*Dissolved\s*\(%\)\s*\|(.+?)(?:\n\n|$)", text, re.DOTALL|re.IGNORECASE)
        if tbl:
            rows = [r.strip() for r in tbl.group(1).strip().splitlines() if r.strip() and not r.strip().startswith("|---")]
            out = []
            for r in rows:
                cols = [c.strip() for c in r.split("|") if c.strip()]
                if len(cols) >= 2:
                    try:
                        t = float(cols[0]); y = float(cols[1])
                        out.append({"time": t, "dissolved": y})
                    except: pass
            if out:
                return sorted(out, key=lambda x: x["time"])
        return []

    def critique(self, drafts: List[str], exp_profile: Optional[List[Dict[str,float]]]) -> Dict[str, Any]:
        metrics_by_idx = {}
        scores = []
        for i, d in enumerate(drafts):
            prof = self._extract_json_profile(d)
            qcm = qc_metrics(prof, exp_profile=exp_profile) if prof else {"ok": False, "reason": "no_profile"}
            # heuristic score: prefer ok + higher f2 (or zero if None) + smoother curve
            f2v = qcm.get("f2") or 0.0
            smooth = qcm.get("smoothness_abs_ddiff") or 1e-6
            base = (1.0 if qcm.get("ok") else 0.0)
            score = base*0.6 + (min(100, max(0, f2v))/100.0)*0.3 + (1.0/(1.0+smooth))*0.1
            scores.append(score)
            metrics_by_idx[str(i)] = qcm
        self.logger.log("critic", {
            "scores": scores,
            "exp_profile_present": bool(exp_profile),
            "metrics_by_idx": metrics_by_idx,
        })
        return {"scores": scores, "metrics_by_idx": metrics_by_idx}

class EditorAgent:
    @staticmethod
    def _isotonic_pav(y):
        y = np.asarray(y, float)
        n = len(y)
        w = np.ones(n, float)
        v = y.copy()
        i = 0
        while i < n-1:
            if v[i] <= v[i+1] + 1e-9:
                i += 1; continue
            j = i
            while j >= 0 and v[j] > v[j+1] + 1e-9:
                totw = w[j] + w[j+1]
                avg = (w[j]*v[j] + w[j+1]*v[j+1]) / totw
                v[j], v[j+1] = avg, avg
                w[j], w[j+1] = totw, totw
                j -= 1
            i += 1
        return v

    def repair_profile(self, profile: List[Dict[str,float]]) -> List[Dict[str,float]]:
        if not profile: return profile
        prof = sorted(profile, key=lambda x: x["time"])
        y = np.array([p["dissolved"] for p in prof], dtype=float)
        y = np.clip(y, 0.0, 100.0)
        y = self._isotonic_pav(y)
        y = np.clip(y, 0.0, 100.0)
        return [{"time": float(prof[i]["time"]), "dissolved": float(y[i])} for i in range(len(prof))]

    @staticmethod
    def _replace_json_profile(text: str, new_profile: List[Dict[str,float]]) -> str:
        block = "```json\n" + json.dumps({"profile": new_profile}, ensure_ascii=False, indent=2) + "\n```"
        if re.search(r"```json\s*\{.*?\}\s*```", text, re.DOTALL|re.IGNORECASE):
            return re.sub(r"```json\s*\{.*?\}\s*```", block, text, flags=re.DOTALL|re.IGNORECASE)
        else:
            return text.strip() + "\n\n" + block + "\n"

    @staticmethod
    def _replace_markdown_table(text: str, new_profile: List[Dict[str,float]]) -> str:
        rows = ["| Time (min) | Dissolved (%) |", "|------------|----------------|"]
        for p in new_profile:
            rows.append(f"| {p['time']:.2f} | {p['dissolved']:.2f} |")
        new_tbl = "\n".join(rows)
        pat = re.compile(r"\|?\s*Time\s*\(min\)\s*\|\s*Dissolved\s*\(%\)\s*\|.*?(?:\n\n|$)", re.DOTALL|re.IGNORECASE)
        if pat.search(text):
            return pat.sub(new_tbl + "\n\n", text)
        else:
            m = re.search(r"(###\s*Profile[^\n]*\n)", text, re.IGNORECASE)
            if m:
                idx = m.end()
                return text[:idx] + new_tbl + "\n\n" + text[idx:]
            return text.strip() + "\n\n" + new_tbl + "\n"
        
    def edit(self, text: str, profile: List[Dict[str,float]]) -> Tuple[str, List[Dict[str,float]]]:
        repaired = self.repair_profile(profile)
        edited = self._replace_markdown_table(text, repaired)
        edited = self._replace_json_profile(edited, repaired)
        return edited, repaired

class JudgeAgent:
    def __init__(self, required_headers: List[str], require_json: bool = True):
        self.required = required_headers
        self.require_json = require_json

    @staticmethod
    def _extract_json_profile(text: str) -> List[Dict[str, float]]:
        m = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL|re.IGNORECASE)
        if not m: return []
        try:
            obj = json.loads(m.group(1))
            prof = obj.get("profile")
            if isinstance(prof, list):
                out = []
                for p in prof:
                    if isinstance(p, dict) and "time" in p and "dissolved" in p:
                        out.append({"time": float(p["time"]), "dissolved": float(p["dissolved"])})
                return sorted(out, key=lambda x: x["time"])
        except Exception:
            return []
        return []

    def check(self, text: str) -> Dict[str, Any]:
        headers_ok = all(h.lower() in text.lower() for h in self.required)
        json_ok = (not self.require_json) or (len(self._extract_json_profile(text)) > 0)
        prof = self._extract_json_profile(text)
        times_ok = len(prof) >= 5
        return {"compliant": bool(headers_ok and json_ok and times_ok),
                "headers_ok": headers_ok, "json_ok": json_ok, "timepoints": len(prof)}

# ============================== Orchestrator ==============================

class PharmaDissolveMCP:
    def __init__(self, api_key: str, model_name: str, kb_path: str = "RAG_database.xlsx", persist_dir: str = "faiss_index"):
        self.logger = JSONLLogger(path="mcp_runs.jsonl", project="PharmaDissolve-MCP")
        self.prompt_filename = os.getenv("PROMPT_FILE_NAME", "0813-FS.txt")
        self.client = initialize_deepseek_client(api_key)
        self.model_name = model_name

        # vector store
        self.vs = PharmaVectorStore()
        try:
            self.vs.load(persist_dir)
        except Exception:
            print("Building new knowledge base...")
            self.vs.build_knowledge_base(kb_path, persist_dir=persist_dir)

        # agents
        self.retrieval = RetrievalAgent(self.vs, self.logger)
        self.drafter = DraftingAgent(self.client, model_name, self.logger, temperature=0.4, max_tokens=1000)
        self.critic = CriticAgent(self.logger)
        self.editor = EditorAgent()
        self.judge = JudgeAgent(required_headers=["SinkCondition", "Profile", "Recommendations"], require_json=True)

    @staticmethod
    def _param_summary(docs: List[Document]) -> str:
        def pick_numeric(field):
            vals = [d.metadata.get(field) for d in docs if isinstance(d.metadata.get(field), (int,float)) and d.metadata.get(field) not in [0.0, None]]
            return np.mean(vals) if vals else None
        drugs = sorted({d.metadata.get("drug_name") for d in docs if d.metadata.get("drug_name")})
        shapes = sorted({d.metadata.get("shape") for d in docs if d.metadata.get("shape") and d.metadata.get("shape")!="Unknown"})
        ps = pick_numeric("particle_size_μm")
        sol = pick_numeric("solubility")
        lines = ["### Parameter Summary (from retrieved sources)"]
        if drugs:  lines.append(f"- Drug(s): {', '.join(drugs)}")
        if ps:     lines.append(f"- Particle size (μm, mean): {ps:.2f}")
        if sol:    lines.append(f"- Solubility (mg/mL, mean): {sol:.4f}")
        if shapes: lines.append(f"- Shapes: {', '.join(shapes)}")
        return "\n".join(lines)

    @staticmethod
    def _first_exp_profile(docs: List[Document]) -> Optional[List[Dict[str,float]]]:
        for d in docs:
            ts = d.metadata.get("time_series") or []
            if ts and isinstance(ts, list):
                try:
                    return [{"time": float(p["time"]), "dissolved": float(p["dissolved"])} for p in ts]
                except: pass
        return None

    @staticmethod
    def _resample_to_standard_grid(profile: List[Dict[str,float]]) -> Optional[List[Dict[str,float]]]:
        grid_env = os.getenv("STANDARD_GRID", "").strip()
        if not grid_env:
            return None
        try:
            grid = [float(x) for x in re.split(r"[,\s]+", grid_env) if x.strip()!=""]
        except Exception:
            return None
        if len(grid) < 3:
            return None
        grid = sorted(set(grid))
        return _interp_profile(profile, grid)

    def _load_profile_fallback(self, path: Optional[str]) -> Optional[List[Dict[str,float]]]:
        if not path or not os.path.exists(path):
            return None
        if path.lower().endswith(".json"):
            obj = json.load(open(path, "r", encoding="utf-8"))
            prof = obj.get("profile") or obj
            return [{"time": float(p["time"]), "dissolved": float(p["dissolved"])} for p in prof]
        # CSV with columns time,dissolved
        df = pd.read_csv(path)
        return [{"time": float(t), "dissolved": float(y)} for t, y in zip(df.iloc[:,0], df.iloc[:,1])]

    def _build_prompt(self, query: str, docs: List[Document]) -> str:
        context = "\n\n".join([d.page_content for d in docs]) if docs else "No relevant context found"
        summary = self._param_summary(docs)
        prompt = (
            f"{summary}\n\n"
            f"## Context\n{context}\n\n"
            "## Task\n"
            "Produce a Markdown report with sections: '### SinkCondition', '### Profile', '### Recommendations'.\n"
            "- 'Profile' must include a Markdown table with columns 'Time (min)' and 'Dissolved (%)', "
            "and a JSON code block with key \"profile\" listing objects {time, dissolved}.\n"
            "- Ensure monotonic increase, 0–100% bounds.\n"
            f"## Query\n{query}\n"
        )
        self.logger.log("prompt", {
            "length": len(prompt),
            "prompt_sha256": sha256_text(prompt),
            "model": self.model_name,
            "provider": os.getenv("LLM_PROVIDER","openrouter"),
            "prompt_filename": self.prompt_filename
        })
        return prompt

    @staticmethod
    def _ensure_dir(path: str):
        os.makedirs(path, exist_ok=True)

    def run(self, query: str, n_candidates: int = 3) -> Dict[str, Any]:
        artifacts_dir = os.path.join("artifacts", f"run_{self.logger.run_id}")
        self._ensure_dir(artifacts_dir)

        # 1) Retrieve (adaptive)
        docs = self.retrieval.retrieve(query, k=6)
        exp_profile = self._first_exp_profile(docs)
        if exp_profile is None:
            exp_profile = self._load_profile_fallback(os.getenv("EXP_PROFILE_PATH"))

        # 2) Prompt & Draft -------------- FIXED ORDER --------------
        prompt = self._build_prompt(query, docs)

        # Write prompt to the env-controlled filename
        prompt_path = os.path.join(artifacts_dir, self.prompt_filename)
        with open(prompt_path, "w", encoding="utf-8") as f:
            f.write(prompt)

        # Also keep your legacy copy (preserves prior behavior)
        with open(os.path.join(artifacts_dir, "0813-FS.txt"), "w", encoding="utf-8") as f:
            f.write(prompt)

        drafts_resp = self.drafter.draft(prompt, n=n_candidates)
        drafts = drafts_resp["choices"]

        # 3) Critique
        crit = self.critic.critique(drafts, exp_profile=exp_profile)
        scores = crit["scores"]
        metrics_by_idx = crit["metrics_by_idx"]
        chosen_idx = int(np.argmax(scores))
        chosen_text = drafts[chosen_idx]
        chosen_profile = CriticAgent._extract_json_profile(chosen_text)

        # 4) Edit (repair curve + replace table/json)
        before_smooth = metrics_by_idx.get(str(chosen_idx), {}).get("smoothness_abs_ddiff")
        edited_text, repaired_profile = self.editor.edit(chosen_text, chosen_profile)

        # 4b) Optional standard grid resample
        resampled = self._resample_to_standard_grid(repaired_profile)
        if resampled:
            repaired_profile = resampled
            # update both table and json blocks with the resampled profile
            edited_text = EditorAgent._replace_markdown_table(edited_text, repaired_profile)
            edited_text = EditorAgent._replace_json_profile(edited_text, repaired_profile)

        edited_qc = qc_metrics(repaired_profile, exp_profile=exp_profile)
        edit_ops = ["isotonic_smooth","clamp_0_100","replace_table_json"]
        if resampled: edit_ops.append("resample_standard_grid")
        self.logger.log("edit", {
            "chosen_idx": chosen_idx,
            "edits": edit_ops,
            "before": {"smooth": before_smooth},
            "after": {"smooth": edited_qc.get("smoothness_abs_ddiff")},
        })

        # 5) Judge
        judge = self.judge.check(edited_text)
        self.logger.log("judge", judge)

        # 6) Save artifacts
        report_path = os.path.join(artifacts_dir, "report.md")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(edited_text)
        profile_path = os.path.join(artifacts_dir, "profile.json")
        with open(profile_path, "w", encoding="utf-8") as f:
            json.dump(repaired_profile, f, ensure_ascii=False, indent=2)
        srcs = [{"sheet": d.metadata.get("sheet"),
                 "drug_name": d.metadata.get("drug_name"),
                 "shape": d.metadata.get("shape"),
                 "solubility": d.metadata.get("solubility"),
                 "particle_size_μm": d.metadata.get("particle_size_μm")}
                for d in docs]

        # 7) Final log
        self.logger.log("final", {
            "chosen_idx": chosen_idx,
            "metrics": edited_qc,
            "judge": judge,
            "n_sources": len(docs),
            "report_path": report_path,
            "profile_json": profile_path,
            "sources": srcs,
            "prompt_filename": self.prompt_filename
        })

        return {
            "report_path": report_path,
            "profile_path": profile_path,
            "qc_metrics": edited_qc,
            "judge": judge,
            "sources": srcs,
            "run_id": self.logger.run_id,
        }

# ============================== CLI / Main ==============================

if __name__ == "__main__":
    # ---- Env setup ----
    # export LLM_PROVIDER=openrouter
    # export OPENROUTER_API_KEY=or_...        # your OpenRouter key
    # export OPENROUTER_SITE_URL=http://localhost
    # export OPENROUTER_APP_TITLE=PharmaDissolve-MCP
    # export LLM_MODEL=deepseek/deepseek-chat-v3.1:free
    # Optional: export EXP_PROFILE_PATH=./hctz_profile.csv  (or .json with {"profile":[...]})
    # Optional: export STANDARD_GRID="0,5,10,15,30,45,60"
    api_key = os.getenv("OPENROUTER_API_KEY", "xxxxxxxxx")
    model = os.getenv("LLM_MODEL", "deepseek/deepseek-chat-v3.1:free")

    system = PharmaDissolveMCP(api_key=api_key, model_name=model, kb_path="RAG_database.xlsx", persist_dir="faiss_index")

    query = "Predict dissolution profile for Drug Dissolved (%)"
    outcome = system.run(query=query, n_candidates=3)

    print("Run ID:", outcome["run_id"])
    print("Report:", outcome["report_path"])
    print("Profile JSON:", outcome["profile_path"])
    print("QC:", json.dumps(outcome["qc_metrics"], indent=2))
    print("Judge:", json.dumps(outcome["judge"], indent=2))
