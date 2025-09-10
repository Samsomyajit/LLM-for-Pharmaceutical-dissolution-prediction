import numpy as np
import re
import pandas as pd
import json
import datetime
import os
import warnings
from typing import List, Dict, Any, Optional
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from volcenginesdkarkruntime import Ark
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

warnings.filterwarnings("ignore", category=DeprecationWarning)

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# ------------------------ initialization ------------------------
def initialize_deepseek_client(api_key: str):
    import os
    provider = os.getenv("LLM_PROVIDER", "groq")

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
            "X-Title": os.getenv("OPENROUTER_APP_TITLE", "PharmaRAG"),
        }

        def create_fn(**kwargs):
            # add default headers if caller didn't supply any
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
                kwargs["model"] = os.getenv("LLM_MODEL", "llama3.1:8b")
            return create_fn(**kwargs)

    class _Chat:
        def __init__(self): self.completions = _Completions()

    class ShimClient:
        def __init__(self): self.chat = _Chat()

    return ShimClient()




# ------------------------ adap ------------------------
class AdaptiveWeightedRetriever:
    def __init__(self):
        self.default_weights = {
            'particle_size': 0.9,  # Most critical factor (surface area)
            'solubility': 0.3,      # Thermodynamic driving force
            'surfactant': 0.3,      # New parameter for wetting agents
            'polymorph': 0.3,        # Crystal form optimization
            'disintegration': 0.3,  # Tablet breakdown speed
            'model_type': 0.3,
            'diffusion_coeff': 0.3
        }
        self.metadata_boost = 0.3
        self.query_weights = self.default_weights.copy()
        self.surface_area_boost = 2.0  # Direct exponential impact
        self.nano_threshold = 0.2

    def _normalize_query(self, query: str) -> Dict:
        param_patterns = {
            'particle_size': r"(\d+\.?\d*)\s*μm",
            'solubility': r"Solubility:\s*(\d+\.?\d*)",
            'diffusion_coeff': r"Diffusion\s*Coefficient:\s*([\d.e-]+)",
            'true_density': r"Density:\s*(\d+\.?\d*)",
            'surface_area': r"Surface\s*Area:\s*(\d+\.?\d*)",
            'shape': r"Shape:\s*\"([^\"]+)\"",
            'model_type': r"Model:\s*'([^']+)'"
        }
        extracted = {}
        for param, pattern in param_patterns.items():
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                try:
                    extracted[param] = float(match.group(1)) if param != 'shape' else match.group(1).lower()
                except:
                    extracted[param] = match.group(1).lower() if param == 'shape' else 0.0
        return extracted

    def _gaussian_similarity(self, a: float, b: float, sigma: float) -> float:
        return np.exp(-(a - b) ** 2 / (2 * sigma ** 2))

    def _extract_drug_names(self, text: str) -> List[str]:
        patterns = [
            r"\b(?:Drug|API)\s+([A-Z][a-z]+)\b",
            r"for\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+dissolution"
        ]
        names = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            names.extend([m.strip().title() for m in matches])
        return list(set(names))

    def calculate_score(self, doc_meta: Dict, query_params: Dict) -> float:
        score = 0.0
        num_params = ['particle_size', 'solubility', 'diffusion_coeff', 'true_density', 'surface_area']

        # Enhanced numerical matching with dynamic decay
        for param in num_params:
            if param in query_params and param in doc_meta:
                q_val = query_params[param]
                doc_val = doc_meta.get(f"{param}_μm", 0.0) if param == 'particle_size' else doc_meta.get(param, 0.0)

                # Dynamic decay based on parameter type
                decay_factor = {
                    'particle_size': 0.3,
                    'solubility': 0.25,
                    'diffusion_coeff': 0.4,
                    'true_density': 0.5,
                    'surface_area': 0.3
                }.get(param, 0.25)

                sigma = max(abs(q_val * decay_factor), 1e-3)
                score += self.query_weights[param] * self._gaussian_similarity(doc_val, q_val, sigma)
        # Model type exact match boost
        if 'model_type' in query_params and 'model_type' in doc_meta:
            if str(doc_meta['model_type']).lower() == str(query_params['model_type']).lower():
                score += self.query_weights['model_type'] * 1.2  # Extra boost for exact matches
        # Shape handling with synonym matching
        shape_synonyms = {
            'spherical': ['round', 'globular'],
            'irregular': ['angular', 'rough'],
            'crystalline': ['needle-like', 'prismatic']
        }
        if 'shape' in query_params and 'shape' in doc_meta:
            q_shape = str(query_params['shape']).lower()
            doc_shape = str(doc_meta['shape']).lower()
            if q_shape == doc_shape:
                score += self.query_weights['shape']
            else:
                for key, syns in shape_synonyms.items():
                    if q_shape in syns and doc_shape == key:
                        score += self.query_weights['shape'] * 0.8
                        break
        # Enhanced drug name matching
        # inside calculate_score(...)
        drug_boost = 1.0
        q_drug = (query_params.get('drug_name') or "").strip().lower()
        doc_drug = (doc_meta.get('drug_name') or "").strip().lower()
        if q_drug and doc_drug and (q_drug in doc_drug or doc_drug in q_drug):
            drug_boost = 1 + self.metadata_boost
        return score * drug_boost


    def adaptive_retrieval(self, query: str, docs: List[Dict], k: int = 3) -> List[Dict]:
        query_params = self._normalize_query(query)
        total_params = len(query_params)
        if total_params > 0:
            weight_scale = 1 / total_params
            self.query_weights = {k: v * weight_scale for k, v in self.query_weights.items()}

        scored_docs = []
        for doc in docs:
            score = self.calculate_score(doc['metadata'], query_params)
            scored_docs.append((score, doc))

        scored_docs.sort(key=lambda x: x[0], reverse=True)
        return [doc for score, doc in scored_docs[:k]]


# ------------------------ data_process ------------------------
class PharmaDataProcessor:
    @staticmethod
    def _clean_value(raw_value, field_type: str):
        if pd.isna(raw_value) or str(raw_value).lower() in ["", "na", "n/a", "nan"]:
            if field_type == "drug_name":
                return None
            return "Unknown" if field_type in ["model_type", "shape"] else 0.0
        if field_type == "drug_name":
            cleaned = str(raw_value).strip().title()
            return cleaned if len(cleaned) > 2 else None
        if field_type in ["particle_size", "solubility", "diffusion_coeff", "true_density", "surface_area"]:
            try:
                s = str(raw_value).replace("µ", "μ")
                return float(re.sub(r"[^\d.+-eE]", "", s))
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

        metadata = {}
        metadata["drug_name"]       = self._clean_value(find_header([r"^\s*(drug|product|api)\s*name\s*$", r"^\s*drug\s*$"]), "drug_name")
        metadata["particle_size"]   = self._clean_value(find_header([r"particle\s*size"]), "particle_size")
        metadata["solubility"]      = self._clean_value(find_header([r"solubility"]), "solubility")
        metadata["model_type"]      = self._clean_value(find_header([r"model\s*type", r"model"]), "model_type")
        metadata["diffusion_coeff"] = self._clean_value(find_header([r"diffusion\s*coefficient"]), "diffusion_coeff")
        metadata["true_density"]    = self._clean_value(find_header([r"(true\s*)?density"]), "true_density")
        metadata["shape"]           = self._clean_value(find_header([r"^shape$"]), "shape")
        metadata["surface_area"]    = self._clean_value(find_header([r"surface\s*area"]), "surface_area")
        metadata["reference"]       = find_header([r"reference"])

        # time series
        time_data, best = [], None
        time_hdr = re.compile(r"^\s*time", re.IGNORECASE)
        for r in range(n_rows):
            for c in range(n_cols):
                if time_hdr.match(cell(r, c)):
                    cand_cols = [c+1] + [j for j in range(n_cols) if j != c]
                    for dc in cand_cols:
                        rows, rr = [], r + 1
                        while rr < n_rows:
                            t_raw, v_raw = cell(rr, c), cell(rr, dc)
                            if t_raw == "" or v_raw == "":
                                break
                            try:
                                t = float(re.sub(r"[^\d.+-]", "", t_raw))
                                v = float(re.sub(r"[^\d.+-]", "", v_raw))
                                rows.append({"time": t, "dissolved": v})
                            except:
                                break
                            rr += 1
                        if len(rows) >= 2:
                            best = rows; break
                if best: break
            if best: break
        if best: time_data = best

        return metadata, time_data

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
                }
            )
            docs.append(doc)
        return docs



# ------------------------ RAG_mode ------------------------
class PharmaRAGSystem:
    def __init__(self, api_key: str, model_name: str = "deepseek-r1-250120"):
        self.client = initialize_deepseek_client(api_key)
        self.model_name = model_name
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        self.vectorstore = None
        self.processor = PharmaDataProcessor()

    def build_knowledge_base(self, file_path: str, persist_dir: str = "faiss_index"):
        docs = self.processor.process_pharma_data(file_path)
        if not docs:
            raise ValueError(f"No documents parsed from {file_path}. Check sheet headers.")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=75)
        chunks = text_splitter.split_documents(docs)
        if not chunks:
            raise ValueError("Got 0 chunks from documents; inspect parsing/generation.")
        self.vectorstore = FAISS.from_documents(chunks, self.embeddings)
        self.vectorstore.save_local(persist_dir)
        return chunks


    def load_knowledge_base(self, persist_dir: str = "faiss_index"):
        index_path = os.path.join(persist_dir, "index.faiss")
        if not os.path.exists(index_path):
            raise FileNotFoundError(index_path)
        self.vectorstore = FAISS.load_local(
            persist_dir, self.embeddings, allow_dangerous_deserialization=True
        )


    def retrieve_relevant_info(self, query: str, k: int = 3) -> List[Dict]:
        if not self.vectorstore:
            raise ValueError("Knowledge base not loaded")

        raw_docs = self.vectorstore.similarity_search(query, k=k * 2)
        retriever = AdaptiveWeightedRetriever()
        processed = [{"content": d.page_content, "metadata": d.metadata} for d in raw_docs]
        return retriever.adaptive_retrieval(query, processed, k=k)


    def generate_response(self, query: str,
                        use_rag: bool = True,
                        prompt_template: str = None,
                        output_file: str = None) -> str:
        """Builds context (optional), constructs a robust prompt from JSON or plain-text
        template, calls the provider-compatible chat.completions.create, and returns text."""
        # ----- RAG context -----
        docs = []
        context = ""
        if use_rag:
            docs = self.retrieve_relevant_info(query)
            context = "\n\n".join([d["content"] for d in docs]) if docs else ""

        # ----- helper: robust template loader -----
        def _build_prompt_from_template(path: str, ctx: str, q: str) -> str:
            raw = open(path, "r", encoding="utf-8").read()
            try:
                data = json.loads(raw, strict=False)
            except Exception:
                # Treat as plain-text instructions
                return (
                    f"{raw}\n\n"
                    f"## Context\n{ctx or 'No relevant context found'}\n\n"
                    f"## Current Query\n{q}\n"
                    "**STRICT REQUIREMENTS**:\n"
                    "- If an example table is implied, keep the same columns.\n"
                    "- Use LaTeX inline math with \\(...\\).\n"
                    "- Include the requested dissolution time grid if specified.\n"
                )

            # Accept both new (SystemPrompt) and legacy flat schemas
            root = data.get("SystemPrompt", data)

            # Role / background
            role = root.get("Role", "Pharmaceutical formulation expert")
            bg = root.get("Background", {})
            if isinstance(bg, dict):
                background = (
                    bg.get("Experience")
                    or bg.get("Text")
                    or ", ".join(f"{k}: {v}" for k, v in bg.items() if isinstance(v, (str, int, float)))
                )
            else:
                background = str(bg) if bg else ""

            # Request section (robust to missing keys)
            req = root.get("Request") or {}
            objective = req.get("Objective", "")
            inputs = req.get("InputParameters") or {}

            # Constraints may live at either level
            constraints = root.get("Constraints") or req.get("Constraints") or []
            if isinstance(constraints, str):
                constraints = [constraints]

            # Mandatory sections & example
            # Try new-style OutputTemplate first
            out_t = root.get("OutputTemplate") or req.get("OutputTemplate") or {}
            mandatory_sections = []
            example_md = None
            if isinstance(out_t, dict) and out_t:
                mandatory_sections = list(out_t.keys())
                example_md = out_t.get("MarkdownTemplate")

            # Try legacy path if still empty
            if not mandatory_sections:
                try:
                    mandatory_sections = req["OutputRequirements"]["Format"]["MandatorySections"]
                except Exception:
                    mandatory_sections = []

            if not mandatory_sections:
                mandatory_sections = [
                    "Sink Condition Evaluation",
                    "Dissolution Model & Assumptions",
                    "Predicted Dissolution Profile (table)",
                    "Critical Analysis",
                    "Recommendations",
                ]

            if not example_md:
                example_md = req.get("ExampleOutput", {}).get("MarkdownTemplate") or \
                            "\n".join([f"### {s}" for s in mandatory_sections])

            # Assemble prompt
            lines = []
            lines.append(f"# Role Definition\n{role}\n")
            if background:
                lines.append(f"## Background\n{background}\n")
            lines.append(f"## Contextual Data\n{ctx or 'No relevant context found'}\n")
            lines.append("## Request Specification")
            if objective:
                lines.append(f"**Objective**: {objective}\n")
            lines.append("### Input Parameters")
            if inputs:
                for k, v in inputs.items():
                    lines.append(f"- {k}: {v}")
            else:
                lines.append("- (none)")

            lines.append("\n### Mandatory Output Structure")
            for s in mandatory_sections:
                lines.append(f"- {s}")

            lines.append("\n### Critical Constraints")
            if constraints:
                for c in constraints:
                    lines.append(f"- {c}")
            else:
                lines.append("- None specified")

            lines.append("\n### Example Response")
            lines.append(example_md)

            lines.append("\n## Current Query")
            lines.append(q)

            lines.append(
                "\n**STRICT REQUIREMENTS**:\n"
                "- EXACT table format replication if a template provides one\n"
                "- LaTeX equations with \\(...\\) delimiters\n"
                "- Include the requested time grid (e.g., 0–300 min) if specified\n"
            )
            return "\n".join(lines)

        # ----- final prompt -----
        if prompt_template and os.path.exists(prompt_template):
            try:
                final_prompt = _build_prompt_from_template(prompt_template, context, query)
            except Exception as e:
                print(f"Template error (robust loader): {e}")
                final_prompt = (
                    "You are a pharmaceutical formulation expert.\n\n"
                    f"## Context\n{context or 'No relevant context found'}\n\n"
                    f"## Current Query\n{query}\n"
                )
        else:
            final_prompt = (
                "You are a pharmaceutical formulation expert.\n\n"
                f"## Context\n{context or 'No relevant context found'}\n\n"
                f"## Current Query\n{query}\n"
            )

        # ----- LLM call -----
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a pharmaceutical scientist. Respond in EXACTLY this structure:\n"
                        "1) Use Markdown with the specified headers.\n"
                        "2) If an example table exists, keep the same columns.\n"
                        "3) Use LaTeX inline math with \\(...\\).\n"
                        "4) Include the requested dissolution time grid if provided."
                    ),
                },
                {"role": "user", "content": final_prompt},
            ],
            temperature=0.7,
            max_tokens=1000,
            top_p=0.9,
        )

        result = {
            "query": query,
            "response": response.choices[0].message.content,
            "timestamp": datetime.datetime.now().isoformat(),
            "sources": [d["metadata"] for d in docs] if use_rag else [],
        }

        if output_file:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

        return result["response"]


# ------------------------ main ------------------------
if __name__ == "__main__":
    API_KEY = "xxxxxxxxx"
    rag_system = PharmaRAGSystem(API_KEY, model_name=os.getenv("LLM_MODEL","llama-3.1-70b-versatile"))

    try:
        rag_system.load_knowledge_base()
    except:
        print("Building new knowledge base...")
        rag_system.build_knowledge_base("RAG_database.xlsx")

    query = "Predict dissolution profile for Drug Dissolved (%)"
    response = rag_system.generate_response(
        query=query,
        prompt_template="0813-FS-CoT.txt",
    )

    print("Query:", query)
    print("Response:\n", response)
