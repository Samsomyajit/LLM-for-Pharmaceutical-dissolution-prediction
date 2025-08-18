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


# ------------------------ initialization ------------------------
def initialize_deepseek_client(api_key: str) -> Ark:
    return Ark(
        api_key=api_key,
        timeout=1800,
        base_url="https://ark.cn-beijing.volces.com/api/v3"
    )


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
        drug_boost = 1.0
        if 'drug' in doc_meta and any(
                doc_meta['drug'] in name for name in self._extract_drug_names(query_params.get('drug_name', ''))):
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
                return float(re.sub(r"[^\d.+-]", "", str(raw_value)))
            except:
                return 0.0

        return str(raw_value).strip()

    def parse_excel_sheet(self, file_path: str, sheet_name: str) -> tuple:
        df = pd.read_excel(file_path, sheet_name=sheet_name).fillna("")
        metadata = {}
        field_map = {
            "drug_name": [r"(?i)^\s*(drug|product)\s*name\s*$"],
            "particle_size": [r"(?i)particle\s+size"],
            "solubility": [r"(?i)solubility"],
            "model_type": [r"(?i)model\s+type"],
            "diffusion_coeff": [r"(?i)diffusion\s+coefficient"],
            "true_density": [r"(?i)true\s+density"],
            "shape": [r"(?i)^shape$"],
            "surface_area": [r"(?i)surface\s+area"],
            "reference": [r"(?i)reference"]
        }
        for field, patterns in field_map.items():
            for row_idx, row in df.iterrows():
                for col_idx, cell in enumerate(row):
                    if any(re.match(pattern, str(cell), re.IGNORECASE) for pattern in patterns):
                        value = row[col_idx + 1] if col_idx + 1 < len(row) else ""
                        metadata[field] = self._clean_value(value, field)
                        break
                if field in metadata:
                    break

        time_data = []
        for _, row in df.iterrows():
            if re.match(r"(?i)^time", str(row[0])):
                time_col = 0
                diss_col = 1
                for i in range(row.name + 1, len(df)):
                    time_val = df.iloc[i, time_col]
                    diss_val = df.iloc[i, diss_col]
                    if pd.isna(time_val) or pd.isna(diss_val):
                        break
                    try:
                        time_data.append({
                            "time": float(time_val),
                            "dissolved": float(diss_val)
                        })
                    except:
                        continue
                break

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
            [f"{t['time']} min: {t['dissolved']}%"
             for t in time_series[:5]]
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
                continue
            doc = Document(
                page_content=self.generate_semantic_text(meta, times),
                metadata={

                    "model_type": meta.get("model_type", "Unknown"),
                    "particle_size_μm": meta.get("particle_size", 0.0),
                    "shape": meta.get("shape", "Unknown"),
                    "sheet": sheet
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
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=75,
            separators=["\n\n", "Dissolution Profile:", r"\d+ min: "]
        )
        chunks = text_splitter.split_documents(docs)
        self.vectorstore = FAISS.from_documents(chunks, self.embeddings)
        self.vectorstore.save_local(persist_dir)
        return chunks

    def load_knowledge_base(self, persist_dir: str = "faiss_index"):
        self.vectorstore = FAISS.load_local(
            persist_dir,
            self.embeddings,
            allow_dangerous_deserialization=True
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
        docs = []
        context = ""
        if use_rag:
            docs = self.retrieve_relevant_info(query)
            context = "\n\n".join([d["content"] for d in docs])

        base_prompt = """You are a pharmaceutical formulation expert. Always respond in English using markdown format provided in prompt_template ExampleOutput."""

        if prompt_template and os.path.exists(prompt_template):
            with open(prompt_template, 'r', encoding='utf-8') as f:
                try:
                    # Handle JSON parsing with control characters
                    template_data = json.loads(f.read(), strict=False)

                    # Pre-process sections outside f-string
                    mandatory_sections = "  \n".join(
                        template_data['Request']['OutputRequirements']['Format']['MandatorySections']
                    )
                    critical_constraints = "  \n".join(
                        template_data['Request']['Constraints']
                    )
                    example_template = template_data['Request']['ExampleOutput']['MarkdownTemplate']

                    base_prompt = f"""# Role Definition
{template_data['Role']}

## Background
{template_data['Background']['Experience']}
**Key References**: {', '.join(template_data['Background']['KeyReferences'])}

## Contextual Data
{{context}}

## Request Specification
**Objective**: {template_data['Request']['Objective']}

### Input Parameters
"""
                    for param, value in template_data['Request']['InputParameters'].items():
                        base_prompt += f"- {param}: {value}\n"

                    base_prompt += f"""
### Mandatory Output Structure
{mandatory_sections}

### Critical Constraints
{critical_constraints}

### Example Response
{example_template}

## Current Query
{{query}}

**STRICT REQUIREMENTS**:
- EXACT table format replication
- LaTeX equations with \\(...\\) delimiters
- All time points 0-360 min"""
                except Exception as e:
                    print(f"Template error: {str(e)}")
### 1. Sink Condition Evaluation
### 2. Dissolution Model & Assumptions
### 3. Predicted Dissolution Profile (table)
### 4. Critical Analysis
### 5. Recommendations"""

        final_prompt = base_prompt.format(
            context=context if context else "No relevant context found",
            query=query
        )

        # Enhanced system message for strict formatting
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "system",
                    "content": "You are a pharmaceutical scientist. Respond in EXACTLY this structure:"
                               "\n1. Use Markdown with EXACTLY the specified headers"
                               "\n2. Tables MUST use same columns as example"
                               "\n3. Equations MUST use \\(latex syntax\\) with double backslashes"
                               "\n4. Time points 0, 5, 10, 15, 30, 45, 60, 90, 120, 180, 240, 300 min"

                },
                {
                    "role": "user",
                    "content": final_prompt
                }
            ],
            temperature=0.7, # Lower temperature for stricter formatting
            max_tokens=1000,
            top_p=0.9,
        )

        result = {
            "query": query,
            "response": response.choices[0].message.content,
            "timestamp": datetime.datetime.now().isoformat(),
            "sources": [d["metadata"] for d in docs] if use_rag else []
        }

        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

        return result["response"]


# ------------------------ main ------------------------
if __name__ == "__main__":
    API_KEY = "xxxxxxxxx"
    rag_system = PharmaRAGSystem(API_KEY)

    try:
        rag_system.load_knowledge_base()
    except:
        print("Building new knowledge base...")
        rag_system.build_knowledge_base("RAG.xlsx")

    query = "Predict dissolution profile for Drug Dissolved (%)"
    response = rag_system.generate_response(
        query=query,
        prompt_template="0813-FS-CoT.txt",
    )

    print("Query:", query)
    print("Response:\n", response)
