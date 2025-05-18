import os
import re
import json
import time
from typing import List, Dict
from pathlib import Path
from datetime import datetime

import requests
import numpy as np
import torch
from bs4 import BeautifulSoup
from pymilvus import connections, Collection, utility, FieldSchema, CollectionSchema, DataType
from sentence_transformers import CrossEncoder
from transformers import AutoTokenizer, AutoModel
import streamlit as st
from rank_bm25 import BM25Okapi
import jieba
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

#个人配置
your_uri="uri_path"
your_token="token_name"
your_api="secret_api"
file_path=r"your_filepath"

# 云端Milvus连接（建议将敏感信息移到环境变量）
connections.connect(
    uri=your_uri,
    token=your_token
)

# ======================
# 系统配置
# ======================
class SystemConfig:
    EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"
    EMBEDDING_DIM = 512
    EMBEDDING_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    MAX_TEXT_LENGTH = 10000
    CHUNK_SIZE = 512
    CHUNK_OVERLAP = 50
    MAX_DOC_ID_LENGTH = 256
    COLLECTION_NAME = "medical_rag"
    MEDICAL_TERMS_PATH = "medical_terms.json"

    # 新增rerank配置
    RERANK_CONFIG = {
        "enable_rerank": True,
        "cross_encoder_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "initial_recall_num": 20,
        "final_top_k": 5,
        "score_weights": {
            "vector": 0.5,
            "keyword": 0.3,
            "rerank": 0.2,
            "diversity": -0.1
        },
        "diversity_threshold": 0.7,
        "min_rerank_score": 0.2
    }


# ======================
# 硅基流动API客户端
# ======================
class SiliconFlowClient:
    def __init__(self):
        self.base_url = "https://api.siliconflow.cn/v1/chat/completions"
        self.api_key = your_api
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["POST"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("https://", adapter)

    def generate_answer(self, question: str, context: List[str]) -> dict:
        system_prompt = """你是一名专业医疗助手，请基于以下医学资料回答问题：
{}

回答要求：
1. 核心定义简明扼要
2. 关键特征分点列出（最多3点）
3. 必须标注数据来源
4. 使用中文口语化表达""".format('\n'.join([f"[来源:{i + 1}] {text}" for i, text in enumerate(context)]))

        payload = {
            "model": "Qwen/Qwen3-30B-A3B",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question}
            ],
            "temperature": 0.3,
            "max_tokens": 1000
        }

        try:
            response = self.session.post(
                self.base_url,
                headers=self.headers,
                json=payload,
                timeout=(10, 30)
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.Timeout as e:
            st.error("请求超时，建议：1. 检查网络连接 2. 简化问题 3. 稍后重试")
            return {"error": "API请求超时"}
        except requests.exceptions.RequestException as e:
            st.error(f"API请求失败: {str(e)}")
            return {"error": str(e)}


# ======================
# 疾病关键词提取器
# ======================
class DiseaseExtractor:
    def __init__(self):
        self.medical_terms = self._load_medical_terms()
        self._initialize_jieba()

    def _load_medical_terms(self) -> dict:
        default_terms = {
            "diseases": [
                "高血压", "糖尿病", "冠心病", "艾滋病", "抑郁症", "帕金森",
                "白血病", "肺炎", "肝炎", "肾炎", "胃炎", "肠炎", "皮炎",
                "肺癌", "胃癌", "肝癌", "乳腺癌", "前列腺癌"
            ],
            "symptoms": [
                "头痛", "发热", "咳嗽", "呕吐", "腹泻", "皮疹", "瘙痒",
                "疼痛", "肿胀", "乏力", "眩晕", "心悸", "气短"
            ],
            "synonyms": {
                "糖尿病": ["高血糖", "消渴症"],
                "冠心病": ["心肌缺血", "心绞痛"],
                "高血压": ["血压高"]
            }
        }

        try:
            if Path(SystemConfig.MEDICAL_TERMS_PATH).exists():
                with open(SystemConfig.MEDICAL_TERMS_PATH, "r", encoding="utf-8") as f:
                    return json.load(f)
            return default_terms
        except:
            return default_terms

    def _initialize_jieba(self):
        for disease in self.medical_terms["diseases"]:
            jieba.add_word(disease)
        for symptom in self.medical_terms["symptoms"]:
            jieba.add_word(symptom)

    def extract(self, text: str) -> List[str]:
        words = jieba.lcut(text)
        diseases = [w for w in words if w in self.medical_terms["diseases"]]
        symptoms = [w for w in words if w in self.medical_terms["symptoms"]]

        compound_pattern = re.compile(
            r"([\u4e00-\u9fa5]{1,5}[\u4e00-\u9fa5]+(?:病|症|炎|癌|瘤|结石|综合征|综合症|紊乱|衰竭|中毒|损伤))"
        )
        compounds = compound_pattern.findall(text)

        return list(set(diseases + symptoms + compounds))


# ======================
# 文档预处理
# ======================
class MedicalDataPreprocessor:
    def __init__(self):
        self.disease_extractor = DiseaseExtractor()

    def clean_text(self, text: str) -> str:
        if not isinstance(text, str):
            return ""

        text = BeautifulSoup(text, 'html.parser').get_text(separator='\n')
        paragraphs = [p for p in text.split('\n') if len(p) > 15]
        return '\n'.join(paragraphs)[:SystemConfig.MAX_TEXT_LENGTH]

    def get_doc_id(self, file_path: str) -> str:
        fname = Path(file_path).name
        if len(fname) > SystemConfig.MAX_DOC_ID_LENGTH:
            return f"{fname[:8]}_{hash(file_path)[:8]}"
        return fname


# ======================
# 文档分块
# ======================
class MedicalDocumentChunker:
    def __init__(self):
        self.splitter = re.compile(r'(?<=[。！？；\n])')
        self.disease_extractor = DiseaseExtractor()

    def chunk_document(self, document: dict) -> List[dict]:
        content = document.get("content", "")
        sentences = [s.strip() for s in self.splitter.split(content) if s.strip()]

        chunks = []
        current_chunk = []
        current_len = 0

        for sent in sentences:
            sent_len = len(sent)
            if current_len + sent_len > SystemConfig.CHUNK_SIZE:
                if self._is_valid_chunk(current_chunk):
                    chunks.append(self._create_chunk(current_chunk, document["doc_id"], len(chunks)))
                current_chunk = current_chunk[-SystemConfig.CHUNK_OVERLAP:]
                current_len = sum(len(s) for s in current_chunk)

            current_chunk.append(sent)
            current_len += sent_len

        if self._is_valid_chunk(current_chunk):
            chunks.append(self._create_chunk(current_chunk, document["doc_id"], len(chunks)))

        return chunks

    def _create_chunk(self, sentences: List[str], doc_id: str, chunk_id: int) -> dict:
        text = ''.join(sentences)
        return {
            "text": text,
            "doc_id": doc_id,
            "chunk_id": chunk_id,
            "entities": self.disease_extractor.extract(text)
        }

    def _is_valid_chunk(self, sentences: List[str]) -> bool:
        return len(''.join(sentences)) >= 50


# ======================
# 检索模块(完整修改版)
# ======================
class MedicalRetriever:
    def __init__(self, preprocessor: MedicalDataPreprocessor):
        self.preprocessor = preprocessor
        self.tokenizer, self.model, self.cross_encoder = load_models()
        self.collection = self._initialize_milvus()
        self.disease_extractor = DiseaseExtractor()
        self._initialize_reranker()
        self.similarity_model = None

    def _initialize_milvus(self):
        try:
            if not utility.has_collection(SystemConfig.COLLECTION_NAME):
                self._create_collection_with_index()
            else:
                self.collection = Collection(SystemConfig.COLLECTION_NAME)
                self._validate_collection_schema()

            self.collection.load()
            st.success("成功连接云端Milvus服务")
            return self.collection
        except Exception as e:
            st.error(f"Milvus连接失败: {str(e)}")
            raise RuntimeError("无法连接Milvus服务") from e

    def _create_collection_with_index(self):
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=SystemConfig.MAX_DOC_ID_LENGTH),
            FieldSchema(name="chunk_id", dtype=DataType.INT64),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=SystemConfig.EMBEDDING_DIM)
        ]

        schema = CollectionSchema(fields, description="医疗文档向量库")
        self.collection = Collection(SystemConfig.COLLECTION_NAME, schema)

        index_params = {
            "index_type": "HNSW",
            "metric_type": "IP",
            "params": {"M": 16, "efConstruction": 200}
        }
        self.collection.create_index("embedding", index_params)

    def _validate_collection_schema(self):
        current_schema = self.collection.schema
        required_fields = {"doc_id", "text", "embedding"}
        if not required_fields.issubset({f.name for f in current_schema.fields}):
            raise ValueError("集合schema不兼容，建议删除重建")

    def _initialize_reranker(self):
        if SystemConfig.RERANK_CONFIG["enable_rerank"]:
            try:
                if self.cross_encoder is None:
                    self.cross_encoder = CrossEncoder(
                        SystemConfig.RERANK_CONFIG["cross_encoder_model"]
                    )
                self.similarity_model = AutoModel.from_pretrained(
                    "BAAI/bge-small-zh-v1.5",
                    device_map=SystemConfig.EMBEDDING_DEVICE
                )
                st.success("重排序模型加载成功")
            except Exception as e:
                st.warning(f"重排序模型加载失败: {str(e)}")
                SystemConfig.RERANK_CONFIG["enable_rerank"] = False

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        if self.similarity_model is None:
            return 0

        inputs = self.tokenizer(
            [text1, text2],
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(SystemConfig.EMBEDDING_DEVICE)

        with torch.no_grad():
            outputs = self.similarity_model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)
            sim = torch.cosine_similarity(embeddings[0], embeddings[1], dim=0)
            return float(sim)

    def _apply_diversity_penalty(self, results: List[dict]) -> List[dict]:
        if len(results) < 2:
            return results

        sim_matrix = np.zeros((len(results), len(results)))
        for i in range(len(results)):
            for j in range(i + 1, len(results)):
                sim = self._calculate_similarity(
                    results[i]["text"],
                    results[j]["text"]
                )
                sim_matrix[i][j] = sim
                sim_matrix[j][i] = sim

        for i in range(len(results)):
            similar_docs = np.where(
                sim_matrix[i] > SystemConfig.RERANK_CONFIG["diversity_threshold"]
            )[0]
            if len(similar_docs) > 0:
                max_score_idx = max(similar_docs, key=lambda x: results[x]["score"])
                if i != max_score_idx:
                    results[i]["diversity_penalty"] = (
                            SystemConfig.RERANK_CONFIG["score_weights"]["diversity"] *
                            max(sim_matrix[i][similar_docs])
                    )

        return results

    def _rerank_results(self, query: str, results: List[dict]) -> List[dict]:
        if not SystemConfig.RERANK_CONFIG["enable_rerank"] or len(results) < 2:
            return results[:SystemConfig.RERANK_CONFIG["final_top_k"]]

        # 关键词增强
        keywords = self.disease_extractor.extract(query)
        for res in results:
            res["keyword_score"] = sum(
                1 for kw in keywords
                if kw in res["text"] or any(kw in syn for syn in res.get("keywords", []))
            )

        # CrossEncoder重排序
        pairs = [(query, res["text"]) for res in results]
        rerank_scores = self.cross_encoder.predict(pairs)

        max_rerank = max(rerank_scores) if max(rerank_scores) > 0 else 1
        for i, res in enumerate(results):
            res["rerank_score"] = float(rerank_scores[i] / max_rerank)
            if res["rerank_score"] < SystemConfig.RERANK_CONFIG["min_rerank_score"]:
                res["rerank_score"] = 0

        # 多样性惩罚
        results = self._apply_diversity_penalty(results)

        # 综合评分
        max_vec = max(r["score"] for r in results) or 1
        max_kw = max(r["keyword_score"] for r in results) or 1

        for res in results:
            res["final_score"] = (
                    SystemConfig.RERANK_CONFIG["score_weights"]["vector"] * (res["score"] / max_vec) +
                    SystemConfig.RERANK_CONFIG["score_weights"]["keyword"] * (res["keyword_score"] / max_kw) +
                    SystemConfig.RERANK_CONFIG["score_weights"]["rerank"] * res["rerank_score"] +
                    res.get("diversity_penalty", 0)
            )

        return sorted(
            [r for r in results if r["final_score"] > 0],
            key=lambda x: x["final_score"],
            reverse=True
        )[:SystemConfig.RERANK_CONFIG["final_top_k"]]

    def retrieve(self, query: str, top_k: int = None) -> List[dict]:
        if top_k is None:
            top_k = SystemConfig.RERANK_CONFIG["final_top_k"]

        initial_recall = SystemConfig.RERANK_CONFIG["initial_recall_num"]

        query_embedding = self._embed_texts([query])[0]
        search_params = {"metric_type": "IP", "params": {"nprobe": 16}}

        results = self.collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=initial_recall,
            output_fields=["text", "doc_id", "chunk_id"]
        )

        results = [{
            "text": hit.entity.get("text"),
            "doc_id": hit.entity.get("doc_id"),
            "chunk_id": hit.entity.get("chunk_id"),
            "score": hit.score,
            "keywords": self.disease_extractor.extract(hit.entity.get("text"))
        } for hit in results[0]]

        return self._rerank_results(query, results)

    def _embed_texts(self, texts: List[str]) -> List[List[float]]:
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(SystemConfig.EMBEDDING_DEVICE)

        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = torch.mean(outputs.last_hidden_state, dim=1).cpu().numpy()

        return (embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)).tolist()

    def ingest_documents(self, file_paths: List[str]):
        chunker = MedicalDocumentChunker()

        for path in file_paths:
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read()

                doc_id = self.preprocessor.get_doc_id(path)
                clean_content = self.preprocessor.clean_text(content)

                document = {"doc_id": doc_id, "content": clean_content}
                chunks = chunker.chunk_document(document)

                texts = [c["text"] for c in chunks]
                embeddings = self._embed_texts(texts)

                entities = [
                    {"doc_id": c["doc_id"],
                     "chunk_id": c["chunk_id"],
                     "text": c["text"],
                     "embedding": emb}
                    for c, emb in zip(chunks, embeddings)
                ]

                batch_size = 100
                for i in range(0, len(entities), batch_size):
                    batch = entities[i:i + batch_size]
                    self.collection.insert([
                        [e["doc_id"] for e in batch],
                        [e["chunk_id"] for e in batch],
                        [e["text"] for e in batch],
                        [e["embedding"] for e in batch]
                    ])

            except Exception as e:
                st.error(f"文件处理失败: {Path(path).name} - {str(e)}")

        self.collection.flush()
        st.success(f"成功加载 {len(file_paths)} 个文档")


# ======================
# 主系统 (完整修改版)
# ======================
class MedicalRAGSystem:
    def __init__(self):
        self.preprocessor = MedicalDataPreprocessor()
        self.silicon_client = SiliconFlowClient()
        try:
            self.retriever = MedicalRetriever(self.preprocessor)
            self.disease_extractor = DiseaseExtractor()
            self.bm25 = None
        except Exception as e:
            st.error(f"检索系统初始化失败: {str(e)}")
            raise
        self.dialog_history = []
        self.max_history = 5

    def _build_context_prompt(self, question: str, context: List[str]) -> str:
        system_prompt = """你是一名专业医疗助手，请基于以下医学资料和对话历史回答问题：
{}

当前对话历史：
{}

回答要求：
1. 核心定义简明扼要
2. 关键特征分点列出（最多3点）
3. 必须标注数据来源
4. 使用中文口语化表达
5. 如问题涉及之前讨论内容，请保持一致性""".format(
            '\n'.join([f"[来源:{i + 1}] {text}" for i, text in enumerate(context)]),
            '\n'.join([f"Q: {h['question']}\nA: {h['answer']}" for h in self.dialog_history[-3:]])
        )
        return system_prompt

    def _enhance_query(self, question: str) -> str:
        keywords = self.disease_extractor.extract(question)
        if not keywords:
            return question

        enhanced = f"{question} {' '.join(keywords)}"

        for kw in keywords:
            enhanced += " " + " ".join(self.disease_extractor.medical_terms["synonyms"].get(kw, []))

        return enhanced

    def _retrieve_with_keywords(self, query: str, keywords: List[str]) -> List[dict]:
        return self.retriever.retrieve(query)

    def _extract_relevant_snippets(self, question: str, documents: List[dict]) -> List[dict]:
        all_sentences = []
        doc_reference = []
        for doc in documents:
            sentences = [s.strip() for s in re.split(r'(?<=[。！？；\n])', doc["text"]) if s.strip()]
            all_sentences.extend(sentences)
            doc_reference.extend([doc["doc_id"]] * len(sentences))

        tokenized_sentences = [list(jieba.cut(s)) for s in all_sentences]
        self.bm25 = BM25Okapi(tokenized_sentences)

        tokenized_question = list(jieba.cut(question))
        top_n = min(5, len(all_sentences))
        scores = self.bm25.get_scores(tokenized_question)
        top_indices = np.argsort(scores)[-top_n:][::-1]

        snippets = []
        for idx in top_indices:
            if scores[idx] > 0:
                snippets.append({
                    "text": all_sentences[idx].strip(),
                    "doc_id": doc_reference[idx],
                    "score": float(scores[idx]),
                    "full_doc": documents[next(i for i, d in enumerate(documents) if d["doc_id"] == doc_reference[idx])]
                })
        return snippets

    def _generate_concise_answer(self, question: str, snippets: List[dict]) -> str:
        if not snippets:
            return "未找到相关信息"

        context = [s["text"] for s in snippets[:3]]

        api_response = self.silicon_client.generate_answer(
            question=question,
            context=self._build_context_prompt(question, context)
        )

        if "choices" in api_response:
            full_answer = api_response["choices"][0]["message"]["content"]
            source_list = list({s["doc_id"] for s in snippets[:3]})
            return f"{full_answer}\n\n数据来源：{', '.join(source_list)}"

        return "天问模型生成回答时发生错误"

    def _generate_exam_suggestions(self, diseases: List[str]) -> str:
        exam_map = {
            "糖尿病": ["血糖检测", "糖化血红蛋白"],
            "冠心病": ["心电图", "冠脉造影"],
            "肺炎": ["胸部X光", "血常规"],
            "高血压": ["血压监测", "血脂检查"],
            "肝炎": ["肝功能检查", "肝炎病毒检测"],
            "肾炎": ["尿常规", "肾功能检查"]
        }

        suggestions = set()
        for d in diseases:
            suggestions.update(exam_map.get(d, []))

        return "、".join(suggestions) if suggestions else "暂无具体建议"

    def _generate_full_document_response(self, question: str, snippets: List[dict]) -> str:
        if not snippets:
            return "未找到相关医学资料"

        response = f"## 问题: {question}\n\n"
        response += "以下是与您问题最相关的完整文档内容:\n\n"

        unique_docs = {}
        for snippet in snippets:
            doc_id = snippet["doc_id"]
            if doc_id not in unique_docs:
                unique_docs[doc_id] = {
                    "text": snippet["full_doc"]["text"],
                    "score": snippet["score"],
                    "highlighted": []
                }
            unique_docs[doc_id]["highlighted"].append(snippet["text"])

        sorted_docs = sorted(unique_docs.items(), key=lambda x: x[1]["score"], reverse=True)

        for i, (doc_id, doc_data) in enumerate(sorted_docs[:3], 1):
            response += f"### 文档 {i}: {doc_id} (相关性: {doc_data['score']:.2f})\n\n"

            full_text = doc_data["text"]
            for highlight in doc_data["highlighted"]:
                full_text = full_text.replace(highlight, f"**{highlight}**")

            response += f"{full_text[:1000]}{'...' if len(full_text) > 1000 else ''}\n\n"

        return response

    def query(self, question: str, full_docs: bool = False) -> dict:
        try:
            is_follow_up = len(self.dialog_history) > 0 and "?" in question
            keywords = self.disease_extractor.extract(question)

            if is_follow_up:
                last_question = self.dialog_history[-1]["question"]
                enhanced_query = f"{last_question} {question} {' '.join(keywords)}"
            else:
                enhanced_query = self._enhance_query(question)

            retrieved_docs = self._retrieve_with_keywords(enhanced_query, keywords)

            if not retrieved_docs:
                return {
                    "answer": "未找到相关医学资料",
                    "sources": []
                }

            snippets = self._extract_relevant_snippets(question, retrieved_docs)

            if full_docs:
                answer = self._generate_full_document_response(question, snippets)
            else:
                answer = self._generate_concise_answer(question, snippets)

            self.dialog_history.append({
                "question": question,
                "answer": answer,
                "timestamp": datetime.now().isoformat(),
                "mode": "full" if full_docs else "concise"
            })
            if len(self.dialog_history) > self.max_history:
                self.dialog_history.pop(0)

            return {
                "answer": answer,
                "sources": [{
                    "doc_id": s["doc_id"],
                    "text": s["text"][:150] + "...",
                    "score": s["score"]
                } for s in snippets[:3]],
                "mode": "full" if full_docs else "concise"
            }

        except Exception as e:
            return {
                "answer": f"系统处理出错: {str(e)}",
                "sources": [],
                "mode": "error"
            }

    def clear_history(self):
        self.dialog_history = []


# ======================
# 模型加载
# ======================
@st.cache_resource
def load_models():
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            SystemConfig.EMBEDDING_MODEL,
            cache_dir="./models"
        )
        model = AutoModel.from_pretrained(
            SystemConfig.EMBEDDING_MODEL,
            device_map=SystemConfig.EMBEDDING_DEVICE,
            cache_dir="./models"
        )
    except Exception as e:
        st.error(f"嵌入模型加载失败: {e}")
        return None, None, None

    try:
        cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    except Exception as e:
        st.error(f"重排序模型加载失败: {e}")
        return tokenizer, model, None

    return tokenizer, model, cross_encoder


# ======================
# Streamlit界面 (完整修改版)
# ======================
def main():
    st.set_page_config(
        page_title="医疗智能问答系统",
        page_icon="🏥",
        layout="wide"
    )

    st.title("🏥 医疗智能问答系统")

    if "rag_system" not in st.session_state:
        with st.spinner("系统初始化中..."):
            try:
                st.session_state.rag_system = MedicalRAGSystem()
                st.success("系统就绪！")
            except Exception as e:
                st.error(f"初始化失败: {str(e)}")
                return

    with st.sidebar:
        st.header("⚙️ 系统设置")

        enable_rerank = st.checkbox(
            "启用高级重排序",
            value=True,
            help="启用多阶段重排序优化搜索结果质量"
        )
        SystemConfig.RERANK_CONFIG["enable_rerank"] = enable_rerank

        if enable_rerank:
            st.subheader("重排序权重配置")
            SystemConfig.RERANK_CONFIG["score_weights"]["vector"] = st.slider(
                "向量相似度权重", 0.1, 0.8, 0.5, 0.05
            )
            SystemConfig.RERANK_CONFIG["score_weights"]["keyword"] = st.slider(
                "关键词匹配权重", 0.1, 0.8, 0.3, 0.05
            )
            SystemConfig.RERANK_CONFIG["score_weights"]["rerank"] = st.slider(
                "语义相关权重", 0.1, 0.8, 0.2, 0.05
            )

        st.header("📂 文档管理")
        data_dir = st.text_input("文档目录路径", value=file_path)
        if st.button("加载文档", help="从指定目录加载医疗文档"):
            if Path(data_dir).exists():
                files = []
                for ext in ("*.txt", "*.md", "*.html"):
                    files.extend(list(Path(data_dir).rglob(ext)))

                if files:
                    with st.spinner(f"正在加载 {len(files)} 个文档..."):
                        st.session_state.rag_system.retriever.ingest_documents([str(f) for f in files])
                    st.success("文档加载完成！")
                else:
                    st.warning("未找到支持格式的文档")
            else:
                st.error("目录不存在！")

        st.header("🗨️ 对话历史")
        if st.button("清空对话历史", help="清除所有对话记录"):
            st.session_state.rag_system.clear_history()
            st.success("对话历史已清空")

        st.header("ℹ️ 系统信息")
        st.markdown(f"""
        - 当前对话数: {len(st.session_state.rag_system.dialog_history)}
        - 最大历史记录: {st.session_state.rag_system.max_history}
        - 最后更新时间: {datetime.now().strftime("%Y-%m-%d %H:%M")}
        """)

    col1, col2 = st.columns([3, 1])

    with col1:
        if hasattr(st.session_state.rag_system, 'dialog_history') and st.session_state.rag_system.dialog_history:
            st.subheader("📜 当前对话历史")
            for i, dialog in enumerate(st.session_state.rag_system.dialog_history):
                with st.expander(f"对话{i + 1}: {dialog['question'][:30]}...",
                                 expanded=i == len(st.session_state.rag_system.dialog_history) - 1):
                    st.markdown(f"""
                    **问题**: {dialog['question']}  
                    **回答**: {dialog['answer']}  
                    *模式: {dialog['mode']} | 时间: {dialog['timestamp'][11:19]}*
                    """)
        else:
            st.info("💡 暂无对话历史，请输入问题开始对话")

        st.subheader("💬 请输入医疗问题")
        question = st.text_area(
            "输入框",
            height=120,
            label_visibility="collapsed",
            placeholder="例如：糖尿病的症状有哪些？如何治疗？"
        )

        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            if st.button("🤖 智能生成回答", help="从文档提取关键信息生成简洁回答", use_container_width=True):
                if question.strip():
                    with st.spinner("正在提取关键信息并生成回答..."):
                        result = st.session_state.rag_system.query(question, full_docs=False)

                        st.subheader("🩺 专业回答")
                        st.markdown(result["answer"])

                        if result["sources"]:
                            with st.expander("🔍 参考片段（点击展开）"):
                                for i, src in enumerate(result["sources"], 1):
                                    st.markdown(f"""
                                    **片段{i}** (来自文档 `{src['doc_id']}`)
                                    > {src['text']}
                                    """)
                    st.rerun()
                else:
                    st.warning("请输入问题内容")

        with btn_col2:
            if st.button("📄 显示完整文档", help="显示原始文档内容", use_container_width=True):
                if question.strip():
                    with st.spinner("正在检索相关文档..."):
                        result = st.session_state.rag_system.query(question, full_docs=True)

                        st.subheader("📚 相关文档")
                        for i, doc in enumerate(result["sources"], 1):
                            with st.expander(f"文档{i}: {doc['doc_id']} (相关性: {doc['score']:.2f})"):
                                st.markdown(f"""
                                **内容**:  
                                {doc['text']}  
                                """)
                    st.rerun()
                else:
                    st.warning("请输入问题内容")

    with col2:
        st.subheader("🚀 快捷追问")
        if hasattr(st.session_state.rag_system, 'dialog_history') and st.session_state.rag_system.dialog_history:
            last_question = st.session_state.rag_system.dialog_history[-1]["question"]
            suggestions = [
                f"关于'{last_question.split('?')[0]}'，能否详细解释一下？",
                "这个诊断需要做哪些检查？",
                "有哪些治疗方法？",
                "这个病的预防措施是什么？",
                "这些症状严重吗？",
                "需要看什么科室？"
            ]

            for i, suggestion in enumerate(suggestions):
                if st.button(suggestion, key=f"suggestion_{i}", use_container_width=True):
                    st.session_state.question_input = suggestion
                    st.rerun()
        else:
            st.info("输入第一个问题后，这里会显示追问建议")

        st.subheader("🏷️ 相关疾病")
        if hasattr(st.session_state.rag_system, 'dialog_history') and st.session_state.rag_system.dialog_history:
            last_question = st.session_state.rag_system.dialog_history[-1]["question"]
            keywords = st.session_state.rag_system.disease_extractor.extract(last_question)
            if keywords:
                st.markdown("""
                <style>
                .tag {
                    display: inline-block;
                    background-color: #e6f2ff;
                    border-radius: 16px;
                    padding: 4px 12px;
                    margin: 4px;
                    font-size: 0.9em;
                }
                </style>
                """, unsafe_allow_html=True)

                cols = st.columns(2)
                for i, kw in enumerate(keywords):
                    with cols[i % 2]:
                        st.markdown(f'<span class="tag">{kw}</span>', unsafe_allow_html=True)
            else:
                st.info("未识别到疾病关键词")


if __name__ == "__main__":
    main()
