import json
import torch
import numpy as np
import os
import re
import unicodedata
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
from joblib import Parallel, delayed

from normalization import normalize_text, load_hebrew_nlp


# ================================================================
#  Unified text normalization + tokenization for Hebrew BM25
# ================================================================

def normalize_for_bm25(text: str) -> str:
    """
    ריכוך קל לניקוי תווים חריגים מבלי לשנות משמעות מילולית.
    מסיר גרשיים, מאחד צורות כמו נאט״ו/NATO, מתקן מקפים.
    """
    text = unicodedata.normalize("NFKC", text)
    # unify quotation marks
    text = re.sub(r"[\"׳״'’‘]", "", text)
    # unify known foreign words
    text = re.sub(r"נאט.?ו", "נאטו", text, flags=re.IGNORECASE)
    text = re.sub(r"\bN\.?A\.?T\.?O\b", "NATO", text, flags=re.IGNORECASE)
    # replace long dashes and weird hyphens
    text = re.sub(r"[‐–—-]", " ", text)
    # remove control chars
    text = re.sub(r"[\u200f\u200e]", "", text)
    return text


def simple_tokenize(text: str):
    """
    Tokenizer כללי לעברית ואנגלית.
    מפריד מילים לפי אותיות ומספרים בלבד.
    """
    text = normalize_for_bm25(text)
    # החלפת פיסוק לרווח
    text = re.sub(r"[.,!?;:\-–—()\[\]{}<>•·\"']", " ", text)
    # חילוץ טוקנים: אותיות עבריות, לועזיות או ספרות
    tokens = re.findall(r"[א-ת]+|[A-Za-z]+|\d+", text)
    return tokens


# ================================================================
#  E5 Retriever
# ================================================================
class E5Retriever:
    def __init__(self, model_name=None, device=None):
        if model_name is None:
            local_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'multilingual-e5-base')
            if os.path.isdir(local_model_path):
                model_name = local_model_path
                print(f"Using local E5 model from: {model_name}")
        
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"Loading {model_name} model on device: {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, dtype=torch.float16).to(self.device)
        self.model.eval()

        self.corpus_ids = []
        self.corpus_embeddings = None

    def embed_texts(self, texts, is_query=False, batch_size=32): 
        if is_query:
            prefixed_texts = [f"query: {t.strip()}" for t in texts]
        else:
            prefixed_texts = [f"passage: {t.strip()}" for t in texts]

        all_embeddings = []
        total_batches = (len(prefixed_texts) + batch_size - 1) // batch_size
        
        for i in range(0, len(prefixed_texts), batch_size):
            batch_texts = prefixed_texts[i:i + batch_size]
            encoded = self.tokenizer(batch_texts, padding=True, truncation=True, max_length=512, return_tensors='pt').to(self.device)
            with torch.no_grad():
                model_output = self.model(**encoded)
                attn = encoded['attention_mask']
                embeddings = model_output.last_hidden_state
                mask = attn.unsqueeze(-1).expand(embeddings.size()).float()
                summed = torch.sum(embeddings * mask, 1)
                summed_mask = torch.clamp(mask.sum(1), min=1e-9)
                embeddings = summed / summed_mask
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            all_embeddings.append(embeddings.cpu())
            del encoded, model_output
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return torch.cat(all_embeddings, dim=0).numpy()


# ================================================================
#  BGE Reranker
# ================================================================
class BGEReranker:
    def __init__(self, model_name=None, device=None):
        if model_name is None:
            local_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'bge-reranker-v2-m3')
            if os.path.isdir(local_model_path):
                model_name = local_model_path
                print(f"Using local BGE model from: {model_name}")
        
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device
        
        print(f"Loading {model_name} reranker on device: {self.device}")
        from transformers import AutoModelForSequenceClassification
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, dtype=torch.float16, trust_remote_code=True).to(self.device)
        self.model.eval()

    def rerank(self, query_text, passages, passage_ids, top_k=20):
        if not passages:
            return []
        scores = []
        batch_size = 4
        for i in range(0, len(passages), batch_size):
            batch_passages = passages[i:i + batch_size]
            batch_queries = [query_text] * len(batch_passages)
            with torch.no_grad():
                inputs = self.tokenizer(batch_queries, batch_passages, padding=True, truncation=True, max_length=512, return_tensors='pt').to(self.device)
                outputs = self.model(**inputs)
                logits = outputs.logits
                if logits.ndim == 1:
                    batch_scores = logits.cpu().numpy()
                elif logits.shape[1] == 1:
                    batch_scores = logits.squeeze(-1).cpu().numpy()
                else:
                    batch_scores = logits[:, 1].cpu().numpy()
            scores.extend(batch_scores.tolist())
        results = sorted(zip(passage_ids, scores), key=lambda x: x[1], reverse=True)
        return results[:top_k]


# ================================================================
#  Pipeline: Preprocess + Predict
# ================================================================
retriever = None
reranker = None
corpus_texts = {}

def preprocess(
    corpus_dict,
    use_normalization=True,
    use_bm25=True,
    bm25_alpha=0.75,
):
    global retriever, reranker, corpus_texts
    print("=" * 60)
    print("PREPROCESSING: Initializing E5 + BM25 + BGE Reranker Pipeline...")
    print("=" * 60)
    retriever = E5Retriever()
    reranker = BGEReranker()

    retriever.corpus_ids = list(corpus_dict.keys())
    passages = [doc.get('passage', doc.get('text', '')) for doc in corpus_dict.values()]
    corpus_texts = {doc_id: passages[i] for i, doc_id in enumerate(retriever.corpus_ids)}

    normalized_passages = passages
    normalization_pipeline = None
    if use_normalization and passages:
        try:
            normalization_pipeline = load_hebrew_nlp()
            print("Applying Hebrew normalization to passages before embedding...")
            normalized_passages = Parallel(n_jobs=-1, prefer='threads')(
                delayed(normalize_text)(p, nlp=normalization_pipeline) for p in passages
            )
        except ImportError:
            print("hebspacy not available; using raw passages for embeddings.")
            normalization_pipeline = None

    print("Computing E5 embeddings...")
    retriever.corpus_embeddings = retriever.embed_texts(normalized_passages, is_query=False, batch_size=32)

    bm25 = None
    if use_bm25:
        print("Building BM25 index...")
        tokenized_corpus = [simple_tokenize(passage) for passage in normalized_passages]
        bm25 = BM25Okapi(tokenized_corpus)
        print("✓ BM25 index built successfully.")

    print("✓ Corpus preprocessing complete!")
    print(f"✓ Generated embeddings for {len(retriever.corpus_ids)} documents")
    print(f"✓ Embedding matrix shape: {retriever.corpus_embeddings.shape}")

    return {
        'retriever': retriever,
        'reranker': reranker,
        'corpus_ids': retriever.corpus_ids,
        'corpus_embeddings': retriever.corpus_embeddings,
        'corpus_texts': corpus_texts,
        'bm25': bm25,
        'bm25_alpha': bm25_alpha,
        'use_bm25': use_bm25,
        'use_normalization': use_normalization,
        'normalization_pipeline': normalization_pipeline,
        'num_documents': len(corpus_dict)
    }


def predict(
    query,
    preprocessed_data,
    retrieval_top_k=100,
    bm25_expansion_k=0,
    rerank_top_k=20,
):
    global retriever, reranker, corpus_texts

    query_text = query.get('query', '')
    if not query_text:
        return []

    retriever = retriever or preprocessed_data.get('retriever')
    reranker = reranker or preprocessed_data.get('reranker')
    corpus_texts = corpus_texts or preprocessed_data.get('corpus_texts', {})
    bm25 = preprocessed_data.get('bm25')
    bm25_alpha = preprocessed_data.get('bm25_alpha', 0.75)
    use_bm25 = preprocessed_data.get('use_bm25', bm25 is not None)
    normalization_pipeline = preprocessed_data.get('normalization_pipeline')
    use_normalization = preprocessed_data.get('use_normalization', normalization_pipeline is not None)

    if retriever is None or reranker is None:
        print("Error: Missing retriever or reranker in preprocessed data")
        return []

    candidate_indices = []
    seen = set()

    normalized_query = query_text
    if use_normalization and normalization_pipeline is not None:
        try:
            normalized_query = normalize_text(query_text, nlp=normalization_pipeline)
        except Exception as norm_err:
            print(f"Query normalization failed: {norm_err}")
            normalized_query = query_text

    if retrieval_top_k:
        query_embedding = retriever.embed_texts([normalized_query], is_query=True, batch_size=1)
        e5_scores = cosine_similarity(query_embedding, retriever.corpus_embeddings)[0]
        k_emb = min(retrieval_top_k, len(e5_scores)) if retrieval_top_k > 0 else len(e5_scores)
        if k_emb >= len(e5_scores):
            emb_indices = np.argsort(e5_scores)[::-1]
        else:
            emb_indices = np.argpartition(e5_scores, -k_emb)[-k_emb:]
            emb_indices = emb_indices[np.argsort(e5_scores[emb_indices])[::-1]]
        for idx in emb_indices:
            if idx not in seen:
                candidate_indices.append(idx)
                seen.add(idx)

        if use_bm25 and bm25 is not None:
            if bm25_expansion_k:
                k_bm25 = min(bm25_expansion_k, len(retriever.corpus_ids))
            else:
                k_bm25 = 0
                if not candidate_indices:
                    fallback_k = retrieval_top_k or len(retriever.corpus_ids)
                    k_bm25 = min(fallback_k, len(retriever.corpus_ids))
            if k_bm25:
                tokens = simple_tokenize(normalized_query)
                bm25_scores = bm25.get_scores(tokens)
                if k_bm25 >= len(bm25_scores):
                    bm25_indices = np.argsort(bm25_scores)[::-1]
                else:
                    bm25_indices = np.argpartition(bm25_scores, -k_bm25)[-k_bm25:]
                    bm25_indices = bm25_indices[np.argsort(bm25_scores[bm25_indices])[::-1]]
                for idx in bm25_indices:
                    if idx not in seen:
                        candidate_indices.append(idx)
                        seen.add(idx)

    if not candidate_indices:
        print("Warning: no candidates gathered; falling back to embeddings-only retrieval.")
        query_embedding = retriever.embed_texts([normalized_query], is_query=True, batch_size=1)
        e5_scores = cosine_similarity(query_embedding, retriever.corpus_embeddings)[0]
        k_emb = min(retrieval_top_k or len(e5_scores), len(e5_scores))
        candidate_indices = np.argsort(e5_scores)[::-1][:k_emb].tolist()

    if rerank_top_k:
        candidate_indices = candidate_indices[: min(rerank_top_k, len(candidate_indices))]

    candidate_ids = [retriever.corpus_ids[idx] for idx in candidate_indices]
    candidate_passages = [corpus_texts.get(pid, '') for pid in candidate_ids]

    reranked_results = reranker.rerank(query_text, candidate_passages, candidate_ids, top_k=rerank_top_k or len(candidate_ids))
    return [{'paragraph_uuid': pid, 'score': float(score)} for pid, score in reranked_results]
