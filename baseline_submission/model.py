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

def preprocess(corpus_dict, use_normalization=True, use_bm25=True, bm25_alpha=0.75):
    global retriever, reranker, corpus_texts
    print("=" * 60)
    print("PREPROCESSING: Initializing E5 + BM25 + BGE Reranker Pipeline...")
    print("=" * 60)

    retriever = E5Retriever()
    reranker = BGEReranker()

    retriever.corpus_ids = list(corpus_dict.keys())
    passages = [doc.get('passage', doc.get('text', '')) for doc in corpus_dict.values()]
    corpus_texts = {doc_id: passages[i] for i, doc_id in enumerate(retriever.corpus_ids)}

    # Embeddings
    print("Computing E5 embeddings...")
    retriever.corpus_embeddings = retriever.embed_texts(passages, is_query=False, batch_size=32)

    # BM25
    bm25 = None
    if use_bm25:
        print("Building BM25 index with normalized tokenization...")
        tokenized_corpus = [simple_tokenize(p) for p in passages]
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
        'num_documents': len(corpus_dict)
    }


def predict(query, preprocessed_data):
    global retriever, reranker, corpus_texts
    query_text = query.get('query', '')
    if not query_text:
        return []

    retriever = retriever or preprocessed_data.get('retriever')
    reranker = reranker or preprocessed_data.get('reranker')
    corpus_texts = corpus_texts or preprocessed_data.get('corpus_texts', {})
    bm25 = preprocessed_data.get('bm25', None)

    if retriever is None or reranker is None or bm25 is None:
        print("Error: Missing retriever/reranker/bm25 in preprocessed data")
        return []

    print("=" * 80)
    print(f"🔍 QUERY: {query_text}")
    print("=" * 80)

    # Stage 1: E5 retrieval
    query_embedding = retriever.embed_texts([query_text], is_query=True, batch_size=1)
    e5_scores = cosine_similarity(query_embedding, retriever.corpus_embeddings)[0]
    e5_top_indices = np.argsort(e5_scores)[::-1][:50]
    e5_ids = [retriever.corpus_ids[i] for i in e5_top_indices]
    
    print("\n📘 e5 top passages:")
    for rank, idx in enumerate(e5_top_indices[:20], start=1):
        pid = retriever.corpus_ids[idx]
        snippet = corpus_texts[pid][:150].replace("\n", " ")
        score = e5_scores[idx]

        print(f"  {rank:2d}. [{pid}]  (score={score:.3f})  →  {snippet}...")
    # Stage 1.5: BM25 expansion
    query_tokens = simple_tokenize(query_text)
    print("BM25 query tokens:", query_tokens)
    bm25_scores = bm25.get_scores(query_tokens)
    bm25_top_indices = np.argsort(bm25_scores)[::-1][:40]
    bm25_ids = [retriever.corpus_ids[i] for i in bm25_top_indices]

 

    # Merge & rerank
    all_ids = list(dict.fromkeys(e5_ids + bm25_ids))
    all_passages = [corpus_texts[i] for i in all_ids]
    print(f"\nStage 2: BGE reranking on {len(all_passages)} candidates...")
    reranked_results = reranker.rerank(query_text, all_passages, all_ids, top_k=20)

    results = [{'paragraph_uuid': pid, 'score': float(score)} for pid, score in reranked_results]
    print(f"✓ Returned {len(results)} reranked results (BM25 used as expander)\n")
    return results
