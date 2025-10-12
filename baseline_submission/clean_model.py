import os, re, json, gc, math, pickle
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

# ---------- utils ----------
def clear_cache():
    gc.collect()
    if torch.backends.mps.is_available(): torch.mps.empty_cache()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

def light_clean(t: str) -> str:
    t = t.replace("״", "\"").replace("׳", "'").replace("’", "'").replace("“", "\"").replace("”", "\"")
    t = re.sub(r"[!?.,:;()\[\]{}]", " ", t)
    return re.sub(r"\s+", " ", t).strip()

# ---------- data ----------
def unwrap_paragraphs(d):
    items = sorted(d.items(), key=lambda kv: int(kv[0].split("_")[-1]))
    return [v["uuid"] for _, v in items], [v["passage"] for _, v in items]

def unwrap_targets(d):
    items = sorted(d.items(), key=lambda kv: int(kv[0].split("_")[-1]))
    return [v for _, v in items]

def load_hsrc(path):
    data = [json.loads(x) for x in open(path, "r", encoding="utf-8")]
    df = pd.DataFrame(data)
    df["paragraph_uuids"], df["paragraph_texts"] = zip(*df["paragraphs"].map(unwrap_paragraphs))
    df["relevances"] = df["target_actions"].map(unwrap_targets)
    rows = []
    for _, r in df.iterrows():
        for pid, ptxt, rel in zip(r["paragraph_uuids"], r["paragraph_texts"], r["relevances"]):
            rows.append({
                "query_uuid": r["query_uuid"],
                "query": r["query"],
                "paragraph_uuid": pid,
                "paragraph_text": ptxt,
                "relevance_num": int(rel) if rel is not None else 0
            })
    return pd.DataFrame(rows)

def build_corpus(df):
    corpus = df[["paragraph_uuid", "paragraph_text"]].drop_duplicates("paragraph_uuid").copy()
    corpus["paragraph_text"] = corpus["paragraph_text"].map(light_clean)
    return corpus.reset_index(drop=True)

def build_relevant_dict(df):
    return {
        qid: set(df.loc[(df["query_uuid"] == qid) & (df["relevance_num"] > 0), "paragraph_uuid"])
        for qid in df["query_uuid"].unique()
    }

# ---------- models ----------
class E5Retriever:
    def __init__(self, model_name="intfloat/multilingual-e5-base", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def embed_texts(self, texts, is_query=False, batch_size=32):
        prefix = "query: " if is_query else "passage: "
        texts = [prefix + t.strip() for t in texts]
        all_embs = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                enc = self.tokenizer(texts[i:i+batch_size], padding=True, truncation=True, max_length=512, return_tensors="pt").to(self.device)
                out = self.model(**enc)
                att = enc["attention_mask"]
                emb = out.last_hidden_state
                mask = att.unsqueeze(-1).expand(emb.size()).float()
                sum_emb = torch.sum(emb * mask, 1)
                sum_mask = torch.clamp(mask.sum(1), min=1e-9)
                emb = sum_emb / sum_mask
                emb = torch.nn.functional.normalize(emb, p=2, dim=1)
                all_embs.append(emb.cpu())
        return torch.cat(all_embs, dim=0).numpy()

class HebrewBERT:
    def __init__(self, model_name="avichr/heBERT", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def embed_texts(self, texts, batch_size=32, is_query=False):
        all_embs = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                enc = self.tokenizer(texts[i:i+batch_size], padding=True, truncation=True, max_length=512, return_tensors="pt").to(self.device)
                out = self.model(**enc)
                att = enc["attention_mask"]
                emb = out.last_hidden_state
                mask = att.unsqueeze(-1).expand(emb.size()).float()
                sum_emb = torch.sum(emb * mask, 1)
                sum_mask = torch.clamp(mask.sum(1), min=1e-9)
                emb = sum_emb / sum_mask
                emb = torch.nn.functional.normalize(emb, p=2, dim=1)
                all_embs.append(emb.cpu())
        return torch.cat(all_embs, dim=0).numpy()

class AlephBERT:
    def __init__(self, model_name="onlplab/alephbert-base", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def embed_texts(self, texts, batch_size=32, is_query=False):
        all_embs = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                enc = self.tokenizer(texts[i:i+batch_size], padding=True, truncation=True, max_length=512, return_tensors="pt").to(self.device)
                out = self.model(**enc)
                att = enc["attention_mask"]
                emb = out.last_hidden_state
                mask = att.unsqueeze(-1).expand(emb.size()).float()
                sum_emb = torch.sum(emb * mask, 1)
                sum_mask = torch.clamp(mask.sum(1), min=1e-9)
                emb = sum_emb / sum_mask
                emb = torch.nn.functional.normalize(emb, p=2, dim=1)
                all_embs.append(emb.cpu())
        return torch.cat(all_embs, dim=0).numpy()

# ---------- metrics ----------
def recall_at_k(pred_ids, relevant_ids, k=100):
    if not relevant_ids: return np.nan
    hits = sum(1 for pid in pred_ids[:k] if pid in relevant_ids)
    return hits / len(relevant_ids)

def embed_texts(model_name, texts, device="mps", batch_size=32):
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModel.from_pretrained(model_name).to(device)
    mdl.eval()
    all_embs = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            enc = tok(texts[i:i+batch_size], truncation=True, padding=True, max_length=512, return_tensors="pt").to(device)
            out = mdl(**enc)
            att = enc['attention_mask']
            emb = (out.last_hidden_state * att.unsqueeze(-1)).sum(1) / att.sum(1, keepdim=True)
            emb = torch.nn.functional.normalize(emb, p=2, dim=1)
            all_embs.append(emb.cpu().numpy())
    return np.vstack(all_embs)

def evaluate_model(model_name, emb_dict, relevant_dict, df_long, k=100, device="mps"):
    print(f"\nRunning {model_name} ...")
    corpus = df_long[['paragraph_uuid','paragraph_text']].drop_duplicates('paragraph_uuid')
    pid = corpus['paragraph_uuid'].tolist()
    P = np.array([emb_dict[p] for p in pid])

    queries = df_long[['query_uuid','query']].drop_duplicates('query_uuid')
    q_texts = queries['query'].tolist()
    qids = queries['query_uuid'].tolist()

    Q = embed_texts(model_name, q_texts, device=device)
    recalls = []
    for q_emb, qid in zip(Q, qids):
        sims = cosine_similarity(q_emb.reshape(1,-1), P).ravel()
        top_idx = np.argsort(-sims)[:k]
        pred_ids = [pid[i] for i in top_idx]
        rec = recall_at_k(pred_ids, relevant_dict[qid], k)
        recalls.append(rec)
    mean_recall = np.nanmean(recalls)
    print(f"Recall@{k}: {mean_recall:.4f}")
    return mean_recall

##
# ---------- run ----------
if __name__ == "__main__":
    file_path = "/Users/gilikurtser/Desktop/Dataset and Baseline/hsrc/hsrc_train.jsonl"
    data = [json.loads(x) for x in open(file_path, "r", encoding="utf-8")]
    df = pd.DataFrame(data)
    df['paragraph_uuids'], df['paragraph_texts'] = zip(*df['paragraphs'].map(lambda d: ([v['uuid'] for v in d.values()], [v['passage'] for v in d.values()])))
    df['relevances'] = df['target_actions'].map(lambda d: [v for v in d.values()])
    rows = []
    for _, r in df.iterrows():
        for pid, ptxt, rel in zip(r['paragraph_uuids'], r['paragraph_texts'], r['relevances']):
            rows.append({"query_uuid": r['query_uuid'], "query": r['query'], "paragraph_uuid": pid, "paragraph_text": ptxt, "relevance_num": rel})
    df_long = pd.DataFrame(rows)

    with open("paragraph_embeddings_e5.pkl","rb") as f: emb_dict_e5 = pickle.load(f)
    with open("paragraph_embeddings_hebert.pkl","rb") as f: emb_dict_hebert = pickle.load(f)
    with open("paragraph_embeddings_alephbert.pkl","rb") as f: emb_dict_alephbert = pickle.load(f)
    with open("relevant_dict.pkl","rb") as f: relevant_dict = pickle.load(f)

    r_e5 = evaluate_model("intfloat/multilingual-e5-base", emb_dict_e5, relevant_dict, df_long)
    r_he = evaluate_model("avichr/heBERT", emb_dict_hebert, relevant_dict, df_long)
    r_al = evaluate_model("onlplab/alephbert-base", emb_dict_alephbert, relevant_dict, df_long)

    print("\n=== Summary ===")
    print(f"E5 Recall@100:      {r_e5:.4f}")
    print(f"heBERT Recall@100:  {r_he:.4f}")
    print(f"AlephBERT Recall@100: {r_al:.4f}")