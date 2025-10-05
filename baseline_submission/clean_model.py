import os, re, json, gc, math
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from scipy.stats import spearmanr

# ---------- utils ----------
def clear_cache():
    gc.collect()
    if torch.backends.mps.is_available(): torch.mps.empty_cache()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

def light_clean(t:str)->str:
    t = t.replace("״","\"").replace("׳","'").replace("’","'").replace("“","\"").replace("”","\"")
    t = re.sub(r'[!?.,:;()\[\]{}]', ' ', t)
    return re.sub(r'\s+',' ', t).strip()

# ---------- data ----------
def unwrap_paragraphs(d):
    items = sorted(d.items(), key=lambda kv:int(kv[0].split('_')[-1]))
    return [v['uuid'] for _,v in items], [v['passage'] for _,v in items]

def unwrap_targets(d):
    items = sorted(d.items(), key=lambda kv:int(kv[0].split('_')[-1]))
    return [v for _,v in items]

def load_hsrc(path):
    data = [json.loads(x) for x in open(path, "r", encoding="utf-8")]
    df = pd.DataFrame(data)
    df['paragraph_uuids'], df['paragraph_texts'] = zip(*df['paragraphs'].map(unwrap_paragraphs))
    df['relevances'] = df['target_actions'].map(unwrap_targets)
    rows = []
    for _, r in df.iterrows():
        for pos, (pid, ptxt, rel) in enumerate(zip(r['paragraph_uuids'], r['paragraph_texts'], r['relevances'])):
            rows.append({
                "query_uuid": r['query_uuid'],
                "query": r['query'],
                "paragraph_uuid": pid,
                "paragraph_text": ptxt,
                "position": pos,
                "relevance_num": int(rel) if rel is not None else 0
            })
    return pd.DataFrame(rows)

def build_corpus(df):
    corpus = df[['paragraph_uuid','paragraph_text']].drop_duplicates('paragraph_uuid').copy()
    corpus['paragraph_text'] = corpus['paragraph_text'].map(light_clean)
    return corpus.reset_index(drop=True)

def build_relevant_dict(df):
    return {
        qid: set(df.loc[(df["query_uuid"]==qid) & (df["relevance_num"]>0), "paragraph_uuid"])
        for qid in df["query_uuid"].unique()
    }

# ---------- dense embeddings ----------
def load_dense(model_name):
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModel.from_pretrained(model_name).to(device)
    mdl.eval()
    return tok, mdl, device

def _mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts

def encode_texts(texts, tok, mdl, device, bs=32, max_len=512):
    embs = []
    with torch.no_grad():
        for i in range(0, len(texts), bs):
            enc = tok(texts[i:i+bs], padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(device)
            out = mdl(**enc)
            pooled = _mean_pool(out.last_hidden_state, enc['attention_mask'])
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            embs.append(pooled.cpu().numpy())
            del enc, out, pooled
    return np.vstack(embs) if embs else np.zeros((0,768), dtype=np.float32)

def dense_scores_for_query(q_text, P, tok, mdl, device):
    q = encode_texts([q_text], tok, mdl, device)          # (1,d)
    return (q @ P.T).ravel()                              # cosine if L2-normalized

# ---------- metrics ----------
def recall_at_k(pred_ids, relevant_ids, k=20):
    if len(relevant_ids)==0: return np.nan
    hits = sum(1 for pid in pred_ids[:k] if pid in relevant_ids)
    return hits/len(relevant_ids)

def safe_spearman(y_true, y_pred):
    if len(set(y_true))<=1: return np.nan
    return float(spearmanr(y_true, y_pred).correlation)

def analyze_query(scores, pid, relevant_ids, k=20):
    order = np.argsort(-scores)
    top_idx = order[:k]
    top_ids = [pid[i] for i in top_idx]
    rels_all = [1 if p in relevant_ids else 0 for p in pid]
    rels_top = [1 if p in relevant_ids else 0 for p in top_ids]

    # 1. Spearman מול כל הקורפוס
    sp_all = safe_spearman(rels_all, scores)

    # 2. Spearman רק ב-Top-k
    sp_top = safe_spearman(rels_top, scores[top_idx])

    # 3. Spearman בתוך ה-Top-k – רלוונטיים מול לא
    pos_scores = [s for p,s in zip(top_ids, scores[top_idx]) if p in relevant_ids]
    neg_scores = [s for p,s in zip(top_ids, scores[top_idx]) if p not in relevant_ids]
    if len(pos_scores)>1 and len(neg_scores)>1:
        sp_posneg = safe_spearman([1]*len(pos_scores)+[0]*len(neg_scores),
                                  pos_scores+neg_scores)
    else:
        sp_posneg = np.nan

    rec = recall_at_k(top_ids, relevant_ids, k=k)
    return sp_all, sp_top, sp_posneg, rec

# ---------- run ----------
def run_full(df_long, corpus, relevant_dict, model_name, k=20):
    print(f"\n=== Running {model_name} ===")
    clear_cache()
    tok, mdl, device = load_dense(model_name)

    P = encode_texts(corpus['paragraph_text'].tolist(), tok, mdl, device)
    pid = corpus['paragraph_uuid'].tolist()

    results = []
    for qid, g in df_long.groupby("query_uuid"):
        q = light_clean(g['query'].iloc[0])
        scores = dense_scores_for_query(q, P, tok, mdl, device)
        sp_all, sp_top, sp_posneg, rec = analyze_query(scores, pid, relevant_dict[qid], k=k)
        results.append({"query_uuid":qid,"sp_all":sp_all,"sp_top":sp_top,
                        "sp_posneg":sp_posneg,"recall@20":rec})

    return pd.DataFrame(results)

# ---------- example ----------
if __name__=="__main__":
    file_path = "/Users/gilikurtser/Desktop/Dataset and Baseline/hsrc/hsrc_train.jsonl"
    df_long = load_hsrc(file_path)
    corpus = build_corpus(df_long)
    relevant_dict = build_relevant_dict(df_long)

    model = "avichr/heBERT"  # אפשר לשים Dicta או מודל אחר
    df_res = run_full(df_long, corpus, relevant_dict, model, k=20)

    df_res.to_csv("experiment_spearman_recall_dicta.csv", index=False)
    df_res.to_json("experiment_spearman_recall_dicta.json", orient="records", force_ascii=False, indent=2)
    print(df_res.head())
    print(df_res.mean(numeric_only=True))
