# train_retriever.py
import torch, json, numpy as np, pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, AdamW
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ---------- metrics ----------
def recall_at_k(y_true, y_score, k=100):
    order = np.argsort(-np.array(y_score))[:k]
    hits = sum(np.array(y_true)[order] > 0)
    total = sum(np.array(y_true) > 0)
    return hits / total if total > 0 else 0.0

def ndcg_at_k(y_true, y_score, k=20):
    order = np.argsort(-np.array(y_score))[:k]
    rel_k = np.take(np.array(y_true), order)
    dcg = np.sum((2**rel_k - 1) / np.log2(np.arange(2, len(rel_k)+2)))
    idcg = np.sum((2**np.sort(y_true)[::-1][:k] - 1) / np.log2(np.arange(2, k+2)))
    return dcg / idcg if idcg > 0 else 0.0

# ---------- dataset ----------
class PairDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=250):
        self.df = df
        self.tok = tokenizer
        self.max_len = max_len
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        r = self.df.iloc[i]
        q, p, y = r["query"], r["paragraph_text"], int(r["relevance_num"] > 0)
        q_enc = self.tok(q, truncation=True, padding='max_length', max_length=self.max_len, return_tensors='pt')
        p_enc = self.tok(p, truncation=True, padding='max_length', max_length=self.max_len, return_tensors='pt')
        return q_enc, p_enc, torch.tensor(y, dtype=torch.float32)

# ---------- model ----------
class BiEncoder(torch.nn.Module):
    def __init__(self, name="onlplab/alephbert-base"):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(name)
    def forward(self, q_enc, p_enc):
        q = self.encoder(**{k:v.squeeze(0) for k,v in q_enc.items()}).last_hidden_state.mean(1)
        p = self.encoder(**{k:v.squeeze(0) for k,v in p_enc.items()}).last_hidden_state.mean(1)
        q = torch.nn.functional.normalize(q, p=2, dim=1)
        p = torch.nn.functional.normalize(p, p=2, dim=1)
        return (q @ p.T).diag()  # similarity for each pair

# ---------- train ----------
def train_model(df_long, model_name="onlplab/alephbert-base", epochs=1, lr=2e-5, batch_size=4):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_name)
    train_q, val_q = train_test_split(df_long['query_uuid'].unique(), test_size=0.1, random_state=42)
    df_train = df_long[df_long['query_uuid'].isin(train_q)]
    df_val   = df_long[df_long['query_uuid'].isin(val_q)]

    train_loader = DataLoader(PairDataset(df_train, tok), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(PairDataset(df_val, tok), batch_size=batch_size)

    model = BiEncoder(model_name).to(device)
    opt = AdamW(model.parameters(), lr=lr)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        model.train()
        losses = []
        for q, p, y in tqdm(train_loader):
            y = y.to(device)
            opt.zero_grad()
            scores = model(q, p)
            loss = loss_fn(scores, y)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        print(f"Epoch {epoch+1}: Train loss {np.mean(losses):.4f}")

    # ---------- evaluate ----------
    model.eval()
    recalls, ndcgs = [], []
    with torch.no_grad():
        for qid, g in df_val.groupby("query_uuid"):
            q = g["query"].iloc[0]
            p_texts = g["paragraph_text"].tolist()
            y_true  = g["relevance_num"].tolist()
            q_enc = tok([q], padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
            p_enc = tok(p_texts, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
            q_out = model.encoder(**q_enc).last_hidden_state.mean(1)
            p_out = model.encoder(**p_enc).last_hidden_state.mean(1)
            q_out = torch.nn.functional.normalize(q_out, p=2, dim=1)
            p_out = torch.nn.functional.normalize(p_out, p=2, dim=1)
            sims = (q_out @ p_out.T).cpu().numpy().ravel()
            recalls.append(recall_at_k(y_true, sims, k=100))
            ndcgs.append(ndcg_at_k(y_true, sims, k=20))
    print(f"Recall@100={np.mean(recalls):.4f}, NDCG@20={np.mean(ndcgs):.4f}")

    model.save_pretrained("my_model")
    tok.save_pretrained("my_model")
    print("saved model to ./my_model")

# ---------- run ----------
if __name__ == "__main__":
    path = "/Users/gilikurtser/Desktop/Dataset and Baseline/hsrc/hsrc_train.jsonl"
    data = [json.loads(x) for x in open(path)]
    df = pd.DataFrame(data)
    df['paragraph_uuids'], df['paragraph_texts'] = zip(*df['paragraphs'].map(lambda d: ([v['uuid'] for v in d.values()], [v['passage'] for v in d.values()])))
    df['relevances'] = df['target_actions'].map(lambda d: [v for v in d.values()])
    rows = []
    for _, r in df.iterrows():
        for pid, ptxt, rel in zip(r['paragraph_uuids'], r['paragraph_texts'], r['relevances']):
            rows.append({"query_uuid": r['query_uuid'], "query": r['query'], "paragraph_uuid": pid, "paragraph_text": ptxt, "relevance_num": rel})
    df_long = pd.DataFrame(rows)
    train_model(df_long)