import argparse
import csv
import json
import logging
import pickle
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from model import preprocess, predict

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect reranker outputs for sampled HSRC queries."
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=Path("hsrc/hsrc_train.jsonl"),
        help="HSRC dataset JSONL containing queries, paragraphs, and labels.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=50,
        help="Number of queries to sample for inspection.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling.",
    )
    parser.add_argument(
        "--retrieval-top-k",
        type=int,
        default=100,
        help="Embedding candidates retrieved before reranking (0 uses all).",
    )
    parser.add_argument(
        "--bm25-top-k",
        type=int,
        default=0,
        help="BM25 candidates to union with embeddings (0 disables expansion).",
    )
    parser.add_argument(
        "--rerank-top-k",
        type=int,
        default=20,
        help="Number of passages kept after reranking.",
    )
    parser.add_argument(
        "--disable-normalization",
        action="store_true",
        help="Skip Hebrew normalization when embedding passages/queries.",
    )
    parser.add_argument(
        "--disable-bm25",
        action="store_true",
        help="Skip BM25 lexical expansion and rely solely on E5 embeddings.",
    )
    parser.add_argument(
        "--bm25-alpha",
        type=float,
        default=0.75,
        help="Weight for E5 similarity when mixing with BM25 (alpha * E5 + (1-alpha) * BM25).",
    )
    parser.add_argument(
        "--embeddings-path",
        type=Path,
        default=None,
        help="Optional pickle of precomputed E5 passage embeddings to reuse.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("finetuning/metrics/reranker_inspection.csv"),
        help="CSV file to store inspection results (top and missed passages).",
    )
    return parser.parse_args()


def load_dataset(path: Path) -> Tuple[List[dict], Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    examples: List[dict] = []
    corpus_texts: Dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            query = row.get("query", "")
            paragraphs = row.get("paragraphs", {})
            target_actions = row.get("target_actions", {})
            if not query or not paragraphs or not target_actions:
                continue

            label_map: Dict[str, int] = {}
            para_keys = sorted(paragraphs.keys())
            target_keys = sorted(target_actions.keys())
            for p_key, t_key in zip(para_keys, target_keys):
                para = paragraphs[p_key]
                uuid = para.get("uuid")
                passage = para.get("passage", "")
                if not uuid or not passage:
                    continue
                corpus_texts.setdefault(str(uuid), passage)
                label = target_actions.get(t_key)
                try:
                    label_map[str(uuid)] = int(label)
                except (TypeError, ValueError):
                    label_map[str(uuid)] = 0

            if not label_map:
                continue
            examples.append({"query": query, "label_map": label_map})
    return examples, corpus_texts


def load_embeddings(path: Path) -> Dict[str, np.ndarray]:
    with path.open("rb") as handle:
        data = pickle.load(handle)
    if not isinstance(data, dict):
        raise ValueError("Embeddings file must be a dict of paragraph_uuid -> vector")
    return {str(k): np.asarray(v) for k, v in data.items()}


def replace_embeddings(preprocessed: dict, saved_embeddings: Dict[str, np.ndarray]) -> None:
    corpus_ids = preprocessed.get("corpus_ids", [])
    vectors = []
    for pid in corpus_ids:
        vec = saved_embeddings.get(pid)
        if vec is None:
            LOGGER.warning("Saved embeddings missing passage %s; keeping computed embeddings.", pid)
            return
        vectors.append(vec)
    preprocessed["corpus_embeddings"] = np.vstack(vectors)
    LOGGER.info("Replaced corpus embeddings with saved vectors (shape=%s)", preprocessed["corpus_embeddings"].shape)


def main() -> None:
    args = parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    dataset, corpus_texts = load_dataset(args.dataset_path)

    rng = random.Random(args.seed)
    rng.shuffle(dataset)
    sampled_queries = dataset[: min(args.sample_size, len(dataset))]

    if not sampled_queries:
        raise ValueError("Sampled query list is empty; adjust --sample-size or dataset path")

    relevant_ids = set()
    for example in sampled_queries:
        relevant_ids.update(example["label_map"].keys())
    corpus_dict = {pid: {"text": corpus_texts[pid]} for pid in relevant_ids if pid in corpus_texts}
    if not corpus_dict:
        raise ValueError("No passages available for the sampled queries.")

    preprocessed = preprocess(
        corpus_dict,
        use_normalization=not args.disable_normalization,
        use_bm25=not args.disable_bm25,
        bm25_alpha=args.bm25_alpha,
    )

    if args.embeddings_path and args.embeddings_path.exists():
        try:
            saved = load_embeddings(args.embeddings_path)
            replace_embeddings(preprocessed, saved)
        except Exception as exc:
            LOGGER.warning("Failed to load saved embeddings: %s", exc)

    output_rows: List[Dict[str, str]] = []
    reranker = preprocessed.get("reranker")
    corpus_ids = set(preprocessed.get("corpus_ids", []))

    for idx, example in enumerate(sampled_queries, 1):
        query_text = example["query"]
        label_map = example["label_map"]

        top_results = predict(
            {"query": query_text},
            preprocessed,
            retrieval_top_k=args.retrieval_top_k,
            bm25_expansion_k=args.bm25_top_k,
            rerank_top_k=args.rerank_top_k,
        )

        top_ids = set()
        for rank, entry in enumerate(top_results, start=1):
            pid = entry["paragraph_uuid"]
            top_ids.add(pid)
            output_rows.append(
                {
                    "sample_index": idx,
                    "query": query_text,
                    "list_type": "top",
                    "rank": rank,
                    "paragraph_uuid": pid,
                    "score": entry["score"],
                    "label": label_map.get(pid, 0),
                    "passage": corpus_texts.get(pid, ""),
                }
            )

        relevant_ids = {pid for pid, lbl in label_map.items() if lbl > 0 and pid in corpus_ids}
        missed_ids = [pid for pid in relevant_ids if pid not in top_ids]
        if missed_ids and reranker is not None:
            reranked_missed = reranker.rerank(
                query_text,
                [corpus_texts.get(pid, "") for pid in missed_ids],
                missed_ids,
                top_k=len(missed_ids),
            )
            score_map = dict(reranked_missed)
            for pid in missed_ids:
                output_rows.append(
                    {
                        "sample_index": idx,
                        "query": query_text,
                        "list_type": "missed",
                        "rank": "",
                        "paragraph_uuid": pid,
                        "score": score_map.get(pid, float("nan")),
                        "label": label_map.get(pid, 0),
                        "passage": corpus_texts.get(pid, ""),
                    }
                )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=[
                "sample_index",
                "query",
                "list_type",
                "rank",
                "paragraph_uuid",
                "score",
                "label",
                "passage",
            ],
        )
        writer.writeheader()
        writer.writerows(output_rows)

    LOGGER.info("Saved reranker inspection CSV to %s", args.output)


if __name__ == "__main__":
    main()
