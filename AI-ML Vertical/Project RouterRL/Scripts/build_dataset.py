"""
RouterRL - Step 1: Build a unified per-turn routing table.

Downloads (or reads locally-downloaded) files from JiaqiXue/mmr-routing-20k
and merges them into a single dataframe with, for every (conversation_hash, turn_idx):
    - the 26 handcrafted routing features
    - the query text
    - each model's response, quality score (0-10), and cost (word count)

Run modes:
    python build_dataset.py --download   # pulls files from the HF Hub first
    python build_dataset.py --local DIR  # reads files already downloaded into DIR

Output: routing_table.parquet
"""
import argparse
import json
import os
import pandas as pd

MODELS = ["qwen3-0.6b", "ministral-8b", "qwen3-30b-a3b", "qwen3-30b-a3b-instruct"]
REPO_ID = "JiaqiXue/mmr-routing-20k"


def download_files(target_dir: str) -> str:
    """Pull only the files we need (not the full 1.77GB repo)."""
    from huggingface_hub import hf_hub_download

    os.makedirs(target_dir, exist_ok=True)
    files_needed = ["data/features/qwen06b_20k.jsonl"]
    for m in MODELS:
        files_needed.append(f"data/{m}/responses.jsonl")
        files_needed.append(f"data/{m}/judge_scores.jsonl")

    for f in files_needed:
        print(f"Downloading {f} ...")
        hf_hub_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            filename=f,
            local_dir=target_dir,
        )
    return os.path.join(target_dir, "data")


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_features(data_dir: str) -> pd.DataFrame:
    rows = load_jsonl(os.path.join(data_dir, "features", "qwen06b_20k.jsonl"))
    flat = []
    for r in rows:
        rec = {"conversation_hash": r["conversation_hash"], "turn_idx": r["turn_idx"]}
        rec.update(r["features"])
        rec["difficulty"] = r.get("difficulty")
        # NOTE: weak_score / strong_score here belong to the features file's own
        # labeling scheme, not the per-model judge scores merged below.
        flat.append(rec)
    return pd.DataFrame(flat)


def load_model_data(data_dir: str, model: str) -> pd.DataFrame:
    """Return one row per (conversation_hash, turn_idx) with query, response,
    cost (word count of response), and quality score for this model."""
    responses = load_jsonl(os.path.join(data_dir, model, "responses.jsonl"))
    scores = load_jsonl(os.path.join(data_dir, model, "judge_scores.jsonl"))

    resp_rows = []
    for conv in responses:
        ch = conv["conversation_hash"]
        for t in conv["turns"]:
            resp_rows.append({
                "conversation_hash": ch,
                "turn_idx": t["turn_idx"],
                "query": t["user_query"],
                f"{model}_response": t["response"],
                f"{model}_cost": len(t["response"].split()),  # 1 token = 1 word
            })
    resp_df = pd.DataFrame(resp_rows)

    score_rows = [{
        "conversation_hash": s["conversation_hash"],
        "turn_idx": s["turn_idx"],
        f"{model}_score": s["score"],
    } for s in scores]
    score_df = pd.DataFrame(score_rows)

    merged = resp_df.merge(score_df, on=["conversation_hash", "turn_idx"], how="left")
    return merged


def build(data_dir: str, out_path: str):
    print("Loading features...")
    df = load_features(data_dir)

    query_added = False
    for model in MODELS:
        print(f"Loading {model}...")
        model_df = load_model_data(data_dir, model)
        if not query_added:
            df = df.merge(model_df, on=["conversation_hash", "turn_idx"], how="inner")
            query_added = True
        else:
            # drop the duplicate query column before merging
            model_df = model_df.drop(columns=["query"])
            df = df.merge(model_df, on=["conversation_hash", "turn_idx"], how="inner")

    print(f"Final table: {len(df)} rows, {len(df.columns)} columns")
    df.to_parquet(out_path, index=False)
    print(f"Saved to {out_path}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true", help="download files from HF Hub")
    parser.add_argument("--local", type=str, default=None, help="path to already-downloaded data/ dir")
    parser.add_argument("--cache_dir", type=str, default="./hf_cache", help="where to download to")
    parser.add_argument("--out", type=str, default="routing_table.parquet")
    args = parser.parse_args()

    if args.download:
        data_dir = download_files(args.cache_dir)
    elif args.local:
        data_dir = args.local
    else:
        raise SystemExit("Pass --download or --local DIR")

    build(data_dir, args.out)
