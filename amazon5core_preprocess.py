
# Script for preprocessing data  to get Artifacts for Dual-Tower/KNN retrieval

# Inputs:
#   Electronics.train.csv  (user_id,parent_asin,rating,timestamp)
#   Electronics.valid.csv  
#   Electronics.test.csv   
#   meta_Electronics.jsonl (per-item metadata)


# Outputs with format:
#   items.parquet       (parent_asin, item_text)
#   train.jsonl         ({user_id, user_idx, history[int], target[int], ts})
#   valid.jsonl
#   test.jsonl
#   mappings.json       ({item2idx, user2idx})


import json, re, gzip, argparse
from pathlib import Path
from typing import Dict, List, Tuple, Iterable, Optional

import pandas as pd


# Optional GPU backend (cuDF)

def _probe_gpu_backend():
    try:
        import cudf
        import cupy as cp
        _ = cudf.Series([1, 2, 3]).sum()
        return True
    except Exception:
        return False

USE_GPU = False
GPU_AVAILABLE = _probe_gpu_backend()


CATEGORY = "Electronics"
INPUT_DIR = Path("./raw_data")                      
OUT_DIR = Path("./preprocessed_data") / CATEGORY
OUT_DIR.mkdir(parents=True, exist_ok=True)


HIST_K = 10 # Length of history
MAX_TOKENS = 256


# Helper functions

def read_interactions_csv(path: Path) -> pd.DataFrame:
    """Read an interactions CSV. Expected columns:
       user_id,parent_asin,rating,timestamp[,history]"""
    df = pd.read_csv(path)
    needed = {"user_id", "parent_asin", "rating", "timestamp"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"{path.name} missing required columns: {missing}")
    # Normalize dtypes
    df["user_id"] = df["user_id"].astype(str)
    df["parent_asin"] = df["parent_asin"].astype(str)
    # timestamps in ms preferred; keep as int
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce").astype("Int64")
    # Optional history column in valid/test
    if "history" in df.columns:
        df["history"] = df["history"].astype(str)
    return df

def read_jsonl_any(path: Path) -> Iterable[dict]:
    """Read jsonl or jsonl.gz file."""
    open_fn = gzip.open if path.suffix == ".gz" else open
    with open_fn(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)

def uniq(seq: Iterable[str]) -> List[str]:
    seen = set()
    out = []
    for x in seq:
        if x and x not in seen:
            seen.add(x); out.append(x)
    return out

def flatten_details(d: dict) -> List[str]:
    """Flatten details dict into 'key: value' short phrases; handle nested dicts (1 level)."""
    out = []
    for k, v in d.items():
        if v is None or v == "":
            continue
        if isinstance(v, dict):
            # e.g., {"Best Sellers Rank": {"SecureDigital Memory Cards": 2870}}
            for k2, v2 in v.items():
                out.append(f"{k} {k2}: {v2}")
        elif isinstance(v, list):
            # compress small lists
            vals = ", ".join([str(x) for x in v[:5]])
            out.append(f"{k}: {vals}")
        else:
            out.append(f"{k}: {v}")
    return out

def safe_join(items: Iterable[str], sep: str = " ") -> str:
    return sep.join([s.strip() for s in items if s and str(s).strip()])

def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()

def truncate_tokens(text: str, max_tokens: int = MAX_TOKENS) -> str:
    toks = text.split()
    if len(toks) <= max_tokens:
        return text
    return " ".join(toks[:max_tokens])

def build_item_text(meta_row: dict) -> str:
    """
    Compose a compact item_text from fields:
    main_category + title + store + average_rating + joined features +
    joined categories + flattened details + bought_together + Manufacturer
    """
    # Pull fields with defaults
    main_category = meta_row.get("main_category") or ""
    title         = meta_row.get("title") or ""
    store         = meta_row.get("store") or ""
    avg_rating    = meta_row.get("average_rating")
    features      = meta_row.get("features") or []
    categories    = meta_row.get("categories") or []
    details       = meta_row.get("details") or {}
    bought_together = meta_row.get("bought_together")
    manufacturer  = None
    if isinstance(details, dict):
        manufacturer = details.get("Manufacturer") or details.get("Brand") or None

    # De-duplicate features/categories
    features = uniq([str(x) for x in features if x])
    categories = uniq([str(x) for x in categories if x])

    # Flatten details compactly
    detail_bits = flatten_details(details) if isinstance(details, dict) else []

    # Compose sections with lightweight labels to aid the LM
    sections = []
    if main_category: sections.append(f"[CAT] {main_category}")
    if title:         sections.append(f"[TITLE] {title}")
    if store:         sections.append(f"[STORE] {store}")
    if avg_rating is not None: sections.append(f"[AVG_RATING] {avg_rating}")
    if features:      sections.append("[FEAT] " + " | ".join(features[:10]))
    if categories:    sections.append("[PATH] " + " > ".join(categories[:8]))
    if detail_bits:   sections.append("[DETAIL] " + " | ".join(detail_bits[:20]))
    if bought_together: sections.append(f"[BT] {bought_together}")
    if manufacturer:  sections.append(f"[MFR] {manufacturer}")

    txt = normalize_whitespace(" ".join(sections))
    txt = truncate_tokens(txt, MAX_TOKENS)
    return txt

def load_metadata(meta_path: Path) -> pd.DataFrame:
    rows = []
    for obj in read_jsonl_any(meta_path):
        # Require parent_asin to merge with interactions
        pasin = obj.get("parent_asin")
        if not pasin: 
            continue
        item_text = build_item_text(obj)
        rows.append({"parent_asin": str(pasin), "item_text": item_text})
    meta_df = pd.DataFrame(rows).drop_duplicates(subset=["parent_asin"])
    return meta_df


def _to_gdf(df: pd.DataFrame):
    import cudf 
    return cudf.from_pandas(df)

def _to_pdf(gdf):
    return gdf.to_pandas()

def filter_to_intersection(items_df, inter_df):
    """Return rows of items_df where parent_asin is in inter_df.parent_asin."""
    if USE_GPU:
        import cudf
        keep = inter_df["parent_asin"].astype("str").dropna().unique()
        out = items_df[items_df["parent_asin"].isin(keep)]
        return out.reset_index(drop=True)
    else:
        keep = set(inter_df["parent_asin"].unique().tolist())
        return items_df[items_df["parent_asin"].isin(keep)].reset_index(drop=True)

def build_mappings(items_df, inter_df_all) -> Tuple[Dict[str,int], Dict[str,int]]:
    """Build item2idx over kept items and user2idx over all users seen."""
    if USE_GPU:
        # items
        item_ids = items_df["parent_asin"].astype("str").unique().to_pandas().tolist()
        # users
        user_ids = inter_df_all["user_id"].astype("str").unique().to_pandas().tolist()
    else:
        item_ids = items_df["parent_asin"].unique().tolist()
        user_ids = inter_df_all["user_id"].unique().tolist()
    item2idx = {asin: i for i, asin in enumerate(item_ids)}
    user2idx = {u: i for i, u in enumerate(user_ids)}
    return item2idx, user2idx

def sort_interactions(df):
    """
    Ensure 'timestamp' is numeric and sort by (user_id, timestamp, parent_asin).
    Returns the same kind (pandas or cuDF) as input.
    """
    if USE_GPU:
        import cudf
        df = df.dropna(subset=["timestamp"])
        df["timestamp"] = df["timestamp"].astype("int64")
        df["user_id"] = df["user_id"].astype("str")
        df["parent_asin"] = df["parent_asin"].astype("str")
        df = df.sort_values(by=["user_id", "timestamp", "parent_asin"]).reset_index(drop=True)
        return df
    else:
        df = df.dropna(subset=["timestamp"]).copy()
        df["timestamp"] = df["timestamp"].astype("int64")
        df = df.sort_values(["user_id", "timestamp", "parent_asin"]).reset_index(drop=True)
        return df

def make_train_triples(df_train: pd.DataFrame, item2idx: Dict[str,int], user2idx: Dict[str,int], k_hist: int = HIST_K) -> Iterable[dict]:
    """
    Function transforms each user’s chronological interaction history into training triples (user, history, target)
    
    k_hist controls how many recent items are included in each history window.

    
    """
    for uid, grp in df_train.groupby("user_id", sort=False):
        seq_items = [asin for asin in grp["parent_asin"].tolist() if asin in item2idx]
        seq_ts    = grp["timestamp"].tolist()
        if len(seq_items) < 2:
            continue
        history = []
        for idx in range(len(seq_items)):
            item = seq_items[idx]
            ts   = seq_ts[idx]
            if len(history) >= 1:
                hist_idx = [item2idx[h] for h in history[-k_hist:]]
                yield {
                    "user_id": uid,
                    "user_idx": user2idx[uid],
                    "history": hist_idx,
                    "target": item2idx[item],
                    "ts": int(ts),
                }
            history.append(item)

def parse_history_col(s: Optional[str]) -> List[str]:
    # history may be space-separated parent_asins
    if not s or pd.isna(s):
        return []
    return [tok for tok in str(s).split() if tok]

def make_eval_rows(df_eval: pd.DataFrame, item2idx: Dict[str,int], user2idx: Dict[str,int], k_hist: int = HIST_K) -> Iterable[dict]:
    """
    Use provided rows if 'history' exists; else apply last-two policy per user.
    """
    if "history" in df_eval.columns:
        # Each row is a single (history -> target)
        for _, row in df_eval.iterrows():
            uid = row["user_id"]
            ts  = int(row["timestamp"]) if pd.notna(row["timestamp"]) else 0
            hist_asins = parse_history_col(row["history"])
            hist_idx = [item2idx[a] for a in hist_asins if a in item2idx][-k_hist:]
            tgt_asin = str(row["parent_asin"])
            if tgt_asin not in item2idx or uid not in user2idx or len(hist_idx) == 0:
                continue
            yield {
                "user_id": uid,
                "user_idx": user2idx[uid],
                "history": hist_idx,
                "target": item2idx[tgt_asin],
                "ts": ts,
            }
    else:
        df_eval_sorted = sort_interactions(df_eval)
        for uid, grp in df_eval_sorted.groupby("user_id", sort=False):
            asins = [a for a in grp["parent_asin"].tolist() if a in item2idx]
            tss   = grp["timestamp"].tolist()
            if len(asins) < 2 or uid not in user2idx:
                continue
            target = asins[-1]
            hist   = asins[:-1][-k_hist:]
            hist_idx = [item2idx[h] for h in hist]
            yield {
                "user_id": uid,
                "user_idx": user2idx[uid],
                "history": hist_idx,
                "target": item2idx[target],
                "ts": int(tss[-1]),
            }

def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for obj in rows:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            n += 1
    return n


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess Amazon 5-core for dual-tower/KNN")
    parser.add_argument("--gpu", action="store_true", help="Use cuDF on GPU when available")
    args = parser.parse_args()
    
    USE_GPU = bool(args.gpu) and GPU_AVAILABLE
    if args.gpu and not USE_GPU:
        print("[warn] --gpu requested but cuDF/CUDA not available; falling back to CPU.")

    # Load train, val, test
    train_path = INPUT_DIR / f"{CATEGORY}.train.csv"
    valid_path = INPUT_DIR / f"{CATEGORY}.valid.csv"
    test_path  = INPUT_DIR / f"{CATEGORY}.test.csv"
    meta_path  = INPUT_DIR / f"meta_{CATEGORY}.jsonl"

    print("Reading interactions...")
    df_train_pd = read_interactions_csv(train_path)
    df_valid_pd = read_interactions_csv(valid_path)
    df_test_pd  = read_interactions_csv(test_path)

    print("Reading metadata...")
    items_df_pd = load_metadata(meta_path)[["parent_asin", "item_text"]]

    
    if USE_GPU:
        import cudf 
        df_train = _to_gdf(df_train_pd)
        df_valid = _to_gdf(df_valid_pd)
        df_test  = _to_gdf(df_test_pd)
        items_df = _to_gdf(items_df_pd)
    else:
        df_train, df_valid, df_test = df_train_pd, df_valid_pd, df_test_pd
        items_df = items_df_pd

   
    # Keep overlapping items only
    
    print("Filtering to intersection (items with interactions + text)...")
    if USE_GPU:
        import cudf 
        inter_all = cudf.concat(
            [df_train[["parent_asin"]], df_valid[["parent_asin"]], df_test[["parent_asin"]]],
            axis=0, ignore_index=True,
        )
    else:
        inter_all = pd.concat(
            [df_train[["parent_asin"]], df_valid[["parent_asin"]], df_test[["parent_asin"]]],
            axis=0
        )
    items_df = filter_to_intersection(items_df, inter_all)

    
    # Sort interactions chronologically (per user)
    
    df_train = sort_interactions(df_train)
    df_valid = sort_interactions(df_valid)
    df_test  = sort_interactions(df_test)

    
    # Build mappings
    
    if USE_GPU:
        import cudf 
        inter_df_all = cudf.concat([df_train, df_valid, df_test], axis=0, ignore_index=True)
    else:
        inter_df_all = pd.concat([df_train, df_valid, df_test], axis=0)
    item2idx, user2idx = build_mappings(items_df, inter_df_all)

    if USE_GPU:
        n_items = int(items_df.shape[0])
        n_users = len(user2idx)
    else:
        n_items = len(items_df)
        n_users = len(user2idx)

    print(f"Items kept: {n_items:,} | Users: {n_users:,}")

    
    items_out = OUT_DIR / "items.parquet"
    if USE_GPU:
        items_df.to_parquet(items_out)
        items_df_pd_head = _to_pdf(items_df.head(3))
    else:
        items_df.to_parquet(items_out, index=False)
        items_df_pd_head = items_df.head(3)
    print(f"Wrote {items_out} ({n_items:,} rows)")

    
    if USE_GPU:
        df_train_pd = _to_pdf(df_train)
        df_valid_pd = _to_pdf(df_valid)
        df_test_pd  = _to_pdf(df_test)
        # Convert dtypes to appropriate format.
        for _df in (df_train_pd, df_valid_pd, df_test_pd):
            _df["user_id"] = _df["user_id"].astype(str)
            _df["parent_asin"] = _df["parent_asin"].astype(str)
            _df["timestamp"] = _df["timestamp"].astype("int64")
    else:
        df_train_pd, df_valid_pd, df_test_pd = df_train, df_valid, df_test

  
    # Build & write train set
  
    train_out = OUT_DIR / "train.jsonl"
    n_train = write_jsonl(train_out, make_train_triples(df_train_pd, item2idx, user2idx, HIST_K))
    print(f"Wrote {train_out} ({n_train:,} triples)")

  
    # Build & write valid set
    
    valid_out = OUT_DIR / "valid.jsonl"
    n_valid = write_jsonl(valid_out, make_eval_rows(df_valid_pd, item2idx, user2idx, HIST_K))
    print(f"Wrote {valid_out} ({n_valid:,} rows)")

  
    # Build & write test set
   
    test_out = OUT_DIR / "test.jsonl"
    n_test = write_jsonl(test_out, make_eval_rows(df_test_pd, item2idx, user2idx, HIST_K))
    print(f"Wrote {test_out} ({n_test:,} rows)")

   
    # Write mappings
   
    maps_out = OUT_DIR / "mappings.json"
    with open(maps_out, "w", encoding="utf-8") as f:
        json.dump({"item2idx": item2idx, "user2idx": user2idx}, f)
    print(f"Wrote {maps_out}")

   
    # Sanity check of the output
 
    print("\nSample items.parquet rows:")
    print(items_df_pd_head)

    print("\nSample train.jsonl lines:")
    with open(train_out, "r", encoding="utf-8") as f:
        for _ in range(3):
            line = f.readline().strip()
            if not line: break
            print(line)
