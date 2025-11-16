import argparse
import json
from pathlib import Path

import pandas as pd

def _probe_gpu_backend() -> bool:
    try:
        import cudf 
        import cupy as cp 
        return True
    except Exception:
        return False

GPU_AVAILABLE = _probe_gpu_backend()
USE_GPU = False 


def load_parquet_gpu(path: Path):
    import cudf
    return cudf.read_parquet(path)


def to_pandas_chunks_gdf(gdf, chunk_rows: int):
    """
    Yield pandas DataFrames of size <= chunk_rows from a cuDF DataFrame
    without materializing the whole table on host at once.
    """
    n = int(gdf.shape[0])
    if n == 0:
        return
    for start in range(0, n, chunk_rows):
        stop = min(start + chunk_rows, n)
        yield gdf.iloc[start:stop].to_pandas()


def write_jsonl_from_df_iter(df_iter, out_path: Path):
    """
    df_iter: iterable of pandas DataFrames, each with columns ['parent_asin', 'item_text']
    """
    total = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for df in df_iter:
            # Ensure required columns exist
            if not {"parent_asin", "item_text"}.issubset(df.columns):
                raise ValueError("Input frame must contain 'parent_asin' and 'item_text'")
            for parent_asin, item_text in zip(df["parent_asin"], df["item_text"]):
                obj = {"parent_asin": parent_asin, "item_text": item_text}
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            total += len(df)
    return total


def main():
    parser = argparse.ArgumentParser(
        description="Convert items.parquet → items.jsonl (GPU-optional via cuDF)."
    )
    parser.add_argument("--category", default="Electronics", help="Dataset category")
    parser.add_argument("--in-dir", default="./data/preprocessed_data", help="Base data dir containing <cat>/items.parquet")
    parser.add_argument("--out-dir", default="./data/preprocessed_data", help="Base data dir to write <cat>/items.jsonl")
    parser.add_argument("--gpu", action="store_true", help="Use cuDF on GPU when available")
    parser.add_argument("--chunk-rows", type=int, default=500_000,
                        help="Rows per chunk when moving cuDF→pandas (GPU mode)")
    args = parser.parse_args()

    global USE_GPU
    USE_GPU = bool(args.gpu) and GPU_AVAILABLE
    if args.gpu and not USE_GPU:
        print("[warn] --gpu requested but cuDF/CUDA not available; falling back to CPU.")

    category = args.category
    in_path = Path(args.in_dir) / category / "items.parquet"
    out_path = Path(args.out_dir) / category / "items.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if USE_GPU:
        import cudf

        gdf = load_parquet_gpu(in_path)
        print(f"[GPU] Loaded {int(gdf.shape[0]):,} rows from {in_path}")

        needed = ["parent_asin", "item_text"]
        missing = [c for c in needed if c not in gdf.columns]
        if missing:
            raise ValueError(f"Parquet is missing required columns: {missing}")
        gdf = gdf[needed]

        total = write_jsonl_from_df_iter(
            df_iter=to_pandas_chunks_gdf(gdf, args.chunk_rows),
            out_path=out_path,
        )
        print(f"Saved to {out_path} ({total:,} lines)")

    else:
        df = pd.read_parquet(in_path)
        print(f"Loaded {len(df):,} rows from {in_path}")

        if not {"parent_asin", "item_text"}.issubset(df.columns):
            raise ValueError("Parquet must contain 'parent_asin' and 'item_text'")

        total = write_jsonl_from_df_iter(df_iter=[df], out_path=out_path)
        print(f"Saved to {out_path} ({total:,} lines)")


if __name__ == "__main__":
    main()
