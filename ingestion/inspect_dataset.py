#!/usr/bin/env python3
"""
inspect_dataset.py

Phase 1: Dataset Understanding and Inspection
Loads and inspects the ai4bharat/MSMARCO-XI dataset schema, configs, and
text length statistics for chunking design using direct PyArrow Parquet reads
to bypass the HF datasets nesting conversion bug and avoid large downloads.
"""

import sys
import argparse
import json
import urllib.request
from typing import Any, Dict, List, Optional
import pandas as pd
import pyarrow.parquet as pq
import fsspec


def load_dataset_info(dataset_name: str) -> Dict[str, Any]:
    """
    Fetches configurations, splits, and schema features from Hugging Face Dataset Server API.
    """
    print(f"\n[1/5] Fetching metadata for dataset '{dataset_name}' via HF Dataset Server API...")
    info_url = f"https://datasets-server.huggingface.co/info?dataset={dataset_name}"
    try:
        with urllib.request.urlopen(info_url, timeout=15) as response:
            data = json.loads(response.read().decode("utf-8"))
        
        configs = list(data.get("dataset_info", {}).keys())
        default_info = data.get("dataset_info", {}).get("default", {})
        splits = default_info.get("splits", {})
        
        print(f"  - Dataset Name: {dataset_name}")
        print(f"  - Available Configurations: {configs}")
        print(f"  - Available Splits: {list(splits.keys())}")
        for split_name, split_data in splits.items():
            print(f"    * Split '{split_name}': {split_data.get('num_examples', 'unknown')} rows")
            
        return {
            "dataset_name": dataset_name,
            "configs": configs,
            "splits": list(splits.keys()),
            "features": default_info.get("features", {}),
            "raw_metadata": data
        }
    except Exception as e:
        print(f"Error fetching dataset metadata via HF API: {e}", file=sys.stderr)
        print("Please check internet connection or if dataset name is correct.", file=sys.stderr)
        sys.exit(1)


def inspect_schema(features: Dict[str, Any]) -> Dict[str, str]:
    """
    Parses and reports the column names, features, and data types/nesting structure.
    """
    print(f"\n[2/5] Inspecting Schema and Features:")
    schema_summary = {}
    for col, val in features.items():
        if isinstance(val, dict):
            if "_type" in val and val["_type"] == "Value":
                dtype = val.get("dtype", "unknown")
            elif "English_passages" in val or "Translated_passages" in val:
                dtype = f"struct with fields: {list(val.keys())}"
            else:
                dtype = f"nested dict ({list(val.keys())})"
        else:
            dtype = str(val)
        schema_summary[col] = dtype
        print(f"  - {col}: {dtype}")
        
    return schema_summary


def get_parquet_urls(dataset_name: str, split: str = "validation") -> List[str]:
    """
    Retrieves the partitioned parquet URLs for the given split.
    """
    parquet_url = f"https://datasets-server.huggingface.co/parquet?dataset={dataset_name}"
    try:
        with urllib.request.urlopen(parquet_url, timeout=15) as response:
            data = json.loads(response.read().decode("utf-8"))
        
        files = data.get("parquet_files", [])
        urls = [f["url"] for f in files if f["split"] == split]
        return urls
    except Exception as e:
        print(f"Error fetching parquet URLs: {e}", file=sys.stderr)
        sys.exit(1)


def load_sample_df(parquet_url: str, num_rows: int = 500) -> pd.DataFrame:
    """
    Reads a small sample slice of rows from a remote parquet file using pyarrow.parquet.
    """
    print(f"\n[3/5] Loading sample data from remote Parquet partition...")
    print(f"  Source URL: {parquet_url}")
    try:
        # Use fsspec to open the file via range requests, avoiding downloading the whole file
        with fsspec.open(parquet_url, "rb") as f:
            pf = pq.ParquetFile(f)
            # Read first row group
            print(f"  Reading first row group (Total rows in file: {pf.metadata.num_rows})...")
            table = pf.read_row_group(0)
            df = table.to_pandas()
            # Slice to desired sample size
            return df.head(num_rows)
    except Exception as e:
        print(f"Error loading remote Parquet sample: {e}", file=sys.stderr)
        sys.exit(1)


def inspect_samples(df: pd.DataFrame, num_samples: int = 2) -> None:
    """
    Displays a few representative examples from the loaded DataFrame.
    """
    print(f"\n[4/5] Displaying {num_samples} Representative Examples:")
    samples = df.head(num_samples)
    for idx, row in samples.iterrows():
        print(f"\n--- Sample Row {idx + 1} ---")
        for col in df.columns:
            val = row[col]
            if isinstance(val, dict):
                print(f"  {col}: (dict)")
                for sub_k, sub_v in val.items():
                    # print first item of lists or truncated strings
                    if isinstance(sub_v, (list, tuple)):
                        sub_v_display = f"list of length {len(sub_v)}. First item: {str(sub_v[0])[:120]}..." if len(sub_v) > 0 else "[]"
                    else:
                        sub_v_display = str(sub_v)[:120] + "..." if len(str(sub_v)) > 120 else str(sub_v)
                    print(f"    * {sub_k}: {sub_v_display}")
            elif isinstance(val, list):
                print(f"  {col} (list length: {len(val)}): {str(val[:2])[:120]}...")
            else:
                val_str = str(val)
                if len(val_str) > 150:
                    val_str = val_str[:150] + "..."
                print(f"  {col}: {val_str}")


def analyze_text_lengths(df: pd.DataFrame) -> None:
    """
    Computes text length statistics (char/word count minimum, maximum, mean, median, P90, P99)
    for relevant text fields.
    """
    print(f"\n[5/5] Analyzing Text Lengths on {len(df)} records...")
    
    # Storage for lengths
    metrics = {
        "Query (Indic - char)": [],
        "Query (Indic - word)": [],
        "Eng_Query (char)": [],
        "Eng_Query (word)": [],
        "Answer (Indic - char)": [],
        "Answer (Indic - word)": [],
        "Eng_Answer (char)": [],
        "Eng_Answer (word)": [],
        "Passage (Indic - char)": [],
        "Passage (Indic - word)": [],
    }

    for _, row in df.iterrows():
        # Indic Query
        q = row.get("query")
        if isinstance(q, str) and q.strip():
            metrics["Query (Indic - char)"].append(len(q))
            metrics["Query (Indic - word)"].append(len(q.split()))

        # Eng Query
        eq = row.get("Eng_Query")
        if isinstance(eq, str) and eq.strip():
            metrics["Eng_Query (char)"].append(len(eq))
            metrics["Eng_Query (word)"].append(len(eq.split()))

        # Indic Answer
        ans = row.get("Answer")
        if isinstance(ans, str) and ans.strip():
            metrics["Answer (Indic - char)"].append(len(ans))
            metrics["Answer (Indic - word)"].append(len(ans.split()))

        # Eng Answer
        eans = row.get("Eng_Answer")
        if isinstance(eans, str) and eans.strip():
            metrics["Eng_Answer (char)"].append(len(eans))
            metrics["Eng_Answer (word)"].append(len(eans.split()))

        # Passages
        passages = row.get("passages")
        if isinstance(passages, dict):
            translated_passages = passages.get("Translated_passages")
            if hasattr(translated_passages, "__iter__") and not isinstance(translated_passages, (str, bytes)):
                for tp in translated_passages:
                    if isinstance(tp, str) and tp.strip():
                        metrics["Passage (Indic - char)"].append(len(tp))
                        metrics["Passage (Indic - word)"].append(len(tp.split()))
            elif isinstance(translated_passages, str) and translated_passages.strip():
                metrics["Passage (Indic - char)"].append(len(translated_passages))
                metrics["Passage (Indic - word)"].append(len(translated_passages.split()))

    stats_list = []
    for label, lst in metrics.items():
        if lst:
            s = pd.Series(lst)
            stats_list.append({
                "Field": label,
                "Min": int(s.min()),
                "Max": int(s.max()),
                "Mean": round(s.mean(), 2),
                "Median": round(s.median(), 2),
                "P90": round(s.quantile(0.90), 2),
                "P99": round(s.quantile(0.99), 2),
                "Count": len(lst)
            })

    if stats_list:
        df_stats = pd.DataFrame(stats_list)
        print("\nText Length Statistics Summary Table:")
        print(df_stats.to_string(index=False))
    else:
        print("No text statistics could be computed.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect HF dataset MSMARCO-XI via Server API.")
    parser.add_argument("--split", type=str, default="validation", help="Dataset split to inspect (validation or train)")
    parser.add_argument("--samples", type=int, default=2, help="Number of representative samples to display")
    parser.add_argument("--analyze-count", type=int, default=500, help="Number of samples to analyze for length stats")
    args = parser.parse_args()

    dataset_name = "ai4bharat/MSMARCO-XI"

    # Step 1 & 2: Load Info and Inspect Schema
    meta = load_dataset_info(dataset_name)
    inspect_schema(meta["features"])

    # Step 3: Get Parquet files and load sample
    urls = get_parquet_urls(dataset_name, split=args.split)
    if not urls:
        print(f"No parquet files found for split '{args.split}'", file=sys.stderr)
        sys.exit(1)
        
    df = load_sample_df(urls[0], num_rows=args.analyze_count)

    # Step 4: Display Samples
    inspect_samples(df, num_samples=args.samples)

    # Step 5: Analyze Text Lengths
    analyze_text_lengths(df)


if __name__ == "__main__":
    main()
