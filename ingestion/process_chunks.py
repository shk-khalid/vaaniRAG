#!/usr/bin/env python3
"""
process_chunks.py

Loads dataset slices, filters by target language, flattens to canonical document format,
chunks using 5 different strategies, exports chunk files and evaluation query maps,
and prints comparative statistics.
"""

import os
import sys
import argparse
import json
import pandas as pd
from typing import Any, Dict, List

# Add current folder to path to import helpers from inspect_dataset
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from inspect_dataset import get_parquet_urls, load_sample_df
from chunking import (
    normalize_row,
    chunk_passage_level,
    chunk_sentence_level,
    chunk_overlapping_sentences,
    chunk_heuristic_semantic,
    chunk_adaptive
)

# Map requested languages to their partition file index in MSMARCO-XI validation/train splits
LANGUAGE_TO_PARTITION_INDEX = {
    "hin_Deva": 3,
    "hin_Devn": 3,
    "mar_Deva": 6,
    "mar_Devn": 6,
    "urd_Arab": 13,
    "eng_Latn": 3  # English exists in all files, we use partition 3 as default source
}


def apply_chunking_strategy(
    strategy: str,
    canonical_docs: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Applies the specified chunking strategy to a list of canonical documents.
    """
    chunks = []
    for doc in canonical_docs:
        text = doc["text"]
        
        # Apply chunking method
        if strategy == "passage":
            chunk_texts = chunk_passage_level(text)
        elif strategy == "sentence":
            chunk_texts = chunk_sentence_level(text)
        elif strategy == "overlap":
            chunk_texts = chunk_overlapping_sentences(text, window_size=3, overlap=1)
        elif strategy == "semantic":
            chunk_texts = chunk_heuristic_semantic(text, similarity_threshold=0.15)
        elif strategy == "adaptive":
            chunk_texts = chunk_adaptive(text, threshold_words=90)
        else:
            raise ValueError(f"Unknown chunking strategy: {strategy}")
            
        # Wrap chunk text into structured record
        for c_idx, chunk_text in enumerate(chunk_texts):
            chunks.append({
                "chunk_id": f"{doc['document_id']}_c{c_idx}",
                "document_id": doc["document_id"],
                "query_id": doc["query_id"],
                "text": chunk_text,
                "language": doc["language"],
                "is_selected": doc["is_selected"],
                "strategy": strategy,
                "source_passage_index": doc["source_passage_index"]
            })
            
    return chunks


def process_dataset(
    normalized_by_lang: Dict[str, List[Dict[str, Any]]],
    evaluation_queries: Dict[str, Any],
    languages: List[str],
    strategies: List[str],
    output_dir: str
) -> None:
    """
    Processes the normalized documents: applies chunking strategies,
    saves the output datasets, and prints comparison metrics.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Write evaluation queries (independent of chunking strategy)
    eval_queries_file = os.path.join(output_dir, "evaluation_queries.jsonl")
    print(f"\n[2/3] Writing evaluation queries mapping to: {eval_queries_file}...")
    with open(eval_queries_file, "w", encoding="utf-8") as f:
        for q_val in evaluation_queries.values():
            f.write(json.dumps(q_val, ensure_ascii=False) + "\n")

    # 2. Chunking strategies application
    print(f"\n[3/3] Applying chunking strategies and exporting database chunk JSONLs...")
    stats_data = []

    for strategy in strategies:
        strategy_chunks = []
        for lang in languages:
            docs = normalized_by_lang[lang]
            chunks = apply_chunking_strategy(strategy, docs)
            strategy_chunks.extend(chunks)
            
            # Gather statistics
            if chunks:
                word_counts = [len(c["text"].split()) for c in chunks]
                char_counts = [len(c["text"]) for c in chunks]
                s_words = pd.Series(word_counts)
                s_chars = pd.Series(char_counts)
                
                stats_data.append({
                    "Language": lang,
                    "Strategy": strategy,
                    "Total Chunks": len(chunks),
                    "Mean Words": round(s_words.mean(), 1),
                    "Median Words": int(s_words.median()),
                    "P90 Words": int(s_words.quantile(0.90)),
                    "Mean Chars": round(s_chars.mean(), 1),
                    "Selected Chunks": sum(1 for c in chunks if c["is_selected"])
                })

        # Save strategy chunks to JSONL
        out_file = os.path.join(output_dir, f"chunks_{strategy}.jsonl")
        print(f"  - Writing {strategy} chunks to: {out_file}")
        with open(out_file, "w", encoding="utf-8") as f:
            for chunk in strategy_chunks:
                f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    # 3. Print comparison statistics
    print("\nChunking Strategy Comparison Report:")
    if stats_data:
        df_stats = pd.DataFrame(stats_data)
        # Format table output
        print(df_stats.to_string(index=False))
    else:
        print("No statistics collected (verify language matches or record counts).")


def main() -> None:
    parser = argparse.ArgumentParser(description="VaaniRAG Chunking and Evaluation Dataset Creator.")
    parser.add_argument("--split", type=str, default="validation", help="Dataset split to pull from (validation or train)")
    parser.add_argument("--count", type=int, default=100, help="Number of raw query records to load per language partition")
    parser.add_argument("--languages", type=str, default="eng_Latn,hin_Deva,mar_Deva,urd_Arab", help="Comma-separated list of languages to process")
    args = parser.parse_args()

    dataset_name = "ai4bharat/MSMARCO-XI"
    languages_list = [l.strip() for l in args.languages.split(",") if l.strip()]
    strategies_list = ["passage", "sentence", "overlap", "semantic", "adaptive"]
    output_dir = "data"

    print("=================================================================")
    print("      VaaniRAG Phase 2: Chunking & Dataset Preparation")
    print("=================================================================")
    print(f"Languages:  {languages_list}")
    print(f"Strategies: {strategies_list}")
    print(f"Target count: {args.count} query records per language")

    # 1. Fetch remote URL list
    urls = get_parquet_urls(dataset_name, split=args.split)
    if not urls:
        print(f"Error: No parquet files found for split {args.split}.", file=sys.stderr)
        sys.exit(1)

    normalized_by_lang = {lang: [] for lang in languages_list}
    evaluation_queries = {}

    print(f"\n[1/3] Loading partition files and flattening to canonical document format...")
    # Load and normalize per partition
    # Group target languages by the partition index they require to minimize redundant requests
    partition_to_langs = {}
    for lang in languages_list:
        part_idx = LANGUAGE_TO_PARTITION_INDEX.get(lang)
        if part_idx is None:
            print(f"Warning: Unknown language '{lang}', skipping.", file=sys.stderr)
            continue
        partition_to_langs.setdefault(part_idx, []).append(lang)

    for part_idx, langs in partition_to_langs.items():
        if part_idx >= len(urls):
            print(f"Error: Partition index {part_idx} is out of bounds for split URLs.", file=sys.stderr)
            continue
            
        parquet_url = urls[part_idx]
        print(f"  - Fetching {args.count} raw query records from partition index {part_idx} (Target for {langs})...")
        df = load_sample_df(parquet_url, num_rows=args.count)
        
        # Calculate and report passage numbers
        raw_passages_count = 0
        for _, row in df.iterrows():
            passages = row.get("passages")
            if isinstance(passages, dict):
                translated_passages = passages.get("Translated_passages")
                if hasattr(translated_passages, "__iter__"):
                    raw_passages_count += len(translated_passages)
        
        print(f"    * Loaded {len(df)} query records containing {raw_passages_count} passages.")

        # Normalize and filter
        for _, row in df.iterrows():
            row_dict = row.to_dict()
            for lang in langs:
                docs = normalize_row(row_dict, lang)
                if docs:
                    normalized_by_lang[lang].extend(docs)
                    
                    # Gather unique query mapping for evaluation queries
                    query_id = row_dict["query_id"]
                    for d in docs:
                        q_key = f"{query_id}_{lang}"
                        if q_key not in evaluation_queries:
                            evaluation_queries[q_key] = {
                                "query_id": query_id,
                                "query": d["query"],
                                "language": d["language"],
                                "relevant_document_ids": []
                            }
                        if d["is_selected"]:
                            evaluation_queries[q_key]["relevant_document_ids"].append(d["document_id"])

    # Process all chunks, outputs and comparative statistics
    process_dataset(
        normalized_by_lang=normalized_by_lang,
        evaluation_queries=evaluation_queries,
        languages=languages_list,
        strategies=strategies_list,
        output_dir=output_dir
    )


if __name__ == "__main__":
    main()
