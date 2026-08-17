#!/usr/bin/env python3
"""
verify_eval_mapping.py

Verifies the integrity of the evaluation queries mapping (evaluation_queries.jsonl)
against the passage chunks database (chunks_passage.jsonl).
"""

import os
import sys
import json


def verify_integrity(data_dir: str = "data") -> None:
    chunks_file = os.path.join(data_dir, "chunks_passage.jsonl")
    queries_file = os.path.join(data_dir, "evaluation_queries.jsonl")

    if not os.path.exists(chunks_file) or not os.path.exists(queries_file):
        print(f"Error: Required files not found in '{data_dir}/'. Did you run process_chunks.py first?", file=sys.stderr)
        sys.exit(1)

    print("Checking ground truth integrity...")

    # 1. Load chunks database
    selected_docs = set()
    all_docs = {}
    
    with open(chunks_file, "r", encoding="utf-8") as f:
        for line in f:
            chunk = json.loads(line)
            doc_id = chunk["document_id"]
            all_docs[doc_id] = chunk
            if chunk["is_selected"]:
                selected_docs.add(doc_id)

    # 2. Load evaluation queries
    queries_mapped_selected = set()
    queries_by_id = {}
    
    errors = 0
    total_queries = 0

    with open(queries_file, "r", encoding="utf-8") as f:
        for line in f:
            q = json.loads(line)
            total_queries += 1
            q_id = q["query_id"]
            lang = q["language"]
            relevant_docs = q["relevant_document_ids"]
            
            for doc_id in relevant_docs:
                queries_mapped_selected.add(doc_id)
                
                # Check 1: Mapped doc_id must exist in chunks
                if doc_id not in all_docs:
                    print(f"  [Error] Query {q_id} ({lang}) links to non-existent document: {doc_id}")
                    errors += 1
                else:
                    # Check 2: Mapped doc_id must be selected in chunks
                    chk = all_docs[doc_id]
                    if not chk["is_selected"]:
                        print(f"  [Error] Query {q_id} links to document {doc_id} but chunk is_selected is False!")
                        errors += 1

    # Check 3: Every chunk in database with is_selected = True must be linked in queries
    for doc_id in selected_docs:
        if doc_id not in queries_mapped_selected:
            # Get original query id
            q_id = all_docs[doc_id]["query_id"]
            lang = all_docs[doc_id]["language"]
            print(f"  [Error] Document {doc_id} (Query {q_id}, Lang {lang}) is selected in chunks but not linked in queries!")
            errors += 1

    print(f"\nVerification complete:")
    print(f"  - Total queries checked: {total_queries}")
    print(f"  - Total unique selected passages: {len(selected_docs)}")
    print(f"  - Mapped passages in queries: {len(queries_mapped_selected)}")
    print(f"  - Errors found: {errors}")
    
    if errors == 0:
        print("\n[SUCCESS] Ground truth mapping is 100% consistent and ready for retrieval evaluation!")
        sys.exit(0)
    else:
        print("\n[FAILURE] Found inconsistencies in the dataset mapping.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    verify_integrity()
