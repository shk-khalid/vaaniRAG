"""
chunking.py

Implements canonical normalization, language filtering, and the 5 chunking strategies:
1. passage (entire passage)
2. sentence (individual sentences)
3. overlap (overlapping sentence windows)
4. semantic (heuristic vocabulary overlap sentence grouping)
5. adaptive (passage-level if short, sentence-level or overlap if long)
"""

import re
from typing import Any, Dict, List, Set


def split_into_sentences(text: str) -> List[str]:
    """
    Splits text into sentences, supporting both English punctuation and Indic sentence
    delimiters (like । or |).
    """
    if not text:
        return []
    # Split by standard sentence terminators and Indic danda (।)
    # Keep the delimiter with the sentence if possible, or just split cleanly
    sentence_endings = re.compile(r'([.!?।|])\s*')
    splits = sentence_endings.split(text)
    
    sentences = []
    # Reconstruct sentences with their punctuation
    for i in range(0, len(splits) - 1, 2):
        sent = splits[i].strip()
        punct = splits[i+1].strip()
        if sent:
            sentences.append(f"{sent} {punct}".strip())
    
    # Add any trailing text without punctuation
    if len(splits) % 2 == 1:
        last = splits[-1].strip()
        if last:
            sentences.append(last)
            
    return [s for s in sentences if s]


def chunk_passage_level(text: str) -> List[str]:
    """
    Passage-level chunking: keeps the entire text intact.
    """
    return [text] if text.strip() else []


def chunk_sentence_level(text: str) -> List[str]:
    """
    Sentence-level chunking: splits text into individual sentences.
    """
    return split_into_sentences(text)


def chunk_overlapping_sentences(text: str, window_size: int = 3, overlap: int = 1) -> List[str]:
    """
    Sliding window chunker over sentences.
    e.g. window_size=3, overlap=1:
      Chunk 1: S0 + S1 + S2
      Chunk 2: S2 + S3 + S4
    """
    sentences = split_into_sentences(text)
    if not sentences:
        return []
    
    if len(sentences) <= window_size:
        return [" ".join(sentences)]
        
    chunks = []
    step = window_size - overlap
    if step <= 0:
        step = 1  # prevent infinite loop
        
    for i in range(0, len(sentences), step):
        window = sentences[i : i + window_size]
        chunks.append(" ".join(window))
        # Stop if we reached or passed the end of the sentence list
        if i + window_size >= len(sentences):
            break
            
    return chunks


def get_token_sets(sentence: str) -> Set[str]:
    """
    Tokenizes a sentence by splitting on non-alphanumeric chars and lowercasing.
    """
    return set(re.findall(r'\w+', sentence.lower()))


def chunk_heuristic_semantic(text: str, similarity_threshold: float = 0.15) -> List[str]:
    """
    Heuristic semantic chunker: Groups adjacent sentences by vocabulary overlap.
    If the Jaccard similarity between two adjacent sentences falls below a threshold,
    we create a split boundary.
    """
    sentences = split_into_sentences(text)
    if not sentences:
        return []
    if len(sentences) == 1:
        return sentences

    chunks = []
    current_chunk = [sentences[0]]
    current_words = get_token_sets(sentences[0])

    for i in range(1, len(sentences)):
        next_words = get_token_sets(sentences[i])
        
        # Calculate Jaccard Similarity (vocabulary overlap)
        union = current_words.union(next_words)
        if not union:
            similarity = 0.0
        else:
            similarity = len(current_words.intersection(next_words)) / len(union)
            
        # If similarity is low, finalize current chunk and start a new one
        if similarity < similarity_threshold and len(current_chunk) >= 1:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentences[i]]
            current_words = next_words
        else:
            current_chunk.append(sentences[i])
            current_words = current_words.union(next_words)

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def chunk_adaptive(text: str, threshold_words: int = 90) -> List[str]:
    """
    Adaptive chunking:
    - If the passage is short (under threshold_words), it remains intact (passage level).
    - If it is long, splits it using overlapping sentences.
    """
    word_count = len(text.split())
    if word_count <= threshold_words:
        return chunk_passage_level(text)
    else:
        # Fall back to sliding window sentence chunking for long documents
        return chunk_overlapping_sentences(text, window_size=3, overlap=1)


def normalize_row(row: Dict[str, Any], target_language: str) -> List[Dict[str, Any]]:
    """
    Filters and normalizes a raw dataset row into flattened canonical document dictionaries.
    
    target_language can be:
    - 'eng_Latn' (uses Eng_Query and English_passages)
    - 'hin_Deva' or 'hin_Devn' (uses translated fields, filtered for Hindi target_lang)
    - 'mar_Deva' or 'mar_Devn' (uses translated fields, filtered for Marathi target_lang)
    - 'urd_Arab' (uses translated fields, filtered for Urdu target_lang)
    """
    query_id = row.get("query_id")
    target_lang = row.get("target_lang", "")
    
    passages = row.get("passages")
    if not isinstance(passages, dict):
        return []
        
    is_selected_list = passages.get("is_selected", [])
    
    # Determine text sources and language matching
    if target_language == "eng_Latn":
        # We can extract the original English context from any row
        query_text = row.get("Eng_Query", "")
        passages_list = passages.get("English_passages", [])
        lang_label = "eng_Latn"
    else:
        # Match the target language prefix (e.g. hin, mar, urd)
        lang_prefix = target_language.split("_")[0]
        if not target_lang.startswith(lang_prefix):
            return []  # Filtered out (not the requested language)
            
        query_text = row.get("query", "")
        passages_list = passages.get("Translated_passages", [])
        lang_label = target_lang

    if not hasattr(passages_list, "__iter__") or isinstance(passages_list, (str, bytes)):
        return []

    canonical_docs = []
    for idx, text in enumerate(passages_list):
        if not isinstance(text, str) or not text.strip():
            continue
            
        is_sel = False
        if is_selected_list is not None and idx < len(is_selected_list):
            is_sel = bool(is_selected_list[idx])
            
        canonical_docs.append({
            "document_id": f"{query_id}_p{idx}",
            "query": query_text,
            "language": lang_label,
            "text": text,
            "is_selected": is_sel,
            "source_passage_index": idx,
            "query_id": query_id
        })
        
    return canonical_docs
