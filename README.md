# VaaniRAG: Voice-Enabled Retrieval-Augmented Generation

VaaniRAG is a voice-enabled Retrieval-Augmented Generation system designed for the HH Goa 2026 Task 2. It enables robust semantic retrieval and generation using translated Indic datasets.

## Repository Structure

```
vaaniRAG/
├── backend/          # Backend API services
├── ingestion/        # Dataset ingestion & preprocessing pipelines
│   └── inspect_dataset.py
├── evaluation/       # Evaluation scripts and test runs
├── frontend/         # Frontend voice-enable interface
├── data/             # Downloaded datasets/databases
│   └── .gitkeep
├── requirements.txt  # Project dependencies
└── README.md
```

---

## Phase 1: Dataset Understanding and Inspection

### Why Inspect the Dataset First?
Before designing the chunking architecture, it is essential to understand the underlying dataset features, length distribution, and schema structure. Doing so helps us to:
1. **Determine Chunk Sizes:** Analyze character and word lengths of passages to determine the optimal chunk size. Selecting too small a chunk size might fragment individual sentences, while too large a chunk size might dilute semantic meaning or exceed LLM context windows.
2. **Understand Schema Layout:** Discover the fields housing queries, answers, translation targets (`target_lang`), and passages. For example, `ai4bharat/MSMARCO-XI` nests passages under a struct (`passages`) with separate fields for `English_passages` and `Translated_passages`.
3. **Verify Metadata Mapping:** Extract context fields such as `is_selected` (binary flag for relevant passages) and `query_id` to build retrieval index filters.

### Running the Inspection Script
To execute the dataset inspection pipeline, activate your virtual environment and run the script from the repository root:

```bash
# Setup Environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run the Inspection Script
python ingestion/inspect_dataset.py
```

### Dataset Insights & Observations

When running the script, we discover that:
- **Configurations:** The dataset does not partition languages by HF BuilderConfigs; instead, it exposes a single `"default"` configuration containing all languages.
- **Split Sizes:** The dataset is extremely large:
  - **Train:** 10,080,140 rows (~129 GB raw text)
  - **Validation:** 1,371,174 rows (~16.7 GB raw text)
- **Features & Columns:**
  - `query_id`: Unique identifier for the query.
  - `query`: Translated query string (target Indic language).
  - `Eng_Query`: Original English query.
  - `Answer`: Translated answer text.
  - `Eng_Answer`: Original English answer.
  - `target_lang`: Language code of target translation (e.g. `asm_Beng`, `hin_Devn`).
  - `passages`: Nested struct containing:
    - `English_passages`: List of source passages.
    - `Translated_passages`: List of translated passages.
    - `is_selected`: Array of binary values indicating whether each passage is relevant to the query.

#### Text Length Statistics
Based on a sample analysis of 500 validation records (representing 4,995 total passages):

| Field | Minimum | Maximum | Mean | Median | 90th Percentile (P90) | 99th Percentile (P99) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Query (Indic chars)** | 7 | 90 | 35.54 | 34.0 | 53.0 | 71.01 |
| **Query (Indic words)** | 1 | 19 | 5.60 | 5.0 | 8.0 | 11.00 |
| **Passage (Indic chars)** | 51 | 3790 | 327.24 | 288.0 | 507.6 | 778.00 |
| **Passage (Indic words)** | 6 | 678 | 50.72 | 45.0 | 80.0 | 123.00 |
| **Answer (Indic chars)** | 1 | 3379 | 75.15 | 26.0 | 137.1 | 350.22 |
| **Answer (Indic words)** | 1 | 682 | 12.39 | 5.0 | 20.0 | 56.07 |
