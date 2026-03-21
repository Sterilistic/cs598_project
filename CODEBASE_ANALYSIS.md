# ReXKG — Codebase Analysis & Architecture Document

**Project**: ReXKG (Radiology Examination Knowledge Graph)  
**Paper**: *"Uncovering Knowledge Gaps in Radiology Report Generation Models through Knowledge Graphs"*  
**Purpose**: Builds a structured radiology knowledge graph from chest X-ray reports (CheXpert Plus dataset) via a multi-stage NLP pipeline.

---

## Table of Contents

1. [High-Level Architecture](#high-level-architecture)
2. [Phase 1 — GPT-4 Data Annotation](#phase-1--gpt-4-data-annotation)
3. [Phase 2 — NER Model Training](#phase-2--ner-model-training)
4. [Phase 3 — Full-Corpus Inference & Post-Processing](#phase-3--full-corpus-inference--post-processing)
5. [Phase 4 — Knowledge Graph Construction](#phase-4--knowledge-graph-construction)
6. [Output Files](#output-files)
7. [Key Design Decisions](#key-design-decisions)
8. [Dependencies & Environment](#dependencies--environment)

---

## High-Level Architecture

The system has **4 major phases**, each feeding into the next:

```
Raw Radiology Reports (CSV)
        │
        ▼
┌──────────────────────────┐
│  Phase 1: GPT-4 Annot.   │  ← Create NER training data
└──────────┬───────────────┘
           ▼
┌──────────────────────────┐
│  Phase 2: PURE Training   │  ← Train entity & relation models
└──────────┬───────────────┘
           ▼
┌──────────────────────────┐
│  Phase 3: Inference       │  ← Extract entities/relations corpus-wide
└──────────┬───────────────┘
           ▼
┌──────────────────────────┐
│  Phase 4: KG Construction │  ← Build UMLS-grounded knowledge graph
└──────────────────────────┘
```

### Project Structure

```
src/
├── ner/                          # Named Entity Recognition & Relation Extraction
│   ├── data/                     # Data preparation scripts
│   │   ├── gpt4_entity_extraction.py
│   │   ├── gpt4_relation_extraction.py
│   │   ├── structure_data.py
│   │   └── get_inference_data.py
│   ├── entity/                   # Entity extraction model
│   │   ├── models.py
│   │   └── utils.py
│   ├── relation/                 # Relation extraction model
│   │   ├── models.py
│   │   └── utils.py
│   ├── shared/                   # Shared constants & data structures
│   │   ├── const.py
│   │   ├── data_structures.py
│   │   └── data_aug.py
│   ├── run_entity.py             # Entity model training script
│   ├── run_entity.sh             # Entity SLURM launcher
│   ├── run_relation.py           # Relation model training script
│   ├── run_relation.sh           # Relation SLURM launcher
│   ├── run_inference.sh          # Two-stage inference launcher
│   └── result/
│       └── run_relation/
│           └── reverse_structure_data.py  # Post-processing
└── kg_construct/                 # Knowledge Graph Construction
    ├── code/
    │   ├── auto_build_kg.sh      # Pipeline orchestrator
    │   ├── get_entities.py       # Step 1: Aggregate entities/relations
    │   ├── get_umls_entities.py  # Step 2: UMLS linking
    │   ├── filter_cui.py         # Step 3: Filter best CUIs
    │   ├── structure_entities.py # Step 4a: Decompose entities
    │   ├── merge_entities.py     # Step 4b: Embedding dedup
    │   ├── get_kg_nodes.py       # Step 5: Build final KG
    │   └── get_size_relations.py # Step 6: Size/measurement edges
    └── result/                   # Example output KG files
        └── chexpert_plus_part1/
            └── chexpert_plus_groundtruth/
                ├── kg_nodes.json
                ├── kg_relations.csv
                └── kg_subgraphs.json
```

---

## Phase 1 — GPT-4 Data Annotation

**Location**: `src/ner/data/`

This phase uses GPT-4o to create silver-standard training data from raw radiology reports, avoiding the need for expensive manual annotation.

### `gpt4_entity_extraction.py`

- **Purpose**: Sends radiology sentences to GPT-4o with few-shot prompts to extract clinical entities.
- **Entity Types** (8 categories):
  - `anatomy` — Body parts/structures (e.g., "lungs", "right hilum")
  - `disorder_present` — Present conditions (e.g., "pleural effusion")
  - `disorder_notpresent` — Negated conditions (e.g., "no pneumothorax")
  - `concept` — Descriptive terms (e.g., "clear", "stable")
  - `procedures` — Medical procedures (e.g., "sternotomy")
  - `devices` — General medical devices
  - `devices_present` — Devices that are present
  - `devices_notpresent` — Devices that are absent
- **Key Functions**:
  - `make_entity_prompt()` — Constructs few-shot prompt with radiologist persona
  - `call_gpt4o()` — Calls GPT-4o API with JSON response format
  - `main()` — Reads CheXpert Plus CSV, iterates through findings, saves annotations
- **Output**: `gpt4_entities_chexpert_plus.json` — Per-image-ID dict with `section_findings` and `entities`

### `gpt4_relation_extraction.py`

- **Purpose**: Takes previously extracted entities and sends them back to GPT-4o to identify relations.
- **Relation Types** (3 categories):
  - `modify` — Descriptive modification (e.g., "small" → "effusion")
  - `located_at` — Anatomical location (e.g., "effusion" → "pleural")
  - `suggestive_of` — Diagnostic suggestion (e.g., "opacity" → "pneumonia")
- **Key Functions**:
  - `make_relation_prompt()` — Few-shot prompt defining source/target ordering conventions
  - `call_gpt4o()` — API call for relation extraction
  - `post_process()` — Normalizes GPT-4's variable output formats into consistent structure
- **Outputs**:
  - `gpt4_entities_relations_chexpert_plus.json` — Raw entities + relations
  - `gpt4_entities_relations_chexpert_plus_post.json` — Post-processed/cleaned version

### `structure_data.py`

- **Purpose**: Converts GPT-4 annotated JSON into **PURE NER training format** (JSONL with token-level span annotations).
- **Key Functions**:
  - `find_entity_index()` — Locates word-level start/end indices of an entity within a tokenized sentence
  - `get_entity()` — Maps entities to `[start, end, type]` spans, handling negation prefixes, size validation, device type resolution
  - `get_relation()` — Maps relation triplets to `[subj_start, subj_end, obj_start, obj_end, relation_type]`
  - `main()` — Iterates all sentences, builds JSONL with `doc_key`, `ner`, `relations`, `sentences`
- **Output**: `train.json` and `test.json` in `data_split/` (first 100 samples → test, rest → train)

### `get_inference_data.py`

- **Purpose**: Converts raw CSV reports into PURE inference format (sentences with empty NER/relation slots).
- **Key Functions**:
  - `split_into_sentences()` — Splits report text at periods, handling decimal numbers
  - `main()` — Reads CSV, tokenizes each sentence, creates JSONL with empty `ner` and `relations` arrays
- **Output**: A JSONL file ready for inference
- **Connection**: Feeds into `run_inference.sh`

### Data Flow

```
CheXpert+ CSV
    │
    ├─→ gpt4_entity_extraction.py → gpt4_entities_chexpert_plus.json
    │                                        │
    │                                        ▼
    │                              gpt4_relation_extraction.py
    │                                        │
    │                                        ▼
    │                              gpt4_entities_relations_chexpert_plus_post.json
    │                                        │
    │                                        ▼
    │                              structure_data.py → train.json / test.json
    │
    └─→ get_inference_data.py → inference_test.json (for full-corpus inference)
```

---

## Phase 2 — NER Model Training

**Location**: `src/ner/`

Built on the **[PURE](https://github.com/princeton-nlp/PURE)** (Princeton University Relation Extraction) framework, using **BiomedBERT** (`microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext`) as the backbone.

### Shared Components (`shared/`)

#### `const.py`
- Defines label sets for entity and relation types per task
- **Entity labels** (8): `size`, `anatomy`, `disorder_present`, `disorder_notpresent`, `concept`, `procedures`, `devices_present`, `devices_notpresent`
- **Relation labels** (3): `located_at`, `suggestive_of`, `modify`
- Creates bidirectional `label ↔ id` mappings (1-indexed, 0 reserved for "none")

#### `data_structures.py`
Core data representation classes based on the DYGIE++ format:

| Class | Description |
|-------|-------------|
| `Dataset` | Reads JSONL files, optionally merges gold + predictions. Holds list of `Document` objects |
| `Document` | Represents a document with multiple `Sentence` objects. Includes `data_augmentation()` method |
| `Sentence` | Holds tokenized text, gold NER labels, gold relation labels, and predicted versions |
| `Span` | Represents a token span with both document-level and sentence-level offsets |
| `EntityType` | Entity annotation: span + label |
| `NERRelation` | Relation annotation: pair of spans + label |
| `evaluate_predictions()` | Computes P/R/F1 for NER and relations (strict, no-type, and overlap variants) |

#### `data_aug.py`
- Synonym-based data augmentation using NLPAug library
- Replaces anatomy/concept tokens with synonyms to create augmented training examples
- Experimental/proof-of-concept; not directly called in the main pipeline

### Entity Model (`entity/`)

#### `entity/models.py`
- **`EntityModel`** — Span-based entity extraction:
  1. Feeds input through BERT to get contextual token embeddings
  2. Enumerates all candidate spans up to a max width
  3. Creates span embeddings: `[start_token_emb; end_token_emb; width_embedding]`
  4. Classifies each span via a 2-layer FFN with CrossEntropyLoss
- **`AlbertEntityModel`** — Same architecture using ALBERT backbone
- **`EntityPredictionModel`** — Wrapper handling tokenization (BERT subword → span index mapping), batching with dynamic padding, and train/eval forward pass

#### `entity/utils.py`
- `convert_dataset_to_samples()` — Enumerates all spans up to `max_span_length` in each sentence, assigns gold labels from NER annotations, optionally adds context window
- `batchify()` — Groups samples into batches, isolating long sentences (>350 tokens) into individual batches
- `get_train_fold()` / `get_test_fold()` — 10-fold cross-validation helpers

### Relation Model (`relation/`)

#### `relation/models.py`
- **`RelationModel`** — Entity-pair relation classification:
  1. Inserts typed marker tokens (e.g., `<SUBJ_anatomy>`, `<OBJ_disorder_present>`) around subject/object mentions
  2. Feeds through BERT, extracts subject/object token representations via index selection
  3. Concatenates representations, applies LayerNorm + dropout + linear classifier
  4. CrossEntropyLoss for training
- **`AlbertRelationModel`** — ALBERT variant
- **`BertForRelation`** / **`AlbertForRelation`** — Batch-efficient variants using `batched_index_select`

#### `relation/utils.py`
- `generate_relation_data()` — Reads Dataset with predicted/gold entities, generates all entity pair combinations per sentence, assigns gold relation labels. Creates samples with subject/object positions, types, and tokens
- `decode_sample_id()` — Parses composite sample IDs back into document, subject span, and object span

### Training Scripts

#### `run_entity.py`
- Full training and evaluation pipeline for entity extraction
- **Flow**: Load dataset → convert to span samples → train `EntityPredictionModel` with AdamW + linear warmup → evaluate F1 on dev set → save best model → output predictions with `predicted_ner` fields
- **Hyperparameters** (from `run_entity.sh`): lr=1e-5, task_lr=5e-4, batch=8, context_window=100

#### `run_relation.py`
- Full training and evaluation pipeline for relation extraction
- **Flow**: Load entity predictions → generate pair samples → add special marker tokens → train `RelationModel` → evaluate F1 → output predictions with `predicted_relations` fields
- **Key function**: `add_marker_tokens()` — Inserts NER-typed marker tokens around subject/object mentions
- **Hyperparameters** (from `run_relation.sh`): lr=5e-5, batch=128, epochs=20, context_window=100, max_seq_length=256

---

## Phase 3 — Full-Corpus Inference & Post-Processing

### `run_inference.sh`
Two-stage inference pipeline:
1. **Entity inference**: Runs `run_entity.py` on test data → produces entity predictions
2. **Relation inference**: Runs `run_relation.py` using entity predictions → produces full predictions

### `reverse_structure_data.py`
- **Location**: `src/ner/result/run_relation/`
- **Purpose**: Converts PURE's token-index prediction format back into human-readable entity/relation JSON
- **Key function**: `reverse()` — Reads JSONL predictions, reconstructs entity text from token spans, builds structured output
- **Connection**: Bridges NER output → KG construction input

### Data Flow

```
Raw CSV Reports
    │
    ▼
get_inference_data.py → test.json (PURE format)
    │
    ▼
run_inference.sh
    ├── Stage 1: run_entity.py → entity predictions
    └── Stage 2: run_relation.py → entity + relation predictions
                    │
                    ▼
         reverse_structure_data.py → human-readable JSON
                    │
                    ▼
            KG Construction Pipeline
```

---

## Phase 4 — Knowledge Graph Construction

**Location**: `src/kg_construct/code/`

Orchestrated by `auto_build_kg.sh`, a 6-step SLURM-based pipeline.

### Step 1: `get_entities.py` — Aggregate Entities & Relations

- **Purpose**: Collects all predicted entities and relations from the NER output
- **Key Functions**:
  - `get_entities()` — Collects entity frequency counts by type, resolves size entities, filters digit-containing non-size entities, writes entity CSV
  - `get_entity_type()` — For each entity, keeps only the most frequent type assignment; creates per-type CSVs
  - `get_relations()` — Aggregates relation triplets with counts, writes relations CSV
- **Output**: Entity CSVs (by type) and relation CSVs

### Step 2: `get_umls_entities.py` — UMLS Linking

- **Purpose**: Links extracted entities to the **Unified Medical Language System (UMLS)** using scispaCy and QuickUMLS
- **Key Function**: `get_umls()` — For each entity, runs `en_core_sci_lg` + QuickUMLS to find CUI, name, aliases, TUI, definition
- **Output**: `entities_umls.json`

### Step 3: `filter_cui.py` — Filter Best CUIs

- **Purpose**: Filters UMLS linkages to keep the best-matching CUI per entity
- **Key Function**: `filter_cui()` — For each entity, checks:
  1. Exact match with CUI name
  2. Appears in CUI aliases
  3. Falls back to top-ranked candidate
- **Output**: `entities_umls_filtered.json` and `entities_umls_filtered.csv`

### Step 4a: `structure_entities.py` — Decompose Entities

- **Purpose**: The most complex step — decomposes entities into **isolated** (atomic) and **composed** (multi-word) entities
- **Key Functions**:
  - `entities_by_word_count()` — Groups entities by word count (1-word, 2-word, etc.)
  - `get_isolated_composed()` — Hierarchically processes entities:
    - 1-word entities → isolated
    - 2+ word entities → decomposed into constituent isolated entities
    - Example: "right pleural effusion" → ["right", "pleural effusion"]
    - Uses combinatorial splitting patterns
  - `refine_composed()` — Removes "A of B" entities where both A and B exist independently; merges by CUI
  - `structure_entities()` — Converts CSV → structured JSON with UMLS aliases
  - `get_composed_entities()` — Maps composed entities to constituent CUI lists, groups by type
  - `add_umls_to_entities()` — Adds CUI, Name, Possibility columns to entity CSV

### Step 4b: `merge_entities.py` — Embedding-Based Deduplication

- **Purpose**: Merges semantically similar entities using embedding cosine similarity
- **Key Functions**:
  - `get_embeddings()` — Computes embeddings for all entity aliases using `MedCPT-Query-Encoder` or `BioLORD-2023-C`
  - `merge_entities()` — Builds cosine similarity matrix, merges entities above threshold (default **0.95**), combines aliases and counts
- **Output**: Deduplicated entity list

### Step 5: `get_kg_nodes.py` — Build Final KG

- **Purpose**: Builds the final knowledge graph structure files
- **Key Functions**:
  - `get_kg_nodes()` — Creates `kg_nodes.json`: maps CUI → entity with name, definition, TUI, aliases, type, count
  - `get_kg_subgraphs()` — Creates `kg_subgraphs.json`: composed entities as subgraphs of multiple KG nodes
  - `get_kg_relations()` — Creates `kg_relations.csv`: maps source/target entities to KG node indices, aggregates counts, deduplicates
  - `merge_symmetric_relations()` — Merges bidirectional relation pairs

### Step 6: `get_size_relations.py` — Size/Measurement Edges

- **Purpose**: Extracts size/measurement relations (e.g., "3mm" → "nodule")
- **Key Function**: `get_size_relations()` — Finds relations where source is a measurement string, maps target to CUI, aggregates by target CUI

### KG Construction Data Flow

```
NER Predictions (from Phase 3)
    │
    ▼
Step 1: get_entities.py
    │   → entity CSVs (by type), relation CSVs
    ▼
Step 2: get_umls_entities.py
    │   → entities_umls.json (UMLS-linked)
    ▼
Step 3: filter_cui.py
    │   → entities_umls_filtered.json (best CUI per entity)
    ▼
Step 4a: structure_entities.py
    │   → isolated + composed entity decomposition
    │
Step 4b: merge_entities.py
    │   → embedding-based deduplication (cosine sim ≥ 0.95)
    ▼
Step 5: get_kg_nodes.py
    │   → kg_nodes.json, kg_relations.csv, kg_subgraphs.json
    ▼
Step 6: get_size_relations.py
        → size/measurement relation edges
```

---

## Output Files

### `kg_nodes.json` (~22,837 nodes)

Each node contains:

| Field | Description |
|-------|-------------|
| CUI | UMLS Concept Unique Identifier |
| Name | Canonical entity name |
| Definition | UMLS definition text |
| TUI | UMLS Type Unique Identifier |
| Aliases | List of alternative names |
| entity_type | Category (`anatomy`, `disorder`, `concept`, etc.) |
| count | Frequency in corpus |

**Top nodes by frequency**: Lung (81K), Chest (50K), Pleura (49K), Effusion (44K)

### `kg_relations.csv` (~21,626 relations)

| Column | Description |
|--------|-------------|
| source_index | Index of source node in kg_nodes |
| target_index | Index of target node in kg_nodes |
| source_name | Source entity name |
| target_name | Target entity name |
| relation | Relation type (`modify`, `located_at`, `suggestive_of`) |
| count | Frequency in corpus |

**Top relation**: effusion → pleural (`located_at`, 20,502 occurrences)

### `kg_subgraphs.json`

Compound/composed entities linking to their constituent KG nodes:
- Example: "picc line" = C0179740 + C0205132

---

## Key Design Decisions

1. **Two-model pipeline (PURE)** — Entity extraction first, then relation classification on predicted entities. Avoids joint modeling complexity and allows independent optimization of each component.

2. **GPT-4 as annotator** — Instead of expensive manual annotation, GPT-4o generates silver-standard training data with few-shot prompts. This data is then used to train lightweight BiomedBERT models for scalable corpus-wide inference.

3. **UMLS grounding** — Every entity is linked to a UMLS Concept Unique Identifier (CUI), making the knowledge graph interoperable with standard medical ontologies and enabling cross-system comparisons.

4. **Embedding-based deduplication** — Uses medical domain embeddings (MedCPT-Query-Encoder / BioLORD-2023-C) with a high similarity threshold (0.95) to merge near-duplicate entities without losing meaningful distinctions.

5. **Composed entity decomposition** — Multi-word entities like "right pleural effusion" are split into atomic constituents and stored as subgraphs, preserving both granular and compositional semantics in the KG.

6. **BiomedBERT backbone** — Uses `microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext`, a domain-specific BERT model pre-trained on biomedical literature, for superior performance on medical NER tasks.

---

## Dependencies & Environment

- **Python**: 3.8.18
- **Deep Learning**: PyTorch, AllenNLP 2.2.0
- **NLP**: Transformers (HuggingFace), scispaCy, QuickUMLS
- **Backbone Model**: BiomedBERT (`microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext`)
- **Embedding Models**: MedCPT-Query-Encoder, BioLORD-2023-C
- **API**: OpenAI GPT-4o (for data annotation)
- **Infra**: SLURM-based job scheduling for training and KG construction

Install via: `conda env create -f environment.yml`
