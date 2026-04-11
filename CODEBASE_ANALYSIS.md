# ReXKG — Codebase Analysis (Current Structure)

**Last updated:** April 2026  
**Project:** `ReXKG-main`  
**Goal:** turn radiology report text into a structured knowledge graph via report preprocessing, entity/relation extraction, and KG construction.

---

## 1) Executive summary

The repository is best understood as a **three-layer pipeline**:

1. **Input data** — raw report CSVs and prepared JSON/JSONL files
2. **Processing layer** — NER/relation extraction plus KG construction scripts
3. **Output data** — model predictions, intermediate entity/relation tables, and final KG files

### One-screen flow

```text
src/data/chexpert_plus/*.csv
    ↓
src/ner/data/{get_inference_data.py, structure_data.py}
    ↓
src/ner/data/*.json + src/ner/data/data_split/*.json
    ↓
src/ner/{run_entity.py, run_relation.py, run_inference.sh}
    ↓
src/ner/result/run_entity*/ent_pred*.json
src/ner/result/run_relation*/{predictions.json, ent_rel_pred_test.json}
    ↓
src/ner/result/run_relation/reverse_structure_data.py
    ↓
src/ner/data/*_for_kg.json
    ↓
src/kg_construct/code/*.py
    ↓
src/kg_construct/result/<run_name>/{entities, relation, kg}
```

---

## 2) Current repository layout

```text
ReXKG-main/
├── README.md
├── RUN_GUIDE.md
├── environment.yml
├── src/
│   ├── data/
│   │   └── chexpert_plus/
│   │       ├── df_chexpert_plus_onlyfindings.csv
│   │       ├── df_chexpert_plus_200401_withfindings.csv
│   │       └── df_chexpert_plus_240401.csv
│   ├── ner/
│   │   ├── data/
│   │   ├── entity/
│   │   ├── relation/
│   │   ├── shared/
│   │   ├── run_entity.py
│   │   ├── run_entity.sh
│   │   ├── run_relation.py
│   │   ├── run_relation.sh
│   │   ├── run_inference.sh
│   │   └── result/
│   └── kg_construct/
│       ├── code/
│       └── result/
```

### Practical meaning of the main folders

| Folder | Role |
|---|---|
| `src/data/chexpert_plus/` | **Raw input data** expected by the repo’s default paths |
| `src/ner/data/` | **Preprocessing workspace**: scripts + prepared JSON/JSONL |
| `src/ner/entity/`, `src/ner/relation/`, `src/ner/shared/` | **Model implementation layer** |
| `src/ner/result/` | **Model outputs** for training, inference, and smoke runs |
| `src/kg_construct/code/` | **Knowledge-graph construction layer** |
| `src/kg_construct/result/` | **Final and example KG outputs** |

---

## 3) Input data layer

This repo uses both **raw CSV input** and **prepared JSON/JSONL input**.

### 3.1 Raw input data

**Primary input location:** `src/data/chexpert_plus/`

Current files present:

- `df_chexpert_plus_onlyfindings.csv`
- `df_chexpert_plus_200401_withfindings.csv`
- `df_chexpert_plus_240401.csv`

### Expected CSV schema for preprocessing

`src/ner/data/get_inference_data.py` expects the following default columns:

- `path_to_image` → report/study identifier
- `section_findings` → free-text findings section

If you start with a new CSV, it should be placed under:

```text
src/data/chexpert_plus/
```

and should follow that schema unless you override the script arguments.

### 3.2 Prepared input data

`src/ner/data/` already contains several important derived inputs:

| File / Folder | Type | Used for |
|---|---|---|
| `chexpert_plus_groundtruth.json` | JSONL | inference input for `run_inference.sh` |
| `data_split/train.json` | JSONL | entity/relation training |
| `data_split/test.json` | JSONL | entity/relation evaluation |
| `data_split/train_micro.json`, `train_tiny.json`, `test_tiny.json` | JSONL | quick smoke runs |
| `gpt4_entities_*.json` | JSON | optional GPT-based silver annotation artifacts |
| `local_smoke_for_kg.json` | JSON | smoke-run handoff into KG construction |

### Input guidance by starting point

- **If you only have raw reports:** start from `src/data/chexpert_plus/*.csv`
- **If you already have PURE-format JSONL:** start from `src/ner/data/data_split/*.json`
- **If you already have relation predictions:** start from `reverse_structure_data.py` and then the KG scripts

---

## 4) Processing layer

The processing layer is split into **five practical stages**.

### Stage A — Data preparation (`src/ner/data/`)

This stage converts raw reports or GPT-labeled data into the JSON/JSONL format consumed by the models.

#### Main scripts

- `get_inference_data.py`
  - **Input:** raw CSV from `src/data/chexpert_plus/`
  - **Output:** inference-ready JSONL such as `chexpert_plus_groundtruth.json`

- `structure_data.py`
  - **Input:** GPT-generated annotation JSON
  - **Output:** PURE-style train/test JSONL under `data_split/`

- `gpt4_entity_extraction.py` and `gpt4_relation_extraction.py`
  - **Role:** optional silver-label generation for reproduction of the annotation pipeline
  - **Note:** these are not required for every local run if prepared JSON files already exist

### Stage B — Entity extraction (`src/ner/`)

#### Main entrypoints

- `run_entity.py`
- `run_entity.sh`

#### What it does

Trains or evaluates the entity model and writes entity predictions under:

```text
src/ner/result/run_entity/
src/ner/result/run_entity_local_smoke/
```

#### Typical outputs

- `ent_pred_mimic_headct.json`
- `ent_pred_dev.json` / `ent_pred_test.json` (when explicitly requested)
- `config.json`
- `model.safetensors`
- `train.log`

### Stage C — Relation extraction (`src/ner/`)

#### Main entrypoints

- `run_relation.py`
- `run_relation.sh`
- `run_inference.sh` (two-stage inference wrapper)

#### What it does

Consumes entity predictions and predicts relations between extracted entities.

#### Important current behavior

- `run_relation.sh` uses the entity output directory under `src/ner/result/run_entity/`
- `run_inference.sh` is the correct inference wrapper for this repo
- the relation stage is configured with `--no_cuda`, so local runs are CPU-safe by default

#### Typical outputs

Under:

```text
src/ner/result/run_relation/
src/ner/result/run_relation_local_smoke/
```

you will see files such as:

- `predictions.json`
- `ent_rel_pred_test.json`
- `label_list.json`
- `special_tokens.json`
- `pytorch_model.bin` or local checkpoint files
- `train.log`

### Stage D — Reverse structuring for KG input

**Script:** `src/ner/result/run_relation/reverse_structure_data.py`

This script converts relation model output into the simplified JSON format expected by the KG construction scripts.

#### Examples

- smoke run output → `src/ner/data/local_smoke_for_kg.json`
- full run output → a custom `*_for_kg.json` file in `src/ner/data/`

### Stage E — Knowledge graph construction (`src/kg_construct/code/`)

This is the graph-building layer. It starts from postprocessed entity/relation JSON and ends in final KG files.

#### Main scripts and roles

| Script | Role |
|---|---|
| `get_entities.py` | aggregate entities and relations from model output |
| `get_umls_entities.py` | link entities to UMLS candidates |
| `filter_cui.py` | keep the best CUI match |
| `structure_entities.py` | organize isolated/composed entities |
| `merge_entities.py` | optional semantic dedup/merge helper |
| `get_kg_nodes.py` | build KG nodes, subgraphs, and relation tables |
| `get_size_relations.py` | add size/measurement relation outputs |

#### Important note

`auto_build_kg.sh` is a **cluster/SLURM-oriented template**. It is useful as a reference, but the **current local run style** is to execute the Python scripts directly, as shown in `RUN_GUIDE.md`.

---

## 5) Output data layer

There are two main output zones.

### 5.1 Model outputs (`src/ner/result/`)

This folder stores NER and relation model artifacts.

#### Current subfolders visible in the repo

- `run_entity/`
- `run_entity_local_smoke/`
- `run_relation/`
- `run_relation_local_smoke/`

#### What belongs here

| Output type | Examples |
|---|---|
| Entity predictions | `ent_pred_mimic_headct.json`, `ent_pred_test.json` |
| Relation predictions | `predictions.json`, `ent_rel_pred_test.json` |
| Model checkpoints | `model.safetensors`, `pytorch_model.bin` |
| Logs/config | `train.log`, `config.json`, tokenizer files |

### 5.2 KG outputs (`src/kg_construct/result/`)

This folder stores the graph build artifacts.

#### Current subfolders visible in the repo

- `chexpert_plus_part1/`
- `local_run/`
- `local_run_smoke/`

#### Typical final outputs per run

```text
src/kg_construct/result/<run_name>/
├── entities/
├── relation/
└── kg/
    ├── kg_nodes.json
    ├── kg_relations.csv
    └── kg_subgraphs.json
```

#### Meaning of the main files

| File | Meaning |
|---|---|
| `kg_nodes.json` | canonical node list for the graph |
| `kg_relations.csv` | edge table with relation types and counts |
| `kg_subgraphs.json` | composed entities mapped to constituent nodes |
| `relation/size_relations.csv` | size-specific relation output |

---

## 6) Current end-to-end path (input → processing → output)

### Path 1 — Start from raw CSV reports

```text
Input:
  src/data/chexpert_plus/df_chexpert_plus_onlyfindings.csv

Processing:
  src/ner/data/get_inference_data.py
  → src/ner/run_inference.sh
  → src/ner/result/run_relation/reverse_structure_data.py
  → src/kg_construct/code/*.py

Output:
  src/ner/result/run_entity*/...
  src/ner/result/run_relation*/...
  src/kg_construct/result/<run_name>/kg/*
```

### Path 2 — Start from prepared train/test JSONL

```text
Input:
  src/ner/data/data_split/train.json
  src/ner/data/data_split/test.json

Processing:
  src/ner/run_entity.sh
  → src/ner/run_relation.sh

Output:
  src/ner/result/run_entity/*
  src/ner/result/run_relation/*
```

### Path 3 — Start from relation predictions and only build the KG

```text
Input:
  src/ner/result/run_relation*/predictions.json
  or ent_rel_pred_test.json

Processing:
  reverse_structure_data.py
  → kg_construct/code/*.py

Output:
  src/kg_construct/result/<run_name>/entities/*
  src/kg_construct/result/<run_name>/relation/*
  src/kg_construct/result/<run_name>/kg/*
```

---

## 7) Recommended operational interpretation of the repo

### If your goal is training / experimentation

Work mainly in:

- `src/ner/data/`
- `src/ner/run_entity.py`
- `src/ner/run_relation.py`
- `src/ner/result/`

### If your goal is graph building / downstream analysis

Work mainly in:

- `src/ner/result/`
- `src/ner/result/run_relation/reverse_structure_data.py`
- `src/kg_construct/code/`
- `src/kg_construct/result/`

### If your goal is quick validation on a laptop

Use the **smoke-run path** already reflected in the repo:

- `data_split/train_tiny.json`
- `data_split/test_tiny.json`
- `run_entity_local_smoke/`
- `run_relation_local_smoke/`
- `kg_construct/result/local_run_smoke/`

---

## 8) Alignment notes / current realities

1. **Preferred raw data location:** `src/data/chexpert_plus/`
2. **Correct inference wrapper:** `src/ner/run_inference.sh`
3. **`src/ner/data/` is mixed-purpose:** it contains both scripts and data artifacts
4. **`auto_build_kg.sh` is not the best local entrypoint:** it assumes `srun`; for local runs, use the Python commands from `RUN_GUIDE.md`
5. **Outputs are already separated cleanly:** model artifacts under `src/ner/result/`, graph artifacts under `src/kg_construct/result/`

---

## 9) Bottom line

The current codebase is aligned around this simple mental model:

- **Input data** lives in `src/data/chexpert_plus/` and `src/ner/data/`
- **Processing** happens in `src/ner/` first, then `src/kg_construct/code/`
- **Outputs** land in `src/ner/result/` and `src/kg_construct/result/`

That is the correct current structure to use when navigating, documenting, or extending the repo.
