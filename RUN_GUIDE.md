# ReXKG Run Guide (End-to-End)

This guide gives a **light but complete** path to run the whole project.

- Paper: *Uncovering Knowledge Gaps in Radiology Report Generation Models through Knowledge Graphs*
- Codebase: https://github.com/rajpurkarlab/ReXKG

---

## 1) Data Access

Use official sources:

- CheXpert Plus (Stanford AIMI):
  https://stanfordaimi.azurewebsites.net/datasets/5158c524-d3ab-4e02-96e9-6ee9efc110a1
- MIMIC-CXR v2.0.0 (PhysioNet):
  https://physionet.org/content/mimic-cxr/2.0.0/

> Notes:
> - PhysioNet data is credentialed (DUA + required training).
> - Keep data under `src/data/chexpert_plus/` for this repo’s default paths.

---

## 2) Environment Setup

From repo root:

```bash
cd /path/to/ReXKG-main
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

Install core runtime dependencies used by local runs:

```bash
pip install torch transformers pandas tqdm scikit-learn neraug protobuf sentencepiece tiktoken
```

Optional (for true UMLS linking path):

```bash
pip install spacy scispacy quickumls
```

> If `quickumls` fails to install on your platform, you can still run a local smoke KG build (without true UMLS linking).

---

## 3) Pipeline Overview

End-to-end flow:

1. Prepare/annotate data (`src/ner/data/`)
2. Train/infer entities + relations (`src/ner/`)
3. Post-process relation output (`src/ner/result/run_relation/reverse_structure_data.py`)
4. Build KG (`src/kg_construct/code/`)

---

## 4) Full Pipeline Commands

## 4.1 Data preparation

```bash
cd src/ner/data
python structure_data.py
python get_inference_data.py \
  --input_csv_file ../../data/chexpert_plus/df_chexpert_plus_onlyfindings.csv \
  --save_json_file ./chexpert_plus_groundtruth.json
```

Expected outputs:

- `src/ner/data/data_split/train.json`
- `src/ner/data/data_split/test.json`
- `src/ner/data/chexpert_plus_groundtruth.json`

## 4.2 Entity + Relation model run

From `src/ner`:

### A) Original training scripts

```bash
cd src/ner
sh run_entity.sh
sh run_relation.sh
```

### B) Two-stage inference script

```bash
cd src/ner
sh run_inference.sh
```

Expected outputs under `src/ner/result/`:

- Entity predictions (e.g., `run_entity/ent_pred_*.json`)
- Relation predictions (e.g., `run_relation/*.json`)

## 4.3 Convert relation output for KG construction

```bash
cd src/ner/result/run_relation
python reverse_structure_data.py \
  --input_json_file ../run_relation/predictions.json \
  --save_json_file ../../data/your_test_file.json
```

Expected output:

- `src/ner/data/your_test_file.json`

## 4.4 Build KG

```bash
cd src/kg_construct/code
python get_entities.py \
  --ent_pred_mimic_headct ../../ner/data/your_test_file.json \
  --ent_real_pred_mimic_headct ../../ner/data/your_test_file.json \
  --save_entity_dir ../result/your_run/entities \
  --save_real_dir ../result/your_run/relation

python get_umls_entities.py --save_entity_dir ../result/your_run/entities
python filter_cui.py --save_entity_dir ../result/your_run/entities
python structure_entities.py --save_entity_dir ../result/your_run/entities --ignore_count 10
python get_kg_nodes.py \
  --save_entity_dir ../result/your_run/entities \
  --save_real_dir ../result/your_run/relation \
  --save_kg_dir ../result/your_run/kg
python get_size_relations.py \
  --entity_dir ../result/your_run/entities \
  --real_dir ../result/your_run/relation
```

Final KG artifacts:

- `src/kg_construct/result/your_run/kg/kg_nodes.json`
- `src/kg_construct/result/your_run/kg/kg_subgraphs.json`
- `src/kg_construct/result/your_run/kg/kg_relations.csv`
- `src/kg_construct/result/your_run/relation/size_relations.csv`

---

## 5) Fast Local Smoke Run (Recommended First)

Use this to validate your environment quickly before full-scale training.

## 5.1 Build tiny train/test

```bash
cd src/ner/data
python structure_data.py
cd data_split
head -n 200 train.json > train_tiny.json
head -n 60 test.json > test_tiny.json
```

## 5.2 Run local NER + relation smoke

```bash
cd ../../
PYTHON_BIN=$PWD/../../.venv/bin/python \
MODEL=prajjwal1/bert-tiny \
OUTPUT_DIR=./result/run_entity_local_smoke \
TRAIN_DATA=./data/data_split/train_tiny.json \
DEV_DATA=./data/data_split/test_tiny.json \
TEST_DATA=./data/data_split/test_tiny.json \
sh run_entity.sh

cp ./result/run_entity_local_smoke/ent_pred_mimic_headct.json ./result/run_entity_local_smoke/ent_pred_dev.json
cp ./result/run_entity_local_smoke/ent_pred_mimic_headct.json ./result/run_entity_local_smoke/ent_pred_test.json

PYTHON_BIN=$PWD/../../.venv/bin/python \
MODEL=bert-base-uncased \
OUTPUT_DIR=./result/run_relation_local_smoke \
TRAIN_FILE=./data/data_split/train_tiny.json \
ENTITY_OUTPUT_DIR=./result/run_entity_local_smoke \
ENTITY_PRED_DEV=ent_pred_dev.json \
ENTITY_PRED_TEST=ent_pred_test.json \
sh run_relation.sh
```

## 5.3 Convert to KG input and build smoke KG

```bash
cd result/run_relation
python reverse_structure_data.py \
  --input_json_file ../run_relation_local_smoke/predictions.json \
  --save_json_file ../../data/local_smoke_for_kg.json

cd ../../../kg_construct/code
python get_entities.py \
  --ent_pred_mimic_headct ../../ner/data/local_smoke_for_kg.json \
  --ent_real_pred_mimic_headct ../../ner/data/local_smoke_for_kg.json \
  --save_entity_dir ../result/local_run_smoke/entities \
  --save_real_dir ../result/local_run_smoke/relation
```

Then run KG steps from section 4.4 using `../result/local_run_smoke/...` paths.

---

## 6) Troubleshooting

- **`srun` not found**: use local shell scripts/commands directly (non-cluster mode).
- **CUDA errors**: run CPU-only; reduce batch sizes.
- **Transformers import/tokenizer issues**: install/upgrade `transformers`, `protobuf`, `sentencepiece`, `tiktoken`.
- **`quickumls` install failure**: skip true UMLS linking for smoke run, or use a Linux/conda env that supports it.
- **OOM/slow relation stage**: use micro/tiny train files first.

---

## 7) Quick Verification Checklist

After a successful run, verify:

```bash
ls -lh src/ner/result/run_entity*/ent_pred*.json
ls -lh src/ner/result/run_relation*/predictions.json
ls -lh src/kg_construct/result/*/kg/kg_nodes.json
ls -lh src/kg_construct/result/*/kg/kg_relations.csv
```

If all files exist and are non-empty, your pipeline is wired correctly.
