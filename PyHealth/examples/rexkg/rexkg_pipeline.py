"""ReXKG pipeline example for PyHealth + original ReXKG scripts.

Notebook-first usage:
    PyHealth/examples/rexkg/rexkg_pipeline.ipynb

This example mirrors the project RUN_GUIDE and adds a PyHealth entry point using
RexKGDataset and RexKG tasks.

Usage examples:

1) Build PyHealth sample dataset only:
    python PyHealth/examples/rexkg/rexkg_pipeline_example.py \
        --mode pyhealth \
        --rexkg-data-root src/ner/data

2) Run smoke shell pipeline (tiny data + local scripts):
    python PyHealth/examples/rexkg/rexkg_pipeline_example.py \
        --mode smoke \
        --run-commands

3) Run full shell pipeline (entity/relation + KG):
    python PyHealth/examples/rexkg/rexkg_pipeline_example.py \
        --mode full \
        --run-commands

By default, commands are printed but not executed. Use --run-commands to execute.
"""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path
from typing import Iterable, List


def run_commands(commands: Iterable[str], cwd: Path, execute: bool) -> None:
    """Print or execute shell commands in order."""
    for cmd in commands:
        print(f"$ {cmd}")
        if execute:
            subprocess.run(cmd, cwd=str(cwd), shell=True, check=True)


def pyhealth_phase(data_root: Path) -> None:
    """Build RexKG SampleDataset through PyHealth integration."""
    from pyhealth.datasets import RexKGDataset
    from pyhealth.tasks import (
        RexKGEntityExtractionRadiology,
        RexKGKnowledgeGraphConstruction,
        RexKGRelationExtractionRadiology,
    )

    dataset = RexKGDataset(root=str(data_root))
    print("Loaded dataset:", dataset.dataset_name)
    dataset.stats()

    entity_samples = dataset.set_task(RexKGEntityExtractionRadiology())
    print(f"Entity samples: {len(entity_samples)}")

    relation_samples = dataset.set_task(RexKGRelationExtractionRadiology())
    print(f"Relation samples: {len(relation_samples)}")

    kg_samples = dataset.set_task(RexKGKnowledgeGraphConstruction())
    print(f"KG samples: {len(kg_samples)}")

    if len(entity_samples) > 0:
        first = entity_samples[0]
        print("First entity sample keys:", sorted(first.keys()))
        print("First sample text chars:", len(first.get("text", "")))


def smoke_pipeline_commands(project_root: Path) -> List[tuple[Path, List[str]]]:
    """Return command groups for the RUN_GUIDE smoke workflow."""
    ner_data = project_root / "src" / "ner" / "data"
    ner_root = project_root / "src" / "ner"
    relation_dir = ner_root / "result" / "run_relation"
    kg_code = project_root / "src" / "kg_construct" / "code"

    groups = [
        (
            ner_data,
            [
                "python structure_data.py",
                "cd data_split && head -n 200 train.json > train_tiny.json && head -n 60 test.json > test_tiny.json",
            ],
        ),
        (
            ner_root,
            [
                "PYTHON_BIN=$PWD/../../.venv/bin/python MODEL=prajjwal1/bert-tiny OUTPUT_DIR=./result/run_entity_local_smoke TRAIN_DATA=./data/data_split/train_tiny.json DEV_DATA=./data/data_split/test_tiny.json TEST_DATA=./data/data_split/test_tiny.json sh run_entity.sh",
                "cp ./result/run_entity_local_smoke/ent_pred_mimic_headct.json ./result/run_entity_local_smoke/ent_pred_dev.json",
                "cp ./result/run_entity_local_smoke/ent_pred_mimic_headct.json ./result/run_entity_local_smoke/ent_pred_test.json",
                "PYTHON_BIN=$PWD/../../.venv/bin/python MODEL=bert-base-uncased OUTPUT_DIR=./result/run_relation_local_smoke TRAIN_FILE=./data/data_split/train_tiny.json ENTITY_OUTPUT_DIR=./result/run_entity_local_smoke ENTITY_PRED_DEV=ent_pred_dev.json ENTITY_PRED_TEST=ent_pred_test.json sh run_relation.sh",
            ],
        ),
        (
            relation_dir,
            [
                "python reverse_structure_data.py --input_json_file ../run_relation_local_smoke/predictions.json --save_json_file ../../data/local_smoke_for_kg.json",
            ],
        ),
        (
            kg_code,
            [
                "python get_entities.py --ent_pred_mimic_headct ../../ner/data/local_smoke_for_kg.json --ent_real_pred_mimic_headct ../../ner/data/local_smoke_for_kg.json --save_entity_dir ../result/local_run_smoke/entities --save_real_dir ../result/local_run_smoke/relation",
                "python get_umls_entities.py --save_entity_dir ../result/local_run_smoke/entities",
                "python filter_cui.py --save_entity_dir ../result/local_run_smoke/entities",
                "python structure_entities.py --save_entity_dir ../result/local_run_smoke/entities --ignore_count 10",
                "python get_kg_nodes.py --save_entity_dir ../result/local_run_smoke/entities --save_real_dir ../result/local_run_smoke/relation --save_kg_dir ../result/local_run_smoke/kg",
                "python get_size_relations.py --entity_dir ../result/local_run_smoke/entities --real_dir ../result/local_run_smoke/relation",
            ],
        ),
    ]
    return groups


def full_pipeline_commands(project_root: Path) -> List[tuple[Path, List[str]]]:
    """Return command groups for the RUN_GUIDE full workflow."""
    ner_data = project_root / "src" / "ner" / "data"
    ner_root = project_root / "src" / "ner"
    relation_dir = ner_root / "result" / "run_relation"
    kg_code = project_root / "src" / "kg_construct" / "code"

    groups = [
        (
            ner_data,
            [
                "python structure_data.py",
                "python get_inference_data.py --input_csv_file ../../data/chexpert_plus/df_chexpert_plus_onlyfindings.csv --save_json_file ./chexpert_plus_groundtruth.json",
            ],
        ),
        (
            ner_root,
            [
                "sh run_entity.sh",
                "sh run_relation.sh",
                "sh run_inference.sh",
            ],
        ),
        (
            relation_dir,
            [
                "python reverse_structure_data.py --input_json_file ../run_relation/predictions.json --save_json_file ../../data/your_test_file.json",
            ],
        ),
        (
            kg_code,
            [
                "python get_entities.py --ent_pred_mimic_headct ../../ner/data/your_test_file.json --ent_real_pred_mimic_headct ../../ner/data/your_test_file.json --save_entity_dir ../result/your_run/entities --save_real_dir ../result/your_run/relation",
                "python get_umls_entities.py --save_entity_dir ../result/your_run/entities",
                "python filter_cui.py --save_entity_dir ../result/your_run/entities",
                "python structure_entities.py --save_entity_dir ../result/your_run/entities --ignore_count 10",
                "python get_kg_nodes.py --save_entity_dir ../result/your_run/entities --save_real_dir ../result/your_run/relation --save_kg_dir ../result/your_run/kg",
                "python get_size_relations.py --entity_dir ../result/your_run/entities --real_dir ../result/your_run/relation",
            ],
        ),
    ]
    return groups


def main() -> None:
    parser = argparse.ArgumentParser(description="ReXKG pipeline example")
    parser.add_argument(
        "--mode",
        choices=["pyhealth", "smoke", "full"],
        default="pyhealth",
        help="Pipeline mode: PyHealth integration only, smoke shell pipeline, or full shell pipeline.",
    )
    parser.add_argument(
        "--run-commands",
        action="store_true",
        help="Execute shell commands. If omitted, commands are printed only.",
    )
    parser.add_argument(
        "--rexkg-data-root",
        default="src/ner/data",
        help="Path to ReXKG raw data directory for RexKGDataset (default: src/ner/data).",
    )

    args = parser.parse_args()

    # .../PyHealth/examples/rexkg/rexkg_pipeline_example.py -> repo root
    project_root = Path(__file__).resolve().parents[3]

    if args.mode == "pyhealth":
        data_root = Path(args.rexkg_data_root)
        if not data_root.is_absolute():
            data_root = project_root / data_root
        pyhealth_phase(data_root=data_root)
        return

    if args.mode == "smoke":
        groups = smoke_pipeline_commands(project_root)
    else:
        groups = full_pipeline_commands(project_root)

    print(f"Mode: {args.mode}")
    print(f"Project root: {project_root}")
    if not args.run_commands:
        print("Dry run: showing commands only. Use --run-commands to execute.")

    for cwd, command_group in groups:
        print(f"\n### CWD: {cwd}")
        run_commands(command_group, cwd=cwd, execute=args.run_commands)

    print("\nPipeline example finished.")


if __name__ == "__main__":
    main()
