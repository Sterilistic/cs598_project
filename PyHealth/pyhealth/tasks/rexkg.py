# Description: ReXKG entity and relation extraction tasks for radiology reports

import logging
import argparse
import importlib
import csv
import json
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Union, Type, Any, Optional, Tuple, TYPE_CHECKING

import numpy as np
import pandas as pd
import polars as pl
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset as TorchDataset
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AutoModelForTokenClassification,
)
from transformers.optimization import get_linear_schedule_with_warmup

from pyhealth.data.data import Patient
from pyhealth.processors import TextProcessor, SequenceProcessor
from .base_task import BaseTask

import openai

if TYPE_CHECKING:
    from ..datasets.rexkg import RexKGCheXpertDataset, RexKGDataset

logger = logging.getLogger(__name__)

# gpt loop counter to track gpt inputs lefft to send
counter = 0


class _TokenClsJsonDataset(TorchDataset):
    def __init__(self, encodings: Dict[str, List[List[int]]], labels: List[List[int]]):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


class _RelationClsDataset(TorchDataset):
    def __init__(self, encodings: Dict[str, List[List[int]]], labels: List[int]):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


class RexKGEntityExtractionRadiology(BaseTask):
    """Entity extraction task for radiology reports using ReXKG.

    This task extracts named entities from radiology reports for knowledge graph
    construction. It recognizes 8 entity types commonly found in radiology text:
    anatomy, disorder_present, disorder_notpresent, concept, procedures,
    devices_present, devices_notpresent, and size measurements.

    The task processes clinical radiology notes and produces entity annotations
    that can be used for:
    - Knowledge graph construction
    - Clinical information extraction
    - Structured data generation from unstructured reports
    - Named entity recognition model training/evaluation

    Entity Types:
        - anatomy: Anatomical structures and body regions (e.g., "right lung")
        - disorder_present: Confirmed pathological findings (e.g., "pneumonia")
        - disorder_notpresent: Ruled-out or excluded disorders (e.g., "no effusion")
        - concept: Medical/clinical concepts and observations
        - procedures: Medical procedures and interventions
        - devices_present: Medical devices that are present
        - devices_notpresent: Medical devices that are absent
        - size: Size, dimension, or measurement terms (e.g., "3cm")

    Input:
        Radiology reports in structured format with:
        - text: Free-text radiology report content
        - study_id: Identifier for the imaging study
        - report_type: Type of radiology report (e.g., "CHEST", "ABDOMEN")

    Output:
        - entities: List of extracted entities with type and position information
        - entity_labels: BIO-tagged sequence of entity types

    Args:
        task_name: Name identifying this task
        input_schema: Schema defining input feature types
        output_schema: Schema defining output label types

    Examples:
        >>> from pyhealth.datasets import SampleDataset
        >>> from pyhealth.tasks import RexKGEntityExtractionRadiology
        >>> # Create dataset with radiology reports
        >>> dataset = SampleDataset(...)
        >>> task = RexKGEntityExtractionRadiology()
        >>> samples = dataset.set_task(task)
        >>> # Each sample contains extracted entities from a report
    """

    task_name: str = "rexkg_entity_extraction_radiology"
    input_schema: Dict[str, Union[str, Type]] = {"text": TextProcessor}
    output_schema: Dict[str, Union[str, Type]] = {"entities": SequenceProcessor}

    ENTITY_TYPES = [
        "O",
        "B-anatomy", "I-anatomy",
        "B-disorder_present", "I-disorder_present",
        "B-disorder_notpresent", "I-disorder_notpresent",
        "B-concept", "I-concept",
        "B-procedures", "I-procedures",
        "B-devices_present", "I-devices_present",
        "B-devices_notpresent", "I-devices_notpresent",
        "B-size", "I-size",
    ]

    def pre_filter(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """Filter patients with radiology reports.

        Args:
            df: Lazy polars dataframe of events

        Returns:
            Filtered dataframe containing only patients with radiology text
        """
        # Filter to patients that have radiology report text events
        filtered_df = df.filter(
            pl.col("patient_id").is_in(
                df.filter(pl.col("event_type") == "radiology_reports")
                .select("patient_id")
                .unique()
                .collect()
                .to_series()
            )
        )
        return filtered_df

    def __call__(self, patient: Patient) -> List[Dict]:
        """Extract entities and relations from a patient's radiology reports.

        Processes all radiology reports for a patient and extracts named entities
        that can be used for knowledge graph construction.

        Args:
            patient: Patient object containing radiology report events

        Returns:
            List of samples, each containing:
            - text: Original radiology report text
            - entities: Extracted entity annotations
            - study_id: Associated imaging study identifier
            - report_type: Type of radiology report
        """
        samples = []

        # Get radiology report events
        reports = patient.get_events(event_type="radiology_reports")
        
        if not reports:
            return samples

        for report in reports:
            text = getattr(report, "text", "")
            
            # Skip empty reports
            if not text or text.strip() == "":
                continue

            # Extract text and basic metadata
            sample = {
                "patient_id": patient.patient_id,
                "text": text,
                "entities": [],  # Would be populated by ReXKG NER model
                "report_type": getattr(report, "report_type", "RADIOLOGY"),
            }

            # Include study ID if available
            if hasattr(report, "study_id"):
                sample["study_id"] = report.study_id
            elif hasattr(report, "report_id"):
                sample["study_id"] = report.report_id

            samples.append(sample)

        return samples

    @classmethod
    def run_entity_pipeline(
        cls,
        train_data: Union[str, Path, "RexKGDataset"],
        dev_data: Union[str, Path, "RexKGDataset"],
        test_data: Optional[Union[str, Path, "RexKGDataset"]] = None,
        dataset: Optional["RexKGDataset"] = None,
        task: str = "mimic01",
        model: str = "bert-base-uncased",
        output_dir: str = "./pyhealth_rexkg_entity_output",
        max_span_length: int = 8,
        do_train: bool = True,
        do_eval: bool = True,
        eval_test: bool = True,
        learning_rate: float = 1e-5,
        task_learning_rate: float = 5e-4,
        warmup_proportion: float = 0.1,
        train_batch_size: int = 8,
        eval_batch_size: int = 64,
        num_epoch: int = 1,
        print_loss_step: int = 100,
        eval_per_epoch: int = 1,
        bertadam: bool = False,
        do_aug: bool = False,
        train_shuffle: bool = False,
        use_albert: bool = False,
        bert_model_dir: Optional[str] = None,
        context_window: int = 100,
        seed: int = 42,
        test_pred_filename: str = "ent_pred_mimic_headct.json",
        dev_pred_filename: str = "ent_pred_dev.json",
    ) -> Dict[str, Any]:
        """Run the legacy ReXKG entity extraction pipeline inside PyHealth.

        Split inputs can be provided either as JSON/JSONL paths or RexKGDataset
        instances created from those split files.
        """
        legacy = cls._get_legacy_entity_runtime()

        if dataset is not None:
            from ..datasets.rexkg import RexKGDataset

            if not isinstance(dataset, RexKGDataset):
                raise TypeError("dataset must be a RexKGDataset instance when provided.")

        train_data_path = cls._resolve_entity_split_path(train_data, "train_data")
        dev_data_path = cls._resolve_entity_split_path(dev_data, "dev_data")
        test_data_path = (
            cls._resolve_entity_split_path(test_data, "test_data") if test_data is not None else dev_data_path
        )

        train_dataset = cls._create_entity_split_dataset(
            split_root=train_data_path,
            train_data_path=train_data_path,
            dev_data_path=dev_data_path,
            test_data_path=test_data_path,
            dataset_name="rexkg_train",
        )
        dev_dataset = cls._create_entity_split_dataset(
            split_root=dev_data_path,
            train_data_path=train_data_path,
            dev_data_path=dev_data_path,
            test_data_path=test_data_path,
            dataset_name="rexkg_dev",
        )
        test_dataset = cls._create_entity_split_dataset(
            split_root=test_data_path,
            train_data_path=train_data_path,
            dev_data_path=dev_data_path,
            test_data_path=test_data_path,
            dataset_name="rexkg_test",
        )

        args = argparse.Namespace(
            task=task,
            data_dir=str(cls._find_rexkg_ner_root() / "data"),
            output_dir=str(Path(output_dir).expanduser().resolve()),
            max_span_length=max_span_length,
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            learning_rate=learning_rate,
            task_learning_rate=task_learning_rate,
            warmup_proportion=warmup_proportion,
            num_epoch=num_epoch,
            print_loss_step=print_loss_step,
            eval_per_epoch=eval_per_epoch,
            bertadam=bertadam,
            do_aug=do_aug,
            do_train=do_train,
            train_shuffle=train_shuffle,
            do_eval=do_eval,
            eval_test=eval_test,
            dev_pred_filename=dev_pred_filename,
            test_pred_filename=test_pred_filename,
            train_data=str(train_dataset.split_data_path or train_data_path),
            dev_data=str(dev_dataset.split_data_path or dev_data_path),
            test_data=str(test_dataset.split_data_path or test_data_path),
            use_albert=use_albert,
            model=model,
            bert_model_dir=bert_model_dir,
            seed=seed,
            context_window=context_window,
        )

        if "albert" in args.model:
            logger.info("Use Albert: %s", args.model)
            args.use_albert = True

        cls._setseed(args.seed)

        output_dir_path = Path(args.output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)

        log_path = output_dir_path / ("train.log" if args.do_train else "eval.log")
        file_handler = logging.FileHandler(log_path, "w")
        root_logger = logging.getLogger("root")
        root_logger.addHandler(file_handler)

        try:
            root_logger.info(vars(args))

            ner_label2id, ner_id2label = legacy["get_labelmap"](legacy["task_ner_labels"][args.task])
            # Accept common singular/plural label variants produced by upstream data prep.
            label_aliases = {
                "device_present": "devices_present",
                "device_notpresent": "devices_notpresent",
                "procedure": "procedures",
            }
            for alias, canonical in label_aliases.items():
                if canonical in ner_label2id:
                    ner_label2id[alias] = ner_label2id[canonical]
            num_ner_labels = len(legacy["task_ner_labels"][args.task]) + 1

            model_obj = legacy["EntityModel"](args, num_ner_labels=num_ner_labels)

            dev_data_obj = legacy["Dataset"](args.dev_data)
            dev_samples, dev_ner = legacy["convert_dataset_to_samples"](
                dev_data_obj,
                args.max_span_length,
                ner_label2id=ner_label2id,
                context_window=args.context_window,
            )
            dev_batches = legacy["batchify"](dev_samples, args.eval_batch_size)

            best_result = -1.0
            if args.do_train:
                train_data_obj = legacy["Dataset"](args.train_data, is_augment=args.do_aug)
                train_samples, train_ner = legacy["convert_dataset_to_samples"](
                    train_data_obj,
                    args.max_span_length,
                    ner_label2id=ner_label2id,
                    context_window=args.context_window,
                )
                train_batches = legacy["batchify"](train_samples, args.train_batch_size)

                param_optimizer = list(model_obj.bert_model.named_parameters())
                optimizer_grouped_parameters = [
                    {"params": [param for name, param in param_optimizer if "bert" in name]},
                    {
                        "params": [param for name, param in param_optimizer if "bert" not in name],
                        "lr": args.task_learning_rate,
                    },
                ]
                optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
                total_steps = len(train_batches) * args.num_epoch
                scheduler = get_linear_schedule_with_warmup(
                    optimizer,
                    int(total_steps * args.warmup_proportion),
                    total_steps,
                )

                train_loss = 0.0
                train_examples = 0
                global_step = 0
                eval_step = max(1, len(train_batches) // args.eval_per_epoch)
                root_logger.info(
                    "eval_step=%d, train_batches=%d, eval_per_epoch=%d",
                    eval_step,
                    len(train_batches),
                    args.eval_per_epoch,
                )

                for epoch_index in tqdm(range(args.num_epoch)):
                    if args.train_shuffle:
                        random.shuffle(train_batches)
                    for batch_index in tqdm(range(len(train_batches))):
                        output_dict = model_obj.run_batch(train_batches[batch_index], training=True)
                        loss = output_dict["ner_loss"]
                        loss.backward()

                        train_loss += loss.item()
                        train_examples += len(train_batches[batch_index])
                        global_step += 1

                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()

                        if global_step % args.print_loss_step == 0:
                            root_logger.info(
                                "Epoch=%d, iter=%d, loss=%.5f",
                                epoch_index,
                                batch_index,
                                train_loss / max(train_examples, 1),
                            )
                            train_loss = 0.0
                            train_examples = 0

                        if global_step % eval_step == 0:
                            f1 = cls._evaluate_entity_model(model_obj, dev_batches, dev_ner)
                            if f1 > best_result:
                                best_result = f1
                                root_logger.info("!!! Best valid (epoch=%d): %.2f", epoch_index, f1 * 100)
                                cls._save_entity_model(model_obj, args)

            test_f1 = None
            prediction_path = None
            if args.do_eval:
                args.bert_model_dir = args.output_dir
                model_obj = legacy["EntityModel"](args, num_ner_labels=num_ner_labels)
                if args.eval_test:
                    test_data_obj = legacy["Dataset"](args.test_data, is_augment=False)
                    prediction_path = output_dir_path / args.test_pred_filename
                else:
                    test_data_obj = legacy["Dataset"](args.dev_data, is_augment=False)
                    prediction_path = output_dir_path / args.dev_pred_filename

                test_samples, test_ner = legacy["convert_dataset_to_samples"](
                    test_data_obj,
                    args.max_span_length,
                    ner_label2id=ner_label2id,
                    context_window=args.context_window,
                )
                test_batches = legacy["batchify"](test_samples, args.eval_batch_size)
                test_f1 = cls._evaluate_entity_model(model_obj, test_batches, test_ner)
                cls._output_ner_predictions(
                    model_obj,
                    test_batches,
                    test_data_obj,
                    prediction_path,
                    ner_id2label,
                    legacy["NpEncoder"],
                )

            result: Dict[str, Any] = {
                "output_dir": str(output_dir_path),
                "log_file": str(log_path),
                "task": args.task,
                "model": args.model,
            }
            if best_result >= 0:
                result["best_dev_f1"] = best_result
            if test_f1 is not None:
                result["test_f1"] = test_f1
            if prediction_path is not None:
                result["test_prediction_file"] = str(prediction_path)
            return result
        finally:
            root_logger.removeHandler(file_handler)
            file_handler.close()

    @classmethod
    def set_task(
        cls,
        train_data: Union[str, Path, "RexKGDataset"],
        dev_data: Union[str, Path, "RexKGDataset"],
        test_data: Optional[Union[str, Path, "RexKGDataset"]] = None,
        dataset: Optional["RexKGDataset"] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Compatibility alias to run the ReXKG entity pipeline from notebook code."""
        return cls.run_entity_pipeline(
            train_data=train_data,
            dev_data=dev_data,
            test_data=test_data,
            dataset=dataset,
            **kwargs,
        )

    @classmethod
    def _resolve_entity_split_path(
        cls,
        split_data: Union[str, Path, "RexKGDataset"],
        arg_name: str,
    ) -> str:
        from ..datasets.rexkg import RexKGDataset

        if isinstance(split_data, RexKGDataset):
            split_path = split_data.split_data_path
            if not split_path:
                raise ValueError(
                    f"{arg_name} must point to a JSON/JSONL split file when passing RexKGDataset."
                )
            return str(Path(split_path).expanduser().resolve())

        return str(Path(split_data).expanduser().resolve())

    @staticmethod
    def _create_entity_split_dataset(
        split_root: str,
        train_data_path: str,
        dev_data_path: str,
        test_data_path: str,
        dataset_name: str,
    ) -> Any:
        from ..datasets.rexkg import RexKGDataset

        return RexKGDataset(
            root=split_root,
            train_data=train_data_path,
            dev_data=dev_data_path,
            test_data=test_data_path,
            dataset_name=dataset_name,
        )

    @staticmethod
    def _find_rexkg_ner_root() -> Path:
        for parent in Path(__file__).resolve().parents:
            candidate = parent / "src" / "ner"
            if (candidate / "run_entity.py").exists():
                return candidate
        raise FileNotFoundError("Could not locate src/ner for ReXKG legacy entity pipeline")

    @classmethod
    def _get_legacy_entity_runtime(cls) -> Dict[str, Any]:
        ner_root = cls._find_rexkg_ner_root()
        ner_root_str = str(ner_root)
        if ner_root_str not in sys.path:
            sys.path.insert(0, ner_root_str)

        shared_data_structures = importlib.import_module("shared.data_structures")
        shared_const = importlib.import_module("shared.const")
        entity_utils = importlib.import_module("entity.utils")
        entity_models = importlib.import_module("entity.models")

        return {
            "Dataset": shared_data_structures.Dataset,
            "task_ner_labels": shared_const.task_ner_labels,
            "get_labelmap": shared_const.get_labelmap,
            "convert_dataset_to_samples": entity_utils.convert_dataset_to_samples,
            "batchify": entity_utils.batchify,
            "NpEncoder": entity_utils.NpEncoder,
            "EntityModel": entity_models.EntityModel,
        }

    @staticmethod
    def _setseed(seed: int) -> None:
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    @staticmethod
    def _save_entity_model(model_obj: Any, args: argparse.Namespace) -> None:
        logger.info("Saving model to %s...", args.output_dir)
        model_to_save = model_obj.bert_model.module if hasattr(model_obj.bert_model, "module") else model_obj.bert_model
        model_to_save.save_pretrained(args.output_dir)
        model_obj.tokenizer.save_pretrained(args.output_dir)

    @staticmethod
    def _output_ner_predictions(
        model_obj: Any,
        batches: List[List[Dict[str, Any]]],
        dataset: Any,
        output_file: Path,
        ner_id2label: Dict[int, str],
        np_encoder: Type[json.JSONEncoder],
    ) -> None:
        ner_result = {}
        total_pred_entities = 0
        for batch in batches:
            output_dict = model_obj.run_batch(batch, training=False)
            pred_ner = output_dict["pred_ner"]
            for sample, preds in zip(batch, pred_ner):
                sample["doc_key"] = str(sample["doc_key"])
                offset = sample["sent_start_in_doc"] - sample["sent_start"]
                key = sample["doc_key"] + "-" + str(sample["sentence_ix"])
                ner_result[key] = []
                for span, pred in zip(sample["spans"], preds):
                    if pred == 0:
                        continue
                    ner_result[key].append([span[0] + offset, span[1] + offset, ner_id2label[pred]])
                total_pred_entities += len(ner_result[key])

        logger.info("Total pred entities: %d", total_pred_entities)

        js = dataset.js
        for index, doc in enumerate(js):
            doc["doc_key"] = str(doc["doc_key"])
            doc["predicted_ner"] = []
            doc["predicted_relations"] = []
            for sentence_index in range(len(doc["sentences"])):
                key = doc["doc_key"] + "-" + str(sentence_index)
                if key in ner_result:
                    doc["predicted_ner"].append(ner_result[key])
                else:
                    logger.info("%s not in NER results!", key)
                    doc["predicted_ner"].append([])
                doc["predicted_relations"].append([])
            js[index] = doc

        logger.info("Output predictions to %s..", output_file)
        with output_file.open("w", encoding="utf-8") as file_handle:
            file_handle.write("\n".join(json.dumps(doc, cls=np_encoder) for doc in js))

    @staticmethod
    def _evaluate_entity_model(model_obj: Any, batches: List[List[Dict[str, Any]]], total_gold: int) -> float:
        logger.info("Evaluating...")
        current_time = time.time()
        correct = 0
        total_pred = 0
        label_correct = 0
        label_total = 0

        for batch in batches:
            output_dict = model_obj.run_batch(batch, training=False)
            pred_ner = output_dict["pred_ner"]
            for sample, preds in zip(batch, pred_ner):
                for gold, pred in zip(sample["spans_label"], preds):
                    label_total += 1
                    if pred == gold:
                        label_correct += 1
                    if pred != 0 and gold != 0 and pred == gold:
                        correct += 1
                    if pred != 0:
                        total_pred += 1

        acc = label_correct / label_total
        logger.info("Accuracy: %5f", acc)
        logger.info("Cor: %d, Pred TOT: %d, Gold TOT: %d", correct, total_pred, total_gold)
        precision = correct / total_pred if correct > 0 else 0.0
        recall = correct / total_gold if correct > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if correct > 0 else 0.0
        logger.info("P: %.5f, R: %.5f, F1: %.5f", precision, recall, f1)
        logger.info("Used time: %f", time.time() - current_time)
        return f1

    @classmethod
    def _write_rexkg_test_predictions(
        cls,
        model_obj: AutoModelForTokenClassification,
        tokenizer: AutoTokenizer,
        id2label: Dict[int, str],
        test_data_path: str,
        output_path: Path,
        max_length: int,
    ) -> None:
        records = cls._load_json_records(test_data_path)
        model_obj.eval()
        device = model_obj.device

        rendered_records: List[Dict[str, Any]] = []
        for rec in records:
            rec_out = dict(rec)
            sentences = rec.get("sentences")
            if not isinstance(sentences, list):
                text = str(rec.get("text", "")).strip()
                sentences = [text.split()] if text else []

            predicted_ner: List[List[List[Union[int, str]]]] = []
            for sent in sentences:
                tokens = [str(tok) for tok in sent]
                if len(tokens) == 0:
                    predicted_ner.append([])
                    continue

                enc = tokenizer(
                    tokens,
                    is_split_into_words=True,
                    truncation=True,
                    max_length=max_length,
                    return_attention_mask=True,
                    return_tensors="pt",
                )

                with torch.no_grad():
                    logits = model_obj(
                        input_ids=enc["input_ids"].to(device),
                        attention_mask=enc["attention_mask"].to(device),
                    ).logits[0]

                pred_ids = logits.argmax(dim=-1).detach().cpu().tolist()
                word_ids = enc.word_ids(batch_index=0)

                word_level_tags: List[str] = ["O"] * len(tokens)
                seen_words = set()
                for token_idx, word_idx in enumerate(word_ids):
                    if word_idx is None or word_idx in seen_words or word_idx >= len(tokens):
                        continue
                    seen_words.add(word_idx)
                    word_level_tags[word_idx] = id2label.get(int(pred_ids[token_idx]), "O")

                spans = cls._bio_tags_to_spans(word_level_tags)
                predicted_ner.append([[s, e, t] for s, e, t in spans])

            rec_out["predicted_ner"] = predicted_ner
            rec_out["predicted_relations"] = [[] for _ in predicted_ner]
            rendered_records.append(rec_out)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in rendered_records),
            encoding="utf-8",
        )

    @staticmethod
    def _bio_tags_to_spans(tags: List[str]) -> List[Tuple[int, int, str]]:
        spans: List[Tuple[int, int, str]] = []
        start: Optional[int] = None
        cur_type: Optional[str] = None

        for idx, tag in enumerate(tags):
            if tag.startswith("B-"):
                if start is not None and cur_type is not None:
                    spans.append((start, idx - 1, cur_type))
                start = idx
                cur_type = tag[2:]
            elif tag.startswith("I-"):
                ent_type = tag[2:]
                if start is None or cur_type != ent_type:
                    if start is not None and cur_type is not None:
                        spans.append((start, idx - 1, cur_type))
                    start = idx
                    cur_type = ent_type
            else:
                if start is not None and cur_type is not None:
                    spans.append((start, idx - 1, cur_type))
                start = None
                cur_type = None

        if start is not None and cur_type is not None:
            spans.append((start, len(tags) - 1, cur_type))

        return spans

    @classmethod
    def _build_hf_dataset(
        cls,
        json_path: str,
        tokenizer: AutoTokenizer,
        label2id: Dict[str, int],
        max_length: int,
    ) -> _TokenClsJsonDataset:
        records = cls._load_json_records(json_path)

        all_input_ids, all_attention_mask, all_labels = [], [], []
        for rec in records:
            tokens, word_labels = cls._record_to_tokens_and_labels(rec)
            if len(tokens) == 0:
                continue

            encoding = tokenizer(
                tokens,
                is_split_into_words=True,
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_attention_mask=True,
            )

            word_ids = encoding.word_ids()
            label_ids = []
            prev_word = None
            for wid in word_ids:
                if wid is None:
                    label_ids.append(-100)
                elif wid != prev_word:
                    label_ids.append(label2id.get(word_labels[wid], 0))
                else:
                    # same word-piece; keep same tag for simplicity
                    label_ids.append(label2id.get(word_labels[wid], 0))
                prev_word = wid

            all_input_ids.append(encoding["input_ids"])
            all_attention_mask.append(encoding["attention_mask"])
            all_labels.append(label_ids)

        enc = {"input_ids": all_input_ids, "attention_mask": all_attention_mask}
        return _TokenClsJsonDataset(enc, all_labels)

    @staticmethod
    def _load_json_records(path: str) -> List[Dict[str, Any]]:
        p = Path(path).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"Entity pipeline data file not found: {p}")
        raw = p.read_text(encoding="utf-8")
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            # Support JSONL (one JSON object per line), which is common in ReXKG data splits.
            records = []
            for line in raw.splitlines():
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
            return records
        if isinstance(data, dict) and "data" in data:
            return data["data"]
        if isinstance(data, list):
            return data
        raise ValueError(f"Unsupported JSON format in {p}")

    @classmethod
    def _record_to_tokens_and_labels(cls, rec: Dict[str, Any]) -> Tuple[List[str], List[str]]:
        # format A: explicit tokens + ner_tags
        if "tokens" in rec and "ner_tags" in rec:
            tokens = rec["tokens"]
            tags = rec["ner_tags"]
            if len(tokens) != len(tags):
                raise ValueError("tokens and ner_tags length mismatch")
            norm_tags = [t if t in cls.ENTITY_TYPES else "O" for t in tags]
            return tokens, norm_tags

        # format C: ReXKG JSON/JSONL with `sentences` and token-index NER spans.
        # Example span: [start_idx, end_idx, "concept"] where indices are inclusive.
        if "sentences" in rec and "ner" in rec:
            sentences = rec.get("sentences") or []
            ner_by_sentence = rec.get("ner") or []
            if not sentences:
                return [], []

            tokens: List[str] = []
            labels: List[str] = []
            for si, sent_tokens in enumerate(sentences):
                sent = list(sent_tokens)
                sent_labels = ["O"] * len(sent)
                spans = ner_by_sentence[si] if si < len(ner_by_sentence) else []

                for span in spans:
                    if len(span) < 3:
                        continue
                    start, end, ent_type = int(span[0]), int(span[1]), str(span[2]).strip()
                    if start < 0 or end < start:
                        continue
                    b_tag = f"B-{ent_type}"
                    i_tag = f"I-{ent_type}"
                    if b_tag not in cls.ENTITY_TYPES:
                        continue
                    for ti in range(start, min(end, len(sent) - 1) + 1):
                        sent_labels[ti] = b_tag if ti == start else i_tag

                tokens.extend(sent)
                labels.extend(sent_labels)

            return tokens, labels

        # format B: text + entities (char spans)
        text = rec.get("text", "")
        if not text.strip():
            return [], []
        tokens = text.split()
        labels = ["O"] * len(tokens)

        ents = rec.get("entities", [])
        if ents:
            # naive whitespace token span mapping
            offsets = []
            idx = 0
            for tok in tokens:
                start = text.find(tok, idx)
                end = start + len(tok) if start >= 0 else idx + len(tok)
                offsets.append((start, end))
                idx = end

            for e in ents:
                s, en = int(e.get("start", -1)), int(e.get("end", -1))
                et = str(e.get("label", "")).strip()
                if s < 0 or en < 0 or et == "":
                    continue
                b, i = f"B-{et}", f"I-{et}"
                if b not in cls.ENTITY_TYPES:
                    continue
                first = True
                for ti, (ts, te) in enumerate(offsets):
                    if te <= s or ts >= en:
                        continue
                    labels[ti] = b if first else i
                    first = False

        return tokens, labels


class RexKGRelationExtractionRadiology(BaseTask):
    """Relation extraction task for radiology reports using ReXKG.

    This task extracts relationships between entities found in radiology reports.
    It identifies three main relation types:
    - modify: Modifying relationships between entities
    - located_at: Spatial location relationships
    - suggestive_of: Diagnostic suggestions and implications

    Relation extraction follows entity extraction and uses entity pairs to
    determine the type of relationship. This enables construction of structured
    knowledge graphs from unstructured radiology text.

    Relation Types:
        - modify: One entity modifies or describes another (e.g., "right sided"
          modifies "pneumonia")
        - located_at: Spatial relationship between entities (e.g., entity is
          "located_at" anatomical location)
        - suggestive_of: Diagnostic implication (e.g., finding is "suggestive_of"
          a condition)
        - no_relation: Default label when no relation exists between entity pair

    Input:
        Radiology reports with pre-extracted entities:
        - text: Radiology report text
        - entities: List of extracted named entities
        - entity_pairs: Potential entity pairs for relation classification

    Output:
        - relations: List of extracted relations with types
        - relation_labels: Classification of each entity pair's relation type

    Args:
        task_name: Name identifying this task
        input_schema: Schema defining input feature types
        output_schema: Schema defining output label types

    Examples:
        >>> from pyhealth.datasets import SampleDataset
        >>> from pyhealth.tasks import RexKGRelationExtractionRadiology
        >>> dataset = SampleDataset(...)
        >>> task = RexKGRelationExtractionRadiology()
        >>> samples = dataset.set_task(task)
        >>> # Each sample contains extracted relations between entities
    """

    task_name: str = "rexkg_relation_extraction_radiology"
    input_schema: Dict[str, Union[str, Type]] = {
        "text": TextProcessor,
        "entities": SequenceProcessor,
    }
    output_schema: Dict[str, Union[str, Type]] = {"relations": SequenceProcessor}
    _DEFAULT_RELATION_LABELS = {
        "mimic01": ["no_relation", "modify", "located_at", "suggestive_of"],
    }

    def pre_filter(self, df: pl.LazyFrame) -> pl.LazyFrame:
        filtered_df = df.filter(
            pl.col("patient_id").is_in(
                df.filter(pl.col("event_type") == "radiology_reports")
                .select("patient_id")
                .unique()
                .collect()
                .to_series()
            )
        )
        return filtered_df

    def __call__(self, patient: Patient) -> List[Dict]:
        samples = []
        reports = patient.get_events(event_type="radiology_reports")
        if not reports:
            return samples

        for report in reports:
            text = getattr(report, "text", "")
            if not text or text.strip() == "":
                continue

            sample = {
                "patient_id": patient.patient_id,
                "text": text,
                "entities": [],
                "relations": [],
                "entity_pairs": [],
            }
            if hasattr(report, "study_id"):
                sample["study_id"] = report.study_id
            elif hasattr(report, "report_id"):
                sample["study_id"] = report.report_id
            if hasattr(report, "report_type"):
                sample["report_type"] = report.report_type
            samples.append(sample)

        return samples

    # ------------------------------------------------------------------ #
    #  Helpers inlined from run_relation.py (no external src/ner import  #
    #  needed for these pure-Python utilities).                          #
    # ------------------------------------------------------------------ #

    class _InputFeatures:
        def __init__(self, input_ids, input_mask, segment_ids, label_id, sub_idx, obj_idx):
            self.input_ids = input_ids
            self.input_mask = input_mask
            self.segment_ids = segment_ids
            self.label_id = label_id
            self.sub_idx = sub_idx
            self.obj_idx = obj_idx

    @staticmethod
    def _add_marker_tokens(tokenizer: AutoTokenizer, ner_labels: List[str]) -> None:
        new_tokens = ["<SUBJ_START>", "<SUBJ_END>", "<OBJ_START>", "<OBJ_END>"]
        for label in ner_labels:
            new_tokens += [
                "<SUBJ_START=%s>" % label, "<SUBJ_END=%s>" % label,
                "<OBJ_START=%s>" % label, "<OBJ_END=%s>" % label,
            ]
        for label in ner_labels:
            new_tokens += ["<SUBJ=%s>" % label, "<OBJ=%s>" % label]
        tokenizer.add_tokens(new_tokens)
        print("# vocab after adding markers: %d" % len(tokenizer))

    @classmethod
    def _convert_examples_to_features(
        cls,
        examples: List[Dict[str, Any]],
        label2id: Dict[str, int],
        max_seq_length: int,
        tokenizer: AutoTokenizer,
        special_tokens: Dict[str, str],
        unused_tokens: bool = True,
    ) -> List["RexKGRelationExtractionRadiology._InputFeatures"]:
        CLS = "[CLS]"
        SEP = "[SEP]"

        def get_special_token(w: str) -> str:
            if w not in special_tokens:
                if unused_tokens:
                    special_tokens[w] = "[unused%d]" % (len(special_tokens) + 1)
                else:
                    special_tokens[w] = ("<" + w + ">").lower()
            return special_tokens[w]

        num_tokens = 0
        max_tokens = 0
        num_fit_examples = 0
        num_shown_examples = 0
        features = []

        for ex_index, example in enumerate(examples):
            if ex_index % 10000 == 0:
                print("Writing example %d of %d" % (ex_index, len(examples)))

            tokens = [CLS]
            SUBJECT_START_NER = get_special_token("SUBJ_START=%s" % example["subj_type"])
            SUBJECT_END_NER   = get_special_token("SUBJ_END=%s"   % example["subj_type"])
            OBJECT_START_NER  = get_special_token("OBJ_START=%s"  % example["obj_type"])
            OBJECT_END_NER    = get_special_token("OBJ_END=%s"    % example["obj_type"])
            # consume the plain markers so special_tokens tracks them too
            get_special_token("SUBJ_START"); get_special_token("SUBJ_END")
            get_special_token("OBJ_START");  get_special_token("OBJ_END")
            get_special_token("SUBJ=%s" % example["subj_type"])
            get_special_token("OBJ=%s"  % example["obj_type"])

            sub_idx = obj_idx = 0
            for i, token in enumerate(example["token"]):
                if i == example["subj_start"]:
                    sub_idx = len(tokens)
                    tokens.append(SUBJECT_START_NER)
                if i == example["obj_start"]:
                    obj_idx = len(tokens)
                    tokens.append(OBJECT_START_NER)
                for sub_token in tokenizer.tokenize(token):
                    tokens.append(sub_token)
                if i == example["subj_end"]:
                    tokens.append(SUBJECT_END_NER)
                if i == example["obj_end"]:
                    tokens.append(OBJECT_END_NER)
            tokens.append(SEP)

            num_tokens += len(tokens)
            max_tokens = max(max_tokens, len(tokens))

            if len(tokens) > max_seq_length:
                tokens = tokens[:max_seq_length]
                if sub_idx >= max_seq_length:
                    sub_idx = 0
                if obj_idx >= max_seq_length:
                    obj_idx = 0
            else:
                num_fit_examples += 1

            segment_ids = [0] * len(tokens)
            input_ids   = tokenizer.convert_tokens_to_ids(tokens)
            input_mask  = [1] * len(input_ids)
            padding     = [0] * (max_seq_length - len(input_ids))
            input_ids   += padding
            input_mask  += padding
            segment_ids += padding

            try:
                label_id = label2id[example["relation"]]
            except KeyError:
                print(example["relation"])
                label_id = 0

            if num_shown_examples < 20 and (ex_index < 5 or label_id > 0):
                num_shown_examples += 1
                print("*** Example ***")
                print("guid: %s" % example["id"])
                print("tokens: %s" % " ".join(str(x) for x in tokens))
                print("input_ids: %s" % " ".join(str(x) for x in input_ids))
                print("input_mask: %s" % " ".join(str(x) for x in input_mask))
                print("segment_ids: %s" % " ".join(str(x) for x in segment_ids))
                print("label: %s (id = %d)" % (example["relation"], label_id))
                print("sub_idx, obj_idx: %d, %d" % (sub_idx, obj_idx))

            features.append(cls._InputFeatures(
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                label_id=label_id,
                sub_idx=sub_idx,
                obj_idx=obj_idx,
            ))

        print("Average #tokens: %.2f" % (num_tokens * 1.0 / max(len(examples), 1)))
        print("Max #tokens: %d" % max_tokens)
        print("%d (%.2f %%) examples can fit max_seq_length = %d" % (
            num_fit_examples,
            num_fit_examples * 100.0 / max(len(examples), 1),
            max_seq_length,
        ))
        return features

    @staticmethod
    def _simple_accuracy(preds: np.ndarray, labels: np.ndarray) -> float:
        return float((preds == labels).mean())

    @staticmethod
    def _compute_f1_full(
        preds: np.ndarray,
        labels: np.ndarray,
        e2e_ngold: Optional[int],
    ) -> Dict[str, Any]:
        n_gold = n_pred = n_correct = 0
        for pred, label in zip(preds, labels):
            if pred != 0:
                n_pred += 1
            if label != 0:
                n_gold += 1
            if pred != 0 and label != 0 and pred == label:
                n_correct += 1
        if n_correct == 0:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        prec   = n_correct / n_pred
        recall = n_correct / n_gold
        f1     = 2.0 * prec * recall / (prec + recall) if prec + recall > 0 else 0.0
        if e2e_ngold is not None:
            e2e_recall = n_correct / e2e_ngold
            e2e_f1 = 2.0 * prec * e2e_recall / (prec + e2e_recall) if prec + e2e_recall > 0 else 0.0
        else:
            e2e_recall = e2e_f1 = 0.0
        return {
            "precision": prec, "recall": e2e_recall, "f1": e2e_f1,
            "task_recall": recall, "task_f1": f1,
            "n_correct": n_correct, "n_pred": n_pred,
            "n_gold": e2e_ngold, "task_ngold": n_gold,
        }

    @staticmethod
    def _evaluate_model(
        model: Any,
        device: torch.device,
        eval_dataloader: Any,
        eval_label_ids: torch.Tensor,
        num_labels: int,
        e2e_ngold: Optional[int] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any], np.ndarray]:
        from torch.nn import CrossEntropyLoss
        model.eval()
        eval_loss = 0.0
        nb_eval_steps = 0
        all_logits: Optional[np.ndarray] = None

        for input_ids, input_mask, segment_ids, label_ids, sub_idx, obj_idx in eval_dataloader:
            input_ids    = input_ids.to(device)
            input_mask   = input_mask.to(device)
            segment_ids  = segment_ids.to(device)
            label_ids    = label_ids.to(device)
            sub_idx      = sub_idx.to(device)
            obj_idx      = obj_idx.to(device)
            with torch.no_grad():
                logits = model(input_ids, segment_ids, input_mask, labels=None,
                               sub_idx=sub_idx, obj_idx=obj_idx)
            loss_fct = CrossEntropyLoss()
            eval_loss += loss_fct(logits.view(-1, num_labels), label_ids.view(-1)).mean().item()
            nb_eval_steps += 1
            batch_logits = logits.detach().cpu().numpy()
            all_logits = batch_logits if all_logits is None else np.append(all_logits, batch_logits, axis=0)

        eval_loss /= nb_eval_steps
        preds = np.argmax(all_logits, axis=1)
        result = RexKGRelationExtractionRadiology._compute_f1_full(preds, eval_label_ids.numpy(), e2e_ngold)
        result["accuracy"]  = RexKGRelationExtractionRadiology._simple_accuracy(preds, eval_label_ids.numpy())
        result["eval_loss"] = eval_loss

        print("***** Eval results *****")
        for key in sorted(result.keys()):
            print("  %s = %s" % (key, str(result[key])))

        return preds, result, all_logits

    @staticmethod
    def _print_pred_json(
        eval_data: Any,
        eval_examples: List[Dict[str, Any]],
        preds: np.ndarray,
        id2label: Dict[int, str],
        output_file: str,
    ) -> None:
        from relation.utils import decode_sample_id  # type: ignore[import]
        rels: Dict[str, List[Any]] = {}
        for ex, pred in zip(eval_examples, preds):
            doc_sent, sub, obj = decode_sample_id(ex["id"])
            rels.setdefault(doc_sent, [])
            if int(pred) != 0:
                rels[doc_sent].append([sub[0], sub[1], obj[0], obj[1], id2label[int(pred)]])

        js = eval_data.js
        for doc in js:
            doc["predicted_relations"] = []
            for sid in range(len(doc["sentences"])):
                k = "%s@%d" % (doc["doc_key"], sid)
                doc["predicted_relations"].append(rels.get(k, []))

        print("Output predictions to %s.." % output_file)
        with open(output_file, "w") as f:
            f.write("\n".join(json.dumps(doc) for doc in js))

    # ------------------------------------------------------------------ #

    @classmethod
    def run_relation_pipeline(
        cls,
        train_file: str,
        entity_output_dir: str,
        entity_predictions_dev: str = "ent_pred_mimic_headct.json",
        entity_predictions_test: str = "ent_pred_mimic_headct.json",
        task: str = "mimic01",
        model: str = "bert-base-uncased",
        output_dir: str = "./pyhealth_rexkg_relation_output",
        do_train: bool = True,
        do_eval: bool = True,
        do_lower_case: bool = True,
        eval_test: bool = False,
        eval_with_gold: bool = True,
        train_batch_size: int = 16,
        eval_batch_size: int = 32,
        learning_rate: float = 5e-5,
        num_train_epochs: float = 1.0,
        warmup_proportion: float = 0.1,
        max_seq_length: int = 256,
        context_window: int = 0,
        eval_metric: str = "f1",
        eval_per_epoch: int = 1,
        prediction_file: str = "predictions.json",
        train_mode: str = "random_sorted",
        add_new_tokens: bool = False,
        no_cuda: bool = False,
        seed: int = 42,
        negative_label: str = "no_relation",
        ner_src_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run the ReXKG relation pipeline, matching run_relation.py behaviour.

        Parameters
        ----------
        ner_src_dir:
            Absolute path to the ``src/ner`` directory that contains the
            ``relation/``, ``shared/`` packages.  The method tries to
            auto-detect this by walking up from ``output_dir``; supply it
            explicitly when auto-detection fails.
        """
        from torch.utils.data import TensorDataset

        # ── locate src/ner and inject it into sys.path ─────────────────
        if ner_src_dir is None:
            candidate = Path(output_dir).expanduser().resolve()
            for _ in range(10):
                candidate = candidate.parent
                if (candidate / "src" / "ner").is_dir():
                    ner_src_dir = str(candidate / "src" / "ner")
                    break
        if ner_src_dir is None:
            raise RuntimeError(
                "Could not auto-detect src/ner directory. "
                "Pass ner_src_dir=<path to src/ner> explicitly."
            )
        if ner_src_dir not in sys.path:
            sys.path.insert(0, ner_src_dir)

        from relation.utils import generate_relation_data  # type: ignore[import]
        from relation.models import BertForRelation         # type: ignore[import]
        from shared.const import task_rel_labels, task_ner_labels  # type: ignore[import]

        # ── seeding ────────────────────────────────────────────────────
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        device   = torch.device("cuda" if torch.cuda.is_available() and not no_cuda else "cpu")
        n_gpu    = torch.cuda.device_count()

        output_dir_path = Path(output_dir).expanduser().resolve()
        output_dir_path.mkdir(parents=True, exist_ok=True)

        print("device: {}, n_gpu: {}".format(device, n_gpu))

        # ── load / build data ──────────────────────────────────────────
        if do_train:
            train_dataset_obj, train_examples, train_nrel = generate_relation_data(
                train_file, use_gold=True, context_window=context_window
            )
        if (do_eval and do_train) or (do_eval and not eval_test):
            eval_dev_file = str(
                Path(entity_output_dir).expanduser().resolve() / entity_predictions_dev
            )
            eval_dataset_obj, eval_examples, eval_nrel = generate_relation_data(
                eval_dev_file, use_gold=eval_with_gold, context_window=context_window
            )
        if eval_test:
            eval_test_file = str(
                Path(entity_output_dir).expanduser().resolve() / entity_predictions_test
            )
            test_dataset_obj, test_examples, test_nrel = generate_relation_data(
                eval_test_file, use_gold=eval_with_gold, context_window=context_window
            )

        if not do_train and not do_eval:
            raise ValueError("At least one of do_train or do_eval must be True.")

        # ── label list ─────────────────────────────────────────────────
        label_list_path = output_dir_path / "label_list.json"
        if label_list_path.exists():
            with label_list_path.open() as f:
                label_list = json.load(f)
        else:
            label_list = [negative_label] + task_rel_labels[task]
            with label_list_path.open("w") as f:
                json.dump(label_list, f)
        label2id   = {lbl: i for i, lbl in enumerate(label_list)}
        id2label   = {i: lbl for i, lbl in enumerate(label_list)}
        num_labels = len(label_list)

        # ── tokenizer & special tokens ─────────────────────────────────
        tokenizer = AutoTokenizer.from_pretrained(model, do_lower_case=do_lower_case, use_fast=False)
        if add_new_tokens:
            cls._add_marker_tokens(tokenizer, task_ner_labels[task])

        special_tokens_path = output_dir_path / "special_tokens.json"
        if special_tokens_path.exists():
            with special_tokens_path.open() as f:
                special_tokens: Dict[str, str] = json.load(f)
        else:
            special_tokens = {}

        # ── build eval features (dev set) ─────────────────────────────
        if do_eval and (do_train or not eval_test):
            eval_features = cls._convert_examples_to_features(
                eval_examples, label2id, max_seq_length, tokenizer, special_tokens,
                unused_tokens=not add_new_tokens,
            )
            print("***** Dev *****")
            print("  Num examples = %d" % len(eval_examples))
            print("  Batch size = %d" % eval_batch_size)
            all_input_ids   = torch.tensor([f.input_ids   for f in eval_features], dtype=torch.long)
            all_input_mask  = torch.tensor([f.input_mask  for f in eval_features], dtype=torch.long)
            all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
            all_label_ids   = torch.tensor([f.label_id    for f in eval_features], dtype=torch.long)
            all_sub_idx     = torch.tensor([f.sub_idx     for f in eval_features], dtype=torch.long)
            all_obj_idx     = torch.tensor([f.obj_idx     for f in eval_features], dtype=torch.long)
            eval_data_tensor  = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                                              all_label_ids, all_sub_idx, all_obj_idx)
            eval_dataloader   = DataLoader(eval_data_tensor, batch_size=eval_batch_size)
            eval_label_ids_t  = all_label_ids

        with special_tokens_path.open("w") as f:
            json.dump(special_tokens, f)

        # ── training ───────────────────────────────────────────────────
        best_result: Optional[Dict[str, Any]] = None
        relation_model_for_eval: Optional[Any] = None
        if do_train:
            train_features = cls._convert_examples_to_features(
                train_examples, label2id, max_seq_length, tokenizer, special_tokens,
                unused_tokens=not add_new_tokens,
            )
            if train_mode in ("sorted", "random_sorted"):
                train_features = sorted(train_features, key=lambda f: int(np.sum(f.input_mask)))
            else:
                random.shuffle(train_features)

            all_input_ids   = torch.tensor([f.input_ids   for f in train_features], dtype=torch.long)
            all_input_mask  = torch.tensor([f.input_mask  for f in train_features], dtype=torch.long)
            all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
            all_label_ids   = torch.tensor([f.label_id    for f in train_features], dtype=torch.long)
            all_sub_idx     = torch.tensor([f.sub_idx     for f in train_features], dtype=torch.long)
            all_obj_idx     = torch.tensor([f.obj_idx     for f in train_features], dtype=torch.long)
            train_data_tensor = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                                              all_label_ids, all_sub_idx, all_obj_idx)
            train_dataloader  = DataLoader(train_data_tensor, batch_size=train_batch_size)
            train_batches     = list(train_dataloader)

            num_train_optimization_steps = len(train_dataloader) * int(num_train_epochs)

            print("***** Training *****")
            print("  Num examples = %d" % len(train_examples))
            print("  Batch size = %d" % train_batch_size)
            print("  Num steps = %d" % num_train_optimization_steps)

            eval_step = max(1, len(train_batches) // eval_per_epoch)
            print("eval_step=%d, train_batches=%d, eval_per_epoch=%d" % (
                eval_step, len(train_batches), eval_per_epoch))
            print("Eval steps = %d" % num_train_optimization_steps)

            lr            = learning_rate
            relation_model = BertForRelation.from_pretrained(model, cache_dir=None, num_rel_labels=num_labels)
            if hasattr(relation_model, "bert"):
                relation_model.bert.resize_token_embeddings(len(tokenizer))
            elif hasattr(relation_model, "albert"):
                relation_model.albert.resize_token_embeddings(len(tokenizer))
            else:
                raise TypeError("Unknown model class")
            relation_model.to(device)
            relation_model_for_eval = relation_model
            if n_gpu > 1:
                relation_model = torch.nn.DataParallel(relation_model)

            param_optimizer = list(relation_model.named_parameters())
            no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                 "weight_decay": 0.01},
                {"params": [p for n, p in param_optimizer if     any(nd in n for nd in no_decay)],
                 "weight_decay": 0.0},
            ]
            optimizer = AdamW(optimizer_grouped_parameters, lr=lr)
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                int(num_train_optimization_steps * warmup_proportion),
                num_train_optimization_steps,
            )

            import time as _time
            start_time  = _time.time()
            global_step = 0
            tr_loss     = 0.0
            nb_tr_steps = 0

            for epoch in range(int(num_train_epochs)):
                relation_model.train()
                print("Start epoch #{} (lr = {})...".format(epoch, lr))
                if train_mode in ("random", "random_sorted"):
                    random.shuffle(train_batches)

                for step, batch in enumerate(train_batches):
                    batch = tuple(t.to(device) for t in batch)
                    inp_ids, inp_mask, seg_ids, lbl_ids, s_idx, o_idx = batch
                    loss = relation_model(inp_ids, seg_ids, inp_mask, lbl_ids, s_idx, o_idx)
                    if n_gpu > 1:
                        loss = loss.mean()
                    loss.backward()
                    tr_loss     += loss.item()
                    nb_tr_steps += 1
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

                    if (step + 1) % eval_step == 0:
                        print("Epoch: {}, Step: {} / {}, used_time = {:.2f}s, loss = {:.6f}".format(
                            epoch, step + 1, len(train_batches),
                            _time.time() - start_time, tr_loss / nb_tr_steps,
                        ))
                        if do_eval:
                            preds_arr, result, _ = cls._evaluate_model(
                                relation_model, device, eval_dataloader,
                                eval_label_ids_t, num_labels, e2e_ngold=eval_nrel,
                            )
                            relation_model.train()
                            result.update({"global_step": global_step, "epoch": epoch,
                                           "learning_rate": lr, "batch_size": train_batch_size})
                            if best_result is None or result[eval_metric] > best_result[eval_metric]:
                                best_result = result
                                print("!!! Best dev %s (lr=%s, epoch=%d): %.2f" % (
                                    eval_metric, str(lr), epoch, result[eval_metric] * 100.0))
                                cls._save_trained_model(str(output_dir_path), relation_model, tokenizer)
                        else:
                            cls._save_trained_model(str(output_dir_path), relation_model, tokenizer)

                print("Epoch: {}, Step: {} / {}, used_time = {:.2f}s, loss = {:.6f}".format(
                    epoch, step + 1, len(train_batches),
                    _time.time() - start_time, tr_loss / nb_tr_steps,
                ))
                if do_eval:
                    preds_arr, result, _ = cls._evaluate_model(
                        relation_model, device, eval_dataloader,
                        eval_label_ids_t, num_labels, e2e_ngold=eval_nrel,
                    )
                    relation_model.train()
                    result.update({"global_step": global_step, "epoch": epoch,
                                   "learning_rate": lr, "batch_size": train_batch_size})
                    if best_result is None or result[eval_metric] > best_result[eval_metric]:
                        best_result = result
                        print("!!! Best dev %s (lr=%s, epoch=%d): %.2f" % (
                            eval_metric, str(lr), epoch, result[eval_metric] * 100.0))
                        cls._save_trained_model(str(output_dir_path), relation_model, tokenizer)
                else:
                    cls._save_trained_model(str(output_dir_path), relation_model, tokenizer)

        # ── final evaluation ───────────────────────────────────────────
        evaluation_results: Dict[str, Any] = {}
        if do_eval:
            print(special_tokens)
            if eval_test:
                eval_dataset_obj  = test_dataset_obj   # noqa: F821
                eval_examples     = test_examples       # noqa: F821
                eval_features     = cls._convert_examples_to_features(
                    test_examples, label2id, max_seq_length, tokenizer, special_tokens,
                    unused_tokens=not add_new_tokens,
                )
                eval_nrel = test_nrel                   # noqa: F821
                print(special_tokens)
                print("***** Test *****")
                print("  Num examples = %d" % len(test_examples))
                print("  Batch size = %d" % eval_batch_size)
                all_input_ids   = torch.tensor([f.input_ids   for f in eval_features], dtype=torch.long)
                all_input_mask  = torch.tensor([f.input_mask  for f in eval_features], dtype=torch.long)
                all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
                all_label_ids   = torch.tensor([f.label_id    for f in eval_features], dtype=torch.long)
                all_sub_idx     = torch.tensor([f.sub_idx     for f in eval_features], dtype=torch.long)
                all_obj_idx     = torch.tensor([f.obj_idx     for f in eval_features], dtype=torch.long)
                eval_data_tensor = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                                                 all_label_ids, all_sub_idx, all_obj_idx)
                eval_dataloader  = DataLoader(eval_data_tensor, batch_size=eval_batch_size)
                eval_label_ids_t = all_label_ids

            if relation_model_for_eval is not None:
                relation_model = relation_model_for_eval
            else:
                try:
                    relation_model = BertForRelation.from_pretrained(str(output_dir_path), num_rel_labels=num_labels)
                except RuntimeError as exc:
                    raise RuntimeError(
                        "Failed to load relation checkpoint from output_dir. "
                        "This usually means the directory contains an older checkpoint from a different model architecture. "
                        "Use a fresh output_dir (or delete old files in it) and run again."
                    ) from exc
            relation_model.to(device)
            preds_arr, evaluation_results, _ = cls._evaluate_model(
                relation_model, device, eval_dataloader,
                eval_label_ids_t, num_labels, e2e_ngold=eval_nrel,
            )

            print("*** Evaluation Results ***")
            for key in sorted(evaluation_results.keys()):
                print("  %s = %s" % (key, str(evaluation_results[key])))

            cls._print_pred_json(
                eval_dataset_obj, eval_examples, preds_arr, id2label,
                str(output_dir_path / prediction_file),
            )

        return {
            "output_dir": str(output_dir_path),
            "prediction_file": str(output_dir_path / prediction_file),
            "task": task,
            "model": model,
            **evaluation_results,
        }

    @staticmethod
    def _save_trained_model(output_dir: str, model: Any, tokenizer: AutoTokenizer) -> None:
        import os
        WEIGHTS_NAME = "pytorch_model.bin"
        CONFIG_NAME  = "config.json"
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        print("Saving model to %s" % output_dir)
        model_to_save = model.module if hasattr(model, "module") else model
        torch.save(model_to_save.state_dict(), os.path.join(output_dir, WEIGHTS_NAME))
        model_to_save.config.to_json_file(os.path.join(output_dir, CONFIG_NAME))
        tokenizer.save_vocabulary(output_dir)

    @classmethod
    def set_task(
        cls,
        train_file: str,
        entity_output_dir: str,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Compatibility alias to run the ReXKG relation pipeline from notebook code."""
        return cls.run_relation_pipeline(
            train_file=train_file,
            entity_output_dir=entity_output_dir,
            **kwargs,
        )

    @staticmethod
    def _load_json_records(path: str) -> List[Dict[str, Any]]:
        p = Path(path).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"Relation pipeline data file not found: {p}")
        raw = p.read_text(encoding="utf-8")
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            rows = []
            for line in raw.splitlines():
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
            return rows
        if isinstance(data, list):
            return data
        if isinstance(data, dict) and "data" in data:
            return data["data"]
        raise ValueError(f"Unsupported JSON format in {p}")

    @classmethod
    def _infer_relation_label_list(cls, task: str, examples: List[Dict[str, Any]]) -> List[str]:
        if task in cls._DEFAULT_RELATION_LABELS:
            return list(cls._DEFAULT_RELATION_LABELS[task])
        labels = sorted({e["label"] for e in examples if e["label"] != "no_relation"})
        return ["no_relation"] + labels

    @staticmethod
    def _non_overlapping_spans(a: List[int], b: List[int]) -> bool:
        return a[1] < b[0] or b[1] < a[0]

    @classmethod
    def _build_relation_examples(
        cls,
        docs: List[Dict[str, Any]],
        use_gold_entities: bool,
        include_gold_relations: bool,
    ) -> List[Dict[str, Any]]:
        examples: List[Dict[str, Any]] = []
        for doc_idx, doc in enumerate(docs):
            sentences = doc.get("sentences", [])
            if not isinstance(sentences, list):
                continue

            entity_key = "ner" if use_gold_entities and "ner" in doc else "predicted_ner"
            entities_by_sentence = doc.get(entity_key, [])
            relations_by_sentence = doc.get("relations", []) if include_gold_relations else []

            for sent_idx, sent_tokens in enumerate(sentences):
                if not isinstance(sent_tokens, list):
                    continue
                entities = entities_by_sentence[sent_idx] if sent_idx < len(entities_by_sentence) else []
                relations = relations_by_sentence[sent_idx] if sent_idx < len(relations_by_sentence) else []

                gold_map: Dict[Tuple[int, int, int, int], str] = {}
                for rel in relations:
                    if isinstance(rel, list) and len(rel) >= 5:
                        gold_map[(int(rel[0]), int(rel[1]), int(rel[2]), int(rel[3]))] = str(rel[4])

                clean_entities: List[List[Union[int, str]]] = []
                for ent in entities:
                    if isinstance(ent, list) and len(ent) >= 3:
                        s, e, t = int(ent[0]), int(ent[1]), str(ent[2])
                        if 0 <= s <= e < len(sent_tokens):
                            clean_entities.append([s, e, t])

                for i, subj in enumerate(clean_entities):
                    for j, obj in enumerate(clean_entities):
                        if i == j:
                            continue
                        if not cls._non_overlapping_spans(subj, obj):
                            continue
                        key = (int(subj[0]), int(subj[1]), int(obj[0]), int(obj[1]))
                        label = gold_map.get(key, "no_relation")
                        examples.append(
                            {
                                "text": cls._render_pair_text(sent_tokens, subj, obj),
                                "label": label,
                                "doc_idx": doc_idx,
                                "sent_idx": sent_idx,
                                "subj": [int(subj[0]), int(subj[1])],
                                "obj": [int(obj[0]), int(obj[1])],
                            }
                        )
        return examples

    @staticmethod
    def _render_pair_text(tokens: List[str], subj: List[Union[int, str]], obj: List[Union[int, str]]) -> str:
        s0, s1, st = int(subj[0]), int(subj[1]), str(subj[2])
        o0, o1, ot = int(obj[0]), int(obj[1]), str(obj[2])
        rendered: List[str] = []
        for idx, tok in enumerate(tokens):
            if idx == s0:
                rendered.append(f"<SUBJ_START={st}>")
            if idx == o0:
                rendered.append(f"<OBJ_START={ot}>")
            rendered.append(str(tok))
            if idx == s1:
                rendered.append(f"<SUBJ_END={st}>")
            if idx == o1:
                rendered.append(f"<OBJ_END={ot}>")
        return " ".join(rendered)

    @staticmethod
    def _predict_relation_ids(
        model_obj: AutoModelForSequenceClassification,
        tokenizer: AutoTokenizer,
        device: torch.device,
        examples: List[Dict[str, Any]],
        batch_size: int,
        max_length: int,
    ) -> List[int]:
        if len(examples) == 0:
            return []
        texts = [e["text"] for e in examples]
        enc = tokenizer(texts, truncation=True, padding=True, max_length=max_length)
        ds = _RelationClsDataset(enc, [0] * len(texts))
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False)
        preds: List[int] = []

        model_obj.eval()
        with torch.no_grad():
            for batch in dl:
                _ = batch.pop("labels")
                batch = {k: v.to(device) for k, v in batch.items()}
                logits = model_obj(**batch).logits
                preds.extend(logits.argmax(dim=-1).detach().cpu().tolist())
        return preds

    @staticmethod
    def _compute_relation_metrics(pred_ids: List[int], gold_ids: List[int]) -> Dict[str, float]:
        if len(pred_ids) == 0 or len(gold_ids) == 0:
            return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

        correct = 0
        n_pred = 0
        n_gold = 0
        for p, g in zip(pred_ids, gold_ids):
            if p != 0:
                n_pred += 1
            if g != 0:
                n_gold += 1
            if p != 0 and g != 0 and p == g:
                correct += 1

        precision = correct / n_pred if n_pred else 0.0
        recall = correct / n_gold if n_gold else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        accuracy = float(np.mean(np.array(pred_ids) == np.array(gold_ids)))
        return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

    @staticmethod
    def _write_relation_predictions(
        docs: List[Dict[str, Any]],
        examples: List[Dict[str, Any]],
        pred_labels: List[str],
        output_path: Path,
    ) -> None:
        for doc in docs:
            sentences = doc.get("sentences", [])
            doc["predicted_relations"] = [[] for _ in range(len(sentences))]

        for ex, pred_label in zip(examples, pred_labels):
            if pred_label == "no_relation":
                continue
            docs[ex["doc_idx"]]["predicted_relations"][ex["sent_idx"]].append(
                [ex["subj"][0], ex["subj"][1], ex["obj"][0], ex["obj"][1], pred_label]
            )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            f.write("\n".join(json.dumps(doc) for doc in docs))


class RexKGReverseStructureRadiology(BaseTask):
    """Reverse-structure ReXKG relation predictions into flattened JSON docs.

    This task converts relation extraction outputs into a compact document-level
    format used by downstream graph construction and inspection workflows.
    """

    task_name: str = "rexkg_reverse_structure_radiology"
    input_schema: Dict[str, Union[str, Type]] = {"input_json_file": TextProcessor}
    output_schema: Dict[str, Union[str, Type]] = {"processed_docs": SequenceProcessor}

    def __call__(self, patient: Patient) -> List[Dict]:
        """This task is pipeline-oriented and does not operate on Patient events."""
        raise NotImplementedError(
            "RexKGReverseStructureRadiology is a pipeline-style task. "
            "Use RexKGReverseStructureRadiology.set_task(...) instead."
        )

    @classmethod
    def run_reverse_structure_pipeline(
        cls,
        input_json_file: str,
        save_json_file: str,
    ) -> List[Dict[str, Any]]:
        """Convert ReXKG relation prediction records into flattened JSON output."""
        data = cls._load_prediction_records(input_json_file)

        processed_data: List[Dict[str, Any]] = []
        for doc in data:
            sentence_tokens = (doc.get("sentences") or [[]])[0]
            sentence_text = " ".join(sentence_tokens)

            ner_candidates = cls._resolve_candidates(doc, "predicted_ner", "ner")
            rel_candidates = cls._resolve_candidates(doc, "predicted_relations", "relations")

            predicted_entities: Dict[str, str] = {}
            for entity_info in ner_candidates:
                if not entity_info or len(entity_info) < 3:
                    continue
                start, end, entity_type = int(entity_info[0]), int(entity_info[1]), str(entity_info[2])
                entity_text = " ".join(sentence_tokens[start : end + 1])
                predicted_entities[entity_text] = entity_type

            predicted_relations: List[Dict[str, str]] = []
            for relation_info in rel_candidates:
                if not relation_info or len(relation_info) < 5:
                    continue
                start1, end1, start2, end2, relation_type = relation_info
                entity1_text = " ".join(sentence_tokens[int(start1) : int(end1) + 1])
                entity2_text = " ".join(sentence_tokens[int(start2) : int(end2) + 1])
                predicted_relations.append(
                    {
                        "source_entity": entity1_text,
                        "target_entity": entity2_text,
                        "type": str(relation_type),
                    }
                )

            processed_doc = {
                "doc_key": doc.get("doc_key"),
                "sentences": sentence_text,
                "entities": predicted_entities,
                "relations": predicted_relations,
            }
            processed_data.append(processed_doc)

        save_path = Path(save_json_file).expanduser().resolve()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with save_path.open("w", encoding="utf-8") as output_file:
            json.dump(processed_data, output_file, ensure_ascii=False, indent=4)

        return processed_data

    @classmethod
    def set_task(
        cls,
        input_json_file: str,
        save_json_file: str,
    ) -> List[Dict[str, Any]]:
        """Compatibility entry-point for reverse-structure notebook workflows."""
        return cls.run_reverse_structure_pipeline(
            input_json_file=input_json_file,
            save_json_file=save_json_file,
        )

    @staticmethod
    def _load_prediction_records(input_json_file: str) -> List[Dict[str, Any]]:
        p = Path(input_json_file).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"Input prediction file not found: {p}")

        raw = p.read_text(encoding="utf-8")
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            records: List[Dict[str, Any]] = []
            for line in raw.splitlines():
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
            return records

        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return [data]
        raise ValueError(f"Unsupported prediction format in {p}")

    @staticmethod
    def _resolve_candidates(doc: Dict[str, Any], key_pred: str, key_gold: str) -> List[Any]:
        pred = doc.get(key_pred, [[]])
        if pred and isinstance(pred, list) and len(pred) > 0 and pred[0]:
            return pred[0]

        gold = doc.get(key_gold, [[]])
        if gold and isinstance(gold, list) and len(gold) > 0:
            return gold[0]
        return []


class RexKGGetEntitiesRadiology(BaseTask):
    """Generate entity and relation CSV summaries from reverse-structured JSON.

    This class mirrors the behavior of ``src/kg_construct/code/get_entities.py``
    but exposes it via the PyHealth task-style API.
    """

    task_name: str = "rexkg_get_entities_radiology"
    input_schema: Dict[str, Union[str, Type]] = {"ent_pred_mimic_headct": TextProcessor}
    output_schema: Dict[str, Union[str, Type]] = {"all_entities_csv": TextProcessor}

    def __call__(self, patient: Patient) -> List[Dict]:
        raise NotImplementedError(
            "RexKGGetEntitiesRadiology is a pipeline-style task. "
            "Use RexKGGetEntitiesRadiology.set_task(...) instead."
        )

    @classmethod
    def set_task(
        cls,
        ent_pred_mimic_headct: str,
        ent_real_pred_mimic_headct: str,
        save_entity_dir: str = "./entities",
        save_real_dir: str = "./relation",
    ) -> Dict[str, str]:
        """Compatibility entrypoint that matches get_entities.py arguments."""
        return cls.run_get_entities_pipeline(
            ent_pred_mimic_headct=ent_pred_mimic_headct,
            ent_real_pred_mimic_headct=ent_real_pred_mimic_headct,
            save_entity_dir=save_entity_dir,
            save_real_dir=save_real_dir,
        )

    @classmethod
    def run_get_entities_pipeline(
        cls,
        ent_pred_mimic_headct: str,
        ent_real_pred_mimic_headct: str,
        save_entity_dir: str = "./entities",
        save_real_dir: str = "./relation",
    ) -> Dict[str, str]:
        """Run entity and relation aggregation identical to get_entities.py."""
        entity_dir = Path(save_entity_dir).expanduser().resolve()
        real_dir = Path(save_real_dir).expanduser().resolve()
        entity_dir.mkdir(parents=True, exist_ok=True)
        real_dir.mkdir(parents=True, exist_ok=True)

        all_entities_csv = entity_dir / "all_entities.csv"
        all_relations_csv = real_dir / "all_relations.csv"

        cls._extract_entities(
            json_file=str(Path(ent_pred_mimic_headct).expanduser().resolve()),
            output_csv=str(all_entities_csv),
        )
        cls._filter_max_count(str(all_entities_csv), str(entity_dir))
        cls._extract_relations(
            input_json_file=str(Path(ent_real_pred_mimic_headct).expanduser().resolve()),
            save_csv_file=str(all_relations_csv),
        )

        return {
            "all_entities_csv": str(all_entities_csv),
            "all_relations_csv": str(all_relations_csv),
            "save_entity_dir": str(entity_dir),
            "save_real_dir": str(real_dir),
        }

    @staticmethod
    def _is_number(s: str) -> bool:
        return s.isdigit()

    @staticmethod
    def _has_measurement_units(text: str) -> bool:
        pattern = r"\d+\s*(mm|cm|m|km|in|ft|yd|mi)"
        matches = re.findall(pattern, text)
        return bool(matches)

    @staticmethod
    def _contains_digit(s: str) -> bool:
        return any(char.isdigit() for char in s)

    @classmethod
    def _extract_entities(cls, json_file: str, output_csv: str) -> None:
        with open(json_file, "r", encoding="utf-8") as file:
            data = json.load(file)

        all_entities: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

        for entry in data:
            entities = entry.get("entities", {})
            for entity, category in entities.items():
                entity = entity.lower()
                if "cm" in entity.split() or "mm" in entity.split() or "-cm" in entity or "-mm" in entity:
                    category = "size"
                elif cls._has_measurement_units(entity) or cls._is_number(entity):
                    category = "size"
                else:
                    category = category.split("_")[0]

                if cls._contains_digit(entity) and category != "size":
                    continue

                all_entities[entity][category] += 1

        sorted_entities = sorted(all_entities.items(), key=lambda x: sum(x[1].values()), reverse=True)

        with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["entity", "entity_type", "count"])
            for entity, types_count in sorted_entities:
                sorted_types_count = sorted(types_count.items(), key=lambda kv: kv[1], reverse=True)
                for entity_type, count in sorted_types_count:
                    writer.writerow([entity, entity_type, count])

    @staticmethod
    def _filter_max_count(csv_file: str, save_entity_dir: str) -> None:
        entity_max_count: Dict[str, Dict[str, Union[str, int]]] = {}
        with open(csv_file, "r", newline="", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                entity = row["entity"]
                count = int(row["count"])
                if entity not in entity_max_count or count > int(entity_max_count[entity]["count"]):
                    entity_max_count[entity] = {
                        "entity_type": row["entity_type"],
                        "count": count,
                    }

        output_all_entities = Path(save_entity_dir) / "all_entities.csv"
        with open(output_all_entities, "w", newline="", encoding="utf-8") as csvfile:
            fieldnames = ["entity", "entity_type", "count"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for entity, data in entity_max_count.items():
                writer.writerow(
                    {
                        "entity": entity,
                        "entity_type": str(data["entity_type"]),
                        "count": int(data["count"]),
                    }
                )

        entity_types = {str(row["entity_type"]) for row in entity_max_count.values()}
        for entity_type in entity_types:
            csv_file_path = Path(save_entity_dir) / f"{entity_type}.csv"
            with open(csv_file_path, "w", newline="", encoding="utf-8") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["entity", "count"])
                for entity, data in entity_max_count.items():
                    if data["entity_type"] == entity_type:
                        writer.writerow([entity, int(data["count"])])

    @staticmethod
    def _extract_relations(input_json_file: str, save_csv_file: str) -> None:
        with open(input_json_file, "r", encoding="utf-8") as file:
            input_data = json.load(file)

        relation_rows: List[List[str]] = []
        for data_idx in tqdm(input_data):
            relations = data_idx.get("relations", {})
            for relation_idx in relations:
                relation_rows.append(
                    [
                        relation_idx["source_entity"].lower(),
                        relation_idx["target_entity"].lower(),
                        relation_idx["type"],
                    ]
                )

        count_dict = Counter(tuple(row) for row in relation_rows)

        with open(save_csv_file, "w", newline="", encoding="utf-8") as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(["source_entity", "target_entity", "type", "count"])
            for key, value in sorted(count_dict.items(), key=lambda x: x[1], reverse=True):
                csvwriter.writerow([key[0], key[1], key[2], value])


class RexKGGPT4EntityExtractionRadiology(BaseTask):
    """GPT-4 entity extraction pipeline for ReXKG-style radiology findings.

    This class mirrors the logic in ``src/ner/data/gpt4_entity_extraction.py``
    while exposing a task-style API within ``pyhealth.tasks``.
    """

    task_name: str = "rexkg_gpt4_entity_extraction_radiology"
    input_schema: Dict[str, Union[str, Type]] = {"section_findings": TextProcessor}
    output_schema: Dict[str, Union[str, Type]] = {"res": TextProcessor}

    _FEWSHOT_SAMPLES: List[Dict[str, str]] = [
        {
            "context": "<Input> Unchanged position of the left upper extremity PICC line. Again seen are surgical clips projecting over the right hemithorax.   Increased stranding opacities are noted in the left retrocardiac region.<\\Input>",
            "response": "{'Unchanged position of the left upper extremity PICC line.':{'Unchanged': 'concept','position':'concept','left' : 'concept', 'upper': 'concept','extremity':'anatomy','PICC line':'device_present'}, 'Again seen are surgical clips projecting over the right hemithorax. ':{'surgical clips':'device_present', 'right' : 'concept',  'hemithorax': 'anatomy'},'Increased stranding opacities are noted in the left retrocardiac region. ':{'Increased':'concept','stranding' : 'concept','opacities': 'disorder_present','left':'concept','retrocardiac':'anatomy','region':'anatomy'}}",
        }
    ]

    def __call__(self, patient: Patient) -> List[Dict]:
        raise NotImplementedError(
            "RexKGGPT4EntityExtractionRadiology is a pipeline-style task. "
            "Use RexKGGPT4EntityExtractionRadiology.set_task(...) instead."
        )

    @classmethod
    def set_task(
        cls,
        dataset: Optional["RexKGCheXpertDataset"] = None,
        input_csv_file: Optional[str] = None,
        save_json_file: str = "./gpt4_entities_chexpert_plus.json",
        start_idx: int = 0,
        end_idx: int = 1000,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        api_type = str,
        api_version = str,
        model: str = "gpt-4o-2024-05-13",
    ) -> Dict[str, Any]:
        """Run GPT-4 extraction over section findings.

        Either ``dataset`` (RexKGCheXpertDataset) or ``input_csv_file`` must be
        provided. If both are provided, ``input_csv_file`` takes precedence.
        """
        
        openai.api_type = api_type
        openai.api_version = api_version
        openai.api_key = api_key
        openai.api_base = api_base
        resolved_input = cls._resolve_input_csv(dataset=dataset, input_csv_file=input_csv_file)
        output_path = Path(save_json_file).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        return cls._evaluate_notes(
            input_csv_file=str(resolved_input),
            save_json_file=str(output_path),
            start_idx=start_idx,
            end_idx=end_idx,
            api_key=api_key,
            api_base=api_base,
            model=model,
        )

    @staticmethod
    def _resolve_input_csv(
        dataset: Optional["RexKGCheXpertDataset"],
        input_csv_file: Optional[str],
    ) -> Path:
        if input_csv_file:
            p = Path(input_csv_file).expanduser().resolve()
            if not p.exists():
                raise FileNotFoundError(f"Input CSV file not found: {p}")
            return p

        if dataset is not None:
            prepared = Path(dataset.root).expanduser().resolve() / "rexkg-chexpert-pyhealth.csv"
            if not prepared.exists():
                raise FileNotFoundError(
                    f"Prepared CheXpert CSV not found at {prepared}. "
                    "Please construct RexKGCheXpertDataset first so it prepares this file."
                )
            return prepared

        raise ValueError("Either dataset or input_csv_file must be provided.")

    @classmethod
    def _get_messages(cls, query: str) -> List[Dict[str, str]]:
        messages: List[Dict[str, str]] = [
            {
                "role": "system",
                "content": "You are a radiologist performing clinical term extraction from the FINDINGS and IMPRESSION sections in the radiology report.                     Here a clinical term can be in ['anatomy','disorder_present','disorder_notpresent','procedures','devices','concept', 'devices_present','devices_notpresent','size'].                     'anatomy' refers to the anatomical body;                    'disorder_present' refers to findings or diseases are present according to the sentence;                     'disorder_notpresent' refers to findings or diseases are not present according to the sentence;                     'procedures' refers to procedures are used to diagnose, measure, monitor or treat problems;                     'devices' refers to any instrument, apparatus for medical purpose.                     'size' refers to the measurement of disorders or anatomy, for example, '3mm','4x5 cm'.                     'concept' refers to descriptors such as 'acute' or 'chronic','large', size or severity, or other modifiers, or descriptors of anatomy being normal.                     For example, right pleural effusion , 'right' should be a 'concept', and 'pleural' should be  'anatomy' and 'effusion' should be 'disorder-present' or 'disorder-notpresent'.                    For example, normal cardiomediastinal silhouette. 'normal' and 'silhouette' should be 'concept', 'cardiomediastinal' should be 'anatomy'.                      Please extract terms one word at a time whenever possible, avoiding phrases. Note that terms like 'no' and 'no evidence of' are not considered entities.                     Given a list of radiology sentence input in the format:                     <Input><sentence><sentence><\\Input>                     Please reply with the JSON format following template: {'<sentence>':{'entity':'entity type','entity':'entity type'},'<sentence>':{'entity':'entity type','entity':'entity type'}}",
            }
        ]
        for sample in cls._FEWSHOT_SAMPLES:
            messages.append({"role": "user", "content": sample["context"]})
            messages.append({"role": "assistant", "content": sample["response"]})
        messages.append({"role": "user", "content": query})
        return messages


    @staticmethod
    def _estimate_cost(prompt_tokens: int, completion_tokens: int) -> float:
        input_cost = 0.005
        output_cost = 0.015
        return input_cost * prompt_tokens / 1000 + output_cost * completion_tokens / 1000


    @classmethod
    def _chatgpt_input(
        cls,
        messages: List[Dict[str, str]],
        api_key: Optional[str],
        api_base: Optional[str],
        model: str,
    ) -> Tuple[Union[Dict[str, Any], str], float]:
        # _apply_openai_credentials(openai, api_key, api_base)
        # if counter % 100 == 0:
        print("input json:", messages)
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            response_format={"type": "json_object"},
        )
        try:
            res = response["choices"][0]["message"]["content"]
            cost = cls._estimate_cost(
                response["usage"]["prompt_tokens"],
                response["usage"]["completion_tokens"],
            )
            return json.loads(res), cost
        except Exception:
            res = response["choices"][0]["message"]["content"]
            cost = cls._estimate_cost(
                response["usage"]["prompt_tokens"],
                response["usage"]["completion_tokens"],
            )
            return res, cost


    @classmethod
    def _test_prompt(
        cls,
        findings_idx: str,
        api_key: Optional[str],
        api_base: Optional[str],
        model: str,
    ) -> Tuple[Union[Dict[str, Any], str], float]:
        content = "<Input>" + findings_idx + "<\\Input>"
        messages = cls._get_messages(content)
        return cls._chatgpt_input(messages, api_key=api_key, api_base=api_base, model=model)


    @classmethod
    def _evaluate_notes(
        cls,
        input_csv_file: str,
        save_json_file: str,
        start_idx: int,
        end_idx: int,
        api_key: Optional[str],
        api_base: Optional[str],
        model: str,
    ) -> Dict[str, Any]:    
        df = pd.read_csv(input_csv_file)[0:1000]
        image_id_list = df["path_to_image"].to_list()
        findings_list = df["section_findings"].to_list()
        summary_cost = 0
        try:
            with open(save_json_file, "r") as outfile:
                save_data_dict: Dict[str, Any] = json.load(outfile)
        except Exception:
            save_data_dict = {}
        counter = 0
        for idx in range(start_idx, end_idx):
            if counter % 100 == 0:
                print("Extracting patient entity ", counter, " out of ", end_idx, " records." )
            counter = counter + 1
            image_idx = image_id_list[idx]
            findings_idx = findings_list[idx]
            if image_idx not in save_data_dict:
                save_data_dict_idx: Dict[str, Any] = {"section_findings": findings_idx}
                try:
                    res, cost = cls._test_prompt(
                        findings_idx,
                        api_key=api_key,
                        api_base=api_base,
                        model=model,
                    )
                    print({"section_findings": findings_idx})
                    summary_cost += cost
                    save_data_dict_idx["res"] = res
                    save_data_dict_idx["cost"] = cost
                    save_data_dict[image_idx] = save_data_dict_idx
                except Exception:
                    # print(idx, image_idx)
                    time.sleep(1)
                with open(save_json_file, "w") as outfile:
                    json.dump(save_data_dict, outfile, indent=4)
        # print("SUMMARY COST: ", summary_cost)
        return {
            "save_json_file": str(Path(save_json_file).expanduser().resolve()),
            "num_saved": len(save_data_dict),
            "summary_cost": summary_cost,
            "start_idx": start_idx,
            "end_idx": end_idx,
            "model": model,
        }





class RexKGGPT4RelationExtractionRadiology(BaseTask):
    """GPT-4 relation extraction pipeline mirroring gpt4_relation_extraction.py."""

    task_name: str = "rexkg_gpt4_relation_extraction_radiology"
    input_schema: Dict[str, Union[str, Type]] = {"res": TextProcessor}
    output_schema: Dict[str, Union[str, Type]] = {"res_relation": TextProcessor}

    _FEWSHOT_SAMPLES: List[Dict[str, str]] = [
        {
            "context": "{'Bones are stable with mild degenerative changes of the spine.':{'Bones': 'anatomy', 'stable': 'concept', 'mild': 'concept', 'degenerative changes': 'disorder_present', 'spine': 'anatomy'}}",
            "response": "{'Bones are stable with mild degenerative changes of the spine.': [{'stable': 'Bones', 'relation':'modify'}, {'mild':'degenerative changes', 'relation':'modify'}, {'degenerative changes':'spine','relation':'located_at'}]}",
        },
        {
            "context": "{'A dense retrocardiac opacity remains present with slight blunting of the left costophrenic angle, suggestive of a small effusion.': {'dense': 'concept','retrocardiac': 'anatomy','opacity': 'disorder_present','slight': 'concept','blunting': 'disorder_present','left': 'concept','costophrenic': 'anatomy','angle': 'anatomy','small': 'concept','effusion': 'disorder_present'}}",
            "response": "{'A dense retrocardiac opacity remains present with slight blunting of the left costophrenic angle, suggestive of a small effusion.': [{'dense': 'opacity', 'relation': 'modify'}, {'opacity': 'retrocardiac', 'relation': 'located_at'}, {'slight': 'blunting', 'relation': 'modify'}, {'blunting': 'angle', 'relation': 'modify'}, {'left': 'costophrenic', 'relation': 'modify'}, {'small': 'effusion', 'relation': 'modify'}, {'effusion': 'costophrenic', 'relation': 'located_at'},{'opacity':'effusion','relation':'suggestive_of'},{'blunting':'effusion','relation':'suggestive_of'}]}",
        },
    ]


    def __call__(self, patient: Patient) -> List[Dict]:
        raise NotImplementedError(
            "RexKGGPT4RelationExtractionRadiology is a pipeline-style task. "
            "Use RexKGGPT4RelationExtractionRadiology.set_task(...) instead."
        )


    @classmethod
    def set_task(
        cls,
        input_json_file: str,
        save_json_file: str,
        post_proccess_json: str,
        api_type: str,
        api_version: str,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        model: str = "gpt-4o",
    ) -> Dict[str, Any]:
        """Run GPT-4 relation extraction over entity JSON and persist outputs."""
        input_path = Path(input_json_file).expanduser().resolve()
        if not input_path.exists():
            raise FileNotFoundError(f"Input JSON file not found: {input_path}")
        save_path = Path(save_json_file).expanduser().resolve()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        openai.api_type = api_type
        openai.api_version = api_version
        openai.api_key = api_key
        openai.api_base = api_base
        cls._evaluate_notes(model, input_json_file, save_json_file)
        cls._postprocess_json(save_json_file, post_proccess_json)
        return {
            "input_json_file": str(input_path),
            "save_json_file": str(save_path),
            "postprocess_json_file": post_proccess_json,
            # "summary_cost": summary_cost,
            "model": model,
        }
    

    @classmethod
    def _get_messages(cls, query: str) -> List[Dict[str, str]]:
        messages: List[Dict[str, str]] = [
            {
                "role": "system",
                "content": (
                    "You are a radiologist performing relation extraction of entities from the FINDINGS and IMPRESSION sections in the radiology report. "
                    "Here a clinical term can be in ['anatomy','disorder_present','disorder_notpresent','procedures','devices','concept', 'devices_present','devices_notpresent', 'size']. "
                    "And the relation can be in ['modify', 'located_at', 'suggestive_of']. "
                    "'suggestive_of' means the source entity (findings) may suggest the target entity (disease). "
                    "'located_at' means the source entity is located at the target entity. "
                    "'modify' denotes the source entity modifies the target entity. "
                    "Every time there is a 'modify' relationship between concept and anatomy, the direction should be concept -> anatomy. "
                    "For example, right pleural effusion , 'right' (concept), modify  'pleural' (anatomy), 'effusion' (disorder) located_at 'pleural' (anatomy). "
                    "Please ensure the direction of source/target entities is maintained correctly. "
                    "Given a piece of radiology text input in the JSON format: "
                    "{'sentence':{'entity':'entity_type'},'sentence':{'entity':'entity_type'}} "
                    "Please reply with the following JSON format: "
                    "{'sentence':[{source entity:'target entity',relation:'relation'},{source entity:'target entity',relation:'relation'}]}"
                ),
            }
        ]
        for sample in cls._FEWSHOT_SAMPLES:
            messages.append({"role": "user", "content": sample["context"]})
            messages.append({"role": "assistant", "content": sample["response"]})
        messages.append({"role": "user", "content": query})
        return messages


    @staticmethod
    def _estimate_cost(prompt_tokens: int, completion_tokens: int) -> float:
        input_cost = 0.005
        output_cost = 0.015
        return input_cost * prompt_tokens / 1000 + output_cost * completion_tokens / 1000


    @classmethod
    def _chatgpt_input(cls, model: str, messages: list) -> tuple:
        response = openai.ChatCompletion.create(
            engine=model,
            messages=messages,
            response_format={"type": "json_object"},
        )
        try:
            res = response["choices"][0]["message"]["content"]
            cost = cls._estimate_cost(
                response["usage"]["prompt_tokens"],
                response["usage"]["completion_tokens"],
            )
            return json.loads(res), cost
        except Exception:
            res = response["choices"][0]["message"]["content"]
            cost = cls._estimate_cost(
                response["usage"]["prompt_tokens"],
                response["usage"]["completion_tokens"],
            )
            return res, cost


    @classmethod
    def _test_prompt(cls, model: str, input_json: str) -> tuple:
        messages = cls._get_messages(input_json)
        return cls._chatgpt_input(model, messages)


    @classmethod
    def _evaluate_notes(cls, model: str, json_file: str, save_json_file: str) -> None:
        with open(json_file, "r") as file:
            json_data = json.load(file)
        note_id_list = list(json_data.keys())
        summary_cost = 0.0
        skipped_records = []

        try:
            with open(save_json_file, "r") as file:
                save_data_dict = json.load(file)
        except Exception:
            save_data_dict = {}

        counter = 0
        for select_id in note_id_list:
            if counter % 100 == 0:
                print("Extracting patient relation ", counter, " out of ", len(note_id_list))
            counter += 1

            if select_id in save_data_dict:
                continue
            else:
                data_dict_idx = json_data[select_id]
                save_data_dict_idx = data_dict_idx.copy()
                if counter % 100 == 0:
                    print("using input record ", counter, " out of 1000")
                    print("input_json: ", data_dict_idx["res"])

                input_json = data_dict_idx["res"]
                try:
                    res, cost = cls._test_prompt(model, json.dumps(input_json))
                    summary_cost += cost
                    save_data_dict_idx["res_relation"] = res
                    save_data_dict_idx["cost"] = cost
                    save_data_dict[select_id] = save_data_dict_idx
                except Exception as exc:
                    err_msg = str(exc)
                    lower_msg = err_msg.lower()
                    if (
                        "content management policy" in lower_msg
                        or "response was filtered" in lower_msg
                    ):
                        print(f"SKIP filtered records (does not adhere to OpenAI's Safety Standards)  {select_id}")
                    else:
                        print(f"[SKIP error] {select_id}: {err_msg}")
                    skipped_records.append({"id": select_id, "error": err_msg})

            with open(save_json_file, "w") as outfile:
                json.dump(save_data_dict, outfile, indent=4)

        skipped_file = str(Path(save_json_file).with_suffix("")) + "_skipped.json"
        with open(skipped_file, "w") as outfile:
            json.dump(skipped_records, outfile, indent=4)
        # print("SUMMARY COST: ", summary_cost)
        print("SKIPPED RECORDS: ", len(skipped_records), "saved to", skipped_file)


    @staticmethod
    def _convert_json_format(input_dict: dict):
        if len(input_dict) == 2:
            source_entity, target_entity = input_dict.items()
            return {
                "source entity": source_entity[0],
                "target entity": input_dict[source_entity[0]],
                "relation": input_dict[target_entity[0]],
            }
        elif len(input_dict) == 3:
            if "source" in input_dict:
                input_dict["source entity"] = input_dict["source"]
                input_dict["target entity"] = input_dict["target"]
                del input_dict["source"]
                del input_dict["target"]
            return input_dict
        else:
            print("Error:", input_dict)
            return None



    @classmethod
    def _flatten_dict(cls, d: dict) -> dict:
        flat_dict = {}
        for key, value in d.items():
            if isinstance(value, dict):
                flat_dict.update(cls._flatten_dict(value))
            else:
                flat_dict[key] = value
        return flat_dict



    @classmethod
    def _postprocess_json(cls, input_json_file: str, save_json_file: str) -> None:
        with open(input_json_file, "r") as file:
            json_data = json.load(file)
        note_id_list = list(json_data.keys())

        save_data_dict = {}
        post_counter = 0
        for select_id in note_id_list:
            if post_counter % 100 == 0:
                print("Post processing: ", post_counter, " out of ", 1000)
            data_dict_idx = json_data[select_id]
            save_data_dict_idx = data_dict_idx.copy()
            res_dict_idx = data_dict_idx["res"]
            res_relation_dict_idx = data_dict_idx["res_relation"]
            # GPT may return res_relation as a JSON string rather than a parsed dict
            if isinstance(res_relation_dict_idx, str):
                try:
                    res_relation_dict_idx = json.loads(res_relation_dict_idx)
                except json.JSONDecodeError:
                    post_counter += 1
                    continue
            # res may also come back as a JSON string
            if isinstance(res_dict_idx, str):
                try:
                    res_dict_idx = json.loads(res_dict_idx)
                except json.JSONDecodeError:
                    post_counter += 1
                    continue
            save_res_relation_dict_idx = res_relation_dict_idx.copy()
            sentence_list = list(res_dict_idx.keys())
            relation_sentence_list = list(res_relation_dict_idx.keys())
            if len(sentence_list) != len(relation_sentence_list):
                pass
            else:
                for sentence in res_dict_idx:
                    res_dict_idx[sentence] = cls._flatten_dict(res_dict_idx[sentence])
                for sentence in res_relation_dict_idx:
                    sentence_relation_list = res_relation_dict_idx[sentence]
                    save_sentence_relation_list = []
                    for sentence_relation_dict in sentence_relation_list:
                        save_sentence_relation_dict = cls._convert_json_format(sentence_relation_dict)
                        if save_sentence_relation_dict:
                            save_sentence_relation_list.append(save_sentence_relation_dict)
                    save_res_relation_dict_idx[sentence] = save_sentence_relation_list
                save_data_dict_idx["res_relation"] = save_res_relation_dict_idx
                save_data_dict[select_id] = save_data_dict_idx
            post_counter += 1
        with open(save_json_file, "w") as outfile:
            json.dump(save_data_dict, outfile, indent=4)



class RexKGStructureData(BaseTask):
    """Convert post-processed GPT-4 relation JSON into PURE-format train/test splits.

    This class mirrors the logic in ``src/ner/data/structure_data.py`` and
    exposes it via the PyHealth task-style API.
    """

    task_name: str = "rexkg_structure_data"
    input_schema: Dict[str, Union[str, Type]] = {"post_proccess_json": TextProcessor}
    output_schema: Dict[str, Union[str, Type]] = {"save_train_path": TextProcessor}

    def __call__(self, patient: Patient) -> List[Dict]:
        raise NotImplementedError(
            "RexKGStructureData is a pipeline-style task. "
            "Use RexKGStructureData.set_task(...) instead."
        )

    @classmethod
    def set_task(
        cls,
        post_proccess_json: str,
        save_train_path: str,
        save_test_path: str,
        test_slice_end: int = 100,
        train_slice_start: int = 100,
        train_slice_end: int = 1000,
    ) -> Dict[str, str]:
        """Run structure_data pipeline: convert JSON to PURE-format JSONL splits.

        Args:
            post_proccess_json: Path to the post-processed relation JSON file
                (output of RexKGGPT4RelationExtractionRadiology).
            save_train_path: Path to write the training split JSONL file.
            save_test_path: Path to write the test split JSONL file.
            test_slice_end: End index for test slice (default 100).
            train_slice_start: Start index for train slice (default 100).
            train_slice_end: End index for train slice (default 1000).
        """
        input_path = Path(post_proccess_json).expanduser().resolve()
        if not input_path.exists():
            raise FileNotFoundError(f"Input JSON file not found: {input_path}")

        train_path = Path(save_train_path).expanduser().resolve()
        test_path = Path(save_test_path).expanduser().resolve()
        train_path.parent.mkdir(parents=True, exist_ok=True)
        test_path.parent.mkdir(parents=True, exist_ok=True)

        with open(input_path, "r") as f:
            json_data = json.load(f)

        cls._preprocess_sentences_relation(
            cls._dict_slice(json_data, 0, test_slice_end),
            str(test_path),
        )
        cls._preprocess_sentences_relation(
            cls._dict_slice(json_data, train_slice_start, train_slice_end),
            str(train_path),
        )

        return {
            "save_train_path": str(train_path),
            "save_test_path": str(test_path),
        }

    @staticmethod
    def _dict_slice(d: dict, start: int, end: int) -> dict:
        keys = list(d.keys())[start:end]
        return {k: d[k] for k in keys}

    @staticmethod
    def _find_word_indices(sen: list, target_word: str):
        import re
        target_words = re.sub(
            r"(?<! )(?=[/,:,.,!?()])|(?<=[/,-,:,.,!?()])(?! )", r" ", target_word
        ).split()
        start_index = -1
        end_index = -1
        for i, word in enumerate(sen):
            if word == target_words[0] and (start_index == -1 or end_index == -1):
                if sen[i: i + len(target_words)] == target_words:
                    start_index = i
                    end_index = i + len(target_words) - 1
        return start_index, end_index

    @staticmethod
    def _is_number(s: str) -> bool:
        return s.isdigit()

    @staticmethod
    def _has_measurement_units(text: str) -> bool:
        import re
        pattern = r"\d+\s*(mm|cm|m|km|in|ft|yd|mi)"
        return bool(re.findall(pattern, text))

    @classmethod
    def _get_ner_list(cls, sen: list, sentence_info: dict):
        return_ner_dict = {}
        return_ner_list = []
        for entity in list(sentence_info.keys()):
            entity_copy = entity
            if entity.lower() in ["no evidence of", "no evidence", "no"]:
                continue
            elif "no evidence of " in entity.lower():
                entity = entity.replace("no evidence of ", "")
            elif "no evidence " in entity.lower():
                entity = entity.replace("no evidence ", "")
            elif "no " in entity.lower():
                entity = entity.replace("no ", "")
            entity_type = sentence_info[entity_copy].lower()

            if entity_type == "size":
                if (
                    "cm" in entity.split()
                    or "mm" in entity.split()
                    or "-cm" in entity
                    or "-mm" in entity
                ):
                    entity_type = "size"
                elif cls._has_measurement_units(entity) or cls._is_number(entity):
                    entity_type = "size"
                else:
                    entity_type = "concept"

            if entity_type in ["devices", "device"]:
                if "removed" in sen or "removal" in sen:
                    entity_type = "devices_notpresent"
                else:
                    entity_type = "devices_present"

            start_index, end_index = cls._find_word_indices(sen, entity.lower())
            return_ner_list.append([start_index, end_index, entity_type])
            return_ner_dict[entity] = [start_index, end_index]
        return return_ner_list, return_ner_dict

    @classmethod
    def _get_relation_list(cls, sen: list, ner_dict: dict, triplets_list: list) -> list:
        return_relation_list = []
        for triplets in triplets_list:
            source_entity = triplets["source entity"].lower()
            target_entity = triplets["target entity"].lower()
            relation = triplets["relation"]

            for attr in [source_entity, target_entity]:
                pass  # processed below per-variable

            def _strip_negation(e: str) -> str:
                if e in ["no evidence of", "no evidence", "no"]:
                    return ""
                for prefix in ["no evidence of ", "no evidence ", "no "]:
                    if prefix in e:
                        return e.replace(prefix, "")
                return e

            source_entity = _strip_negation(source_entity)
            target_entity = _strip_negation(target_entity)
            if not source_entity or not target_entity:
                continue

            source_start, source_end = cls._find_word_indices(sen, source_entity)
            target_start, target_end = cls._find_word_indices(sen, target_entity)
            if source_start == -1 or source_end == -1:
                try:
                    source_start, source_end = ner_dict[source_entity]
                except KeyError:
                    pass
            if target_start == -1 or target_end == -1:
                try:
                    target_start, target_end = ner_dict[target_entity]
                except KeyError:
                    pass
            return_relation_list.append(
                [source_start, source_end, target_start, target_end, relation]
            )
        return return_relation_list

    @classmethod
    def _preprocess_sentences_relation(cls, json_data: dict, save_json_file: str) -> None:
        import re
        note_id_list = list(json_data.keys())
        final_list = []
        sentence_idx = 0

        for select_id in tqdm(note_id_list):
            data_dict_idx = json_data[select_id]
            sentence_entity_dict = data_dict_idx["res"]
            sentence_relation_dict = data_dict_idx["res_relation"]
            sentence_list = list(sentence_entity_dict.keys())
            relation_sentence_list = list(sentence_relation_dict.keys())

            for sentence in sentence_list:
                sen = re.sub(
                    r"(?<! )(?=[/,-,:,.,!?()])|(?<=[/,-,:,.,!?()])(?! )",
                    r" ",
                    sentence.lower(),
                ).split()
                ner_list, ner_dict = cls._get_ner_list(sen, sentence_entity_dict[sentence])
                try:
                    temp_dict = {
                        "doc_key": str(sentence_idx),
                        "sentences": [sen],
                        "ner": [ner_list],
                    }
                    try:
                        relation_list = cls._get_relation_list(
                            sen, ner_dict, sentence_relation_dict[sentence]
                        )
                    except Exception:
                        relation_sentence = relation_sentence_list[
                            sentence_list.index(sentence)
                        ]
                        relation_list = cls._get_relation_list(
                            sen, ner_dict, sentence_relation_dict[relation_sentence]
                        )
                    temp_dict["relations"] = [relation_list]
                    final_list.append(temp_dict)
                    sentence_idx += 1
                except Exception:
                    print(sentence)

                if sentence_idx % 1000 == 0:
                    print(f"{sentence_idx + 1} sentences done")

        with open(save_json_file, "w") as outfile:
            for item in final_list:
                json.dump(item, outfile)
                outfile.write("\n")


def get_messages(query):
    fewshot_samples = [
        {
            'context': "{'Bones are stable with mild degenerative changes of the spine.':{'Bones': 'anatomy', 'stable': 'concept', 'mild': 'concept', 'degenerative changes': 'disorder_present', 'spine': 'anatomy'}}",
            'response': "{'Bones are stable with mild degenerative changes of the spine.': [{'stable': 'Bones', 'relation':'modify'}, {'mild':'degenerative changes', 'relation':'modify'}, {'degenerative changes':'spine','relation':'located_at'}]}"
        },
        {
            'context': "{'A dense retrocardiac opacity remains present with slight blunting of the left costophrenic angle, suggestive of a small effusion.': {'dense': 'concept','retrocardiac': 'anatomy','opacity': 'disorder_present','slight': 'concept','blunting': 'disorder_present','left': 'concept','costophrenic': 'anatomy','angle': 'anatomy','small': 'concept','effusion': 'disorder_present'}}",
            'response': "{'A dense retrocardiac opacity remains present with slight blunting of the left costophrenic angle, suggestive of a small effusion.': [{'dense': 'opacity', 'relation': 'modify'}, {'opacity': 'retrocardiac', 'relation': 'located_at'}, {'slight': 'blunting', 'relation': 'modify'}, {'blunting': 'angle', 'relation': 'modify'}, {'left': 'costophrenic', 'relation': 'modify'}, {'small': 'effusion', 'relation': 'modify'}, {'effusion': 'costophrenic', 'relation': 'located_at'},{'opacity':'effusion','relation':'suggestive_of'},{'blunting':'effusion','relation':'suggestive_of'}]}"
        }
    ]
    
    messages = [ 
            {"role": "system", "content": "You are a radiologist performing relation extraction of entities from the FINDINGS and IMPRESSION sections in the radiology report. \
                    Here a clinical term can be in ['anatomy','disorder_present','disorder_notpresent','procedures','devices','concept', 'devices_present','devices_notpresent', 'size']. \
                    And the relation can be in ['modify', 'located_at', 'suggestive_of']. \
                    'suggestive_of' means the source entity (findings) may suggest the target entity (disease). \
                    'located_at' means the source entity is located at the target entity. \
                    'modify' denotes the source entity modifies the target entity. \
                    Every time there is a 'modify' relationship between concept and anatomy, the direction should be concept -> anatomy. \
                    For example, right pleural effusion , 'right' (concept), modify  'pleural' (anatomy), 'effusion' (disorder) located_at 'pleural' (anatomy). \
                    Please ensure the direction of source/target entities is maintained correctly. \
                    Given a piece of radiology text input in the JSON format: \
                    {'sentence':{'entity':'entity_type'},'sentence':{'entity':'entity_type'}} \
                    Please reply with the following JSON format: \
                    {'sentence':[{source entity:'target entity',relation:'relation'},{source entity:'target entity',relation:'relation'}]} \
                "
                }
            ]
    
    for sample in fewshot_samples:
        messages.append({"role":"user", "content":sample['context']})
        messages.append({"role":"assistant", "content":sample['response']})
    messages.append({"role":"user", "content":query})
    return messages


def estimate_cost(prompt_tokens, completion_tokens):
    input_cost = 0.005
    output_cost = 0.015
    return (input_cost*prompt_tokens/1000 + output_cost*completion_tokens/1000)



def chatgpt_input(model, messages):
    
    response = openai.ChatCompletion.create(
        # model="gpt-3.5-turbo",
        engine = model,
        messages= messages,
        response_format={"type": "json_object"}
    )
    try:
        res = response["choices"][0]["message"]["content"]
        cost = estimate_cost(response["usage"]["prompt_tokens"],response["usage"]["completion_tokens"])
        return json.loads(res),cost
    except:

        res = response["choices"][0]["message"]["content"]
        cost = estimate_cost(response["usage"]["prompt_tokens"],response["usage"]["completion_tokens"])
        return res,cost


def test_prompt(model, input_json):
    messages = get_messages(input_json)
    res,cost = chatgpt_input(model, messages)
    return res,cost


def evaluate_notes(model, json_file,save_json_file):  
    
    with open(json_file, 'r') as file:
        json_data = json.load(file)
    note_id_list = list(json_data.keys())
    summary_cost = 0
    
    try:
        with open(save_json_file, 'r') as file:
            save_data_dict = json.load(file)
    except:
        save_data_dict = {}
    
    counter = 0
    for select_id in note_id_list:

        if counter % 100 == 0:
            print("Extracting patient relation ", counter, " out of ", len(note_id_list))
        counter = counter + 1
        
        if select_id in save_data_dict:
            pass 
        else:
            data_dict_idx = json_data[select_id]
            save_data_dict_idx = data_dict_idx.copy()
            if counter % 100 == 0:
                print("using input record ", counter, " out of 1000" )
                print("input_json: ", data_dict_idx['res'])   

            input_json = data_dict_idx['res']

            res,cost = test_prompt(model, json.dumps(input_json))
            summary_cost += cost
            save_data_dict_idx['res_relation'] = res
            save_data_dict_idx['cost'] = cost
            save_data_dict[select_id] = save_data_dict_idx
            
        with open(save_json_file, 'w') as outfile:
            json.dump(save_data_dict, outfile, indent=4) 
    # print('SUMMARY COST: ',summary_cost)


def convert_json_format(input_dict):
    if len(input_dict) == 2:
        source_entity, target_entity = input_dict.items()
        return {
            "source entity": source_entity[0],
            "target entity": input_dict[source_entity[0]],
            "relation": input_dict[target_entity[0]]
        }
    elif len(input_dict) == 3:
        if "source" in input_dict:
            input_dict["source entity"] = input_dict["source"]
            input_dict["target entity"] = input_dict["target"]
            del input_dict["source"]
            del input_dict["target"]
        return input_dict
    else:
        # print('Error:',input_dict)
        return None


def flatten_dict(d):
    flat_dict = {}
    
    for key, value in d.items():
        if isinstance(value, dict):
            # 如果值是一个字典，递归展开并合并到当前字典
            flat_dict.update(flatten_dict(value))
        else:
            flat_dict[key] = value
            
    return flat_dict


def postprocess_json(input_json_file,save_json_file):
    with open(input_json_file, 'r') as file:
        json_data = json.load(file)
    note_id_list = list(json_data.keys())
    
    save_data_dict = {}
    post_counter = 0
    for select_id in note_id_list:
        if post_counter % 100 == 0:
            print("Post processing: ", post_counter, " out of ", 1000)


        data_dict_idx = json_data[select_id]
        save_data_dict_idx = data_dict_idx.copy()
        res_dict_idx = data_dict_idx['res']
        res_relation_dict_idx = data_dict_idx['res_relation']
        save_res_relation_dict_idx = res_relation_dict_idx.copy()
        sentence_list = list(res_dict_idx.keys())
        relation_sentence_list = list(res_relation_dict_idx.keys())
        if len(sentence_list) != len(relation_sentence_list):
            pass 
        else:
            for sentence in res_dict_idx:
                res_dict_idx[sentence] = flatten_dict(res_dict_idx[sentence])
            for sentence in res_relation_dict_idx:
                sentence_relation_list = res_relation_dict_idx[sentence]
                # print(sentence,sentence_relation_list)
                save_sentence_relation_list = []
                for sentence_relation_dict in sentence_relation_list:
                    save_sentence_relation_dict = convert_json_format(sentence_relation_dict)
                    if save_sentence_relation_dict:
                        save_sentence_relation_list.append(save_sentence_relation_dict)
                save_res_relation_dict_idx[sentence] = save_sentence_relation_list
            save_data_dict_idx['res_relation'] = save_res_relation_dict_idx
            save_data_dict[select_id] = save_data_dict_idx
    
    with open(save_json_file, 'w') as outfile:
            json.dump(save_data_dict, outfile, indent=4) 
        


class RexKGUMLS(BaseTask):
    
    def __call__(self, patient: Patient) -> List[Dict]:
        raise NotImplementedError(
            "RexKGStructureData is a pipeline-style task. "
            "Use RexKGStructureData.set_task(...) instead."
        )

    @classmethod
    def set_task(
        cls,
        input_dir: str, 
        out_dir: str
    ) -> Dict[str, str]:
        print()
        
