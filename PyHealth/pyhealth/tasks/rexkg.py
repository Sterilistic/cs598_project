# Description: ReXKG entity and relation extraction tasks for radiology reports

import logging
import argparse
import importlib
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Union, Type, Any, Optional, Tuple, TYPE_CHECKING

import numpy as np
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

if TYPE_CHECKING:
    from ..datasets.rexkg import RexKGDataset

logger = logging.getLogger(__name__)


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
        context_window: int = 100,
        eval_metric: str = "f1",
        eval_per_epoch: int = 1,
        prediction_file: str = "predictions.json",
        train_mode: str = "random_sorted",
        add_new_tokens: bool = False,
        no_cuda: bool = False,
        seed: int = 42,
    ) -> Dict[str, Any]:
        _ = context_window
        _ = eval_metric
        _ = eval_per_epoch
        _ = train_mode
        _ = add_new_tokens

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        output_dir_path = Path(output_dir).expanduser().resolve()
        output_dir_path.mkdir(parents=True, exist_ok=True)

        log_path = output_dir_path / ("train.log" if do_train else "eval.log")
        file_handler = logging.FileHandler(log_path, "w")
        logger.addHandler(file_handler)

        try:
            train_docs = cls._load_json_records(train_file)
            train_examples = cls._build_relation_examples(
                docs=train_docs,
                use_gold_entities=True,
                include_gold_relations=True,
            )
            if len(train_examples) == 0:
                raise ValueError("No training relation pairs were generated from train_file")

            label_list = cls._infer_relation_label_list(task, train_examples)
            label2id = {lbl: i for i, lbl in enumerate(label_list)}
            id2label = {i: lbl for i, lbl in enumerate(label_list)}

            tokenizer = AutoTokenizer.from_pretrained(model, do_lower_case=do_lower_case, use_fast=True)
            train_texts = [e["text"] for e in train_examples]
            train_labels = [label2id.get(e["label"], 0) for e in train_examples]
            train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=max_seq_length)
            train_dataset = _RelationClsDataset(train_encodings, train_labels)

            relation_model = AutoModelForSequenceClassification.from_pretrained(
                model,
                num_labels=len(label_list),
                id2label=id2label,
                label2id=label2id,
            )

            device = torch.device("cuda" if torch.cuda.is_available() and not no_cuda else "cpu")
            relation_model.to(device)

            if do_train:
                train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
                optimizer = AdamW(relation_model.parameters(), lr=learning_rate)
                total_steps = max(1, int(len(train_loader) * max(1.0, num_train_epochs)))
                scheduler = get_linear_schedule_with_warmup(
                    optimizer,
                    int(total_steps * warmup_proportion),
                    total_steps,
                )

                relation_model.train()
                for _ in range(int(max(1.0, num_train_epochs))):
                    for batch in train_loader:
                        batch = {k: v.to(device) for k, v in batch.items()}
                        outputs = relation_model(**batch)
                        loss = outputs.loss
                        loss.backward()
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()

                relation_model.save_pretrained(output_dir_path)
                tokenizer.save_pretrained(output_dir_path)

            pred_file_path = output_dir_path / prediction_file
            eval_metrics: Dict[str, Any] = {}
            if do_eval:
                eval_source = Path(entity_output_dir).expanduser().resolve() / (
                    entity_predictions_test if eval_test else entity_predictions_dev
                )
                eval_docs = cls._load_json_records(str(eval_source))
                eval_examples = cls._build_relation_examples(
                    docs=eval_docs,
                    use_gold_entities=eval_with_gold,
                    include_gold_relations=True,
                )
                pred_ids = cls._predict_relation_ids(
                    relation_model,
                    tokenizer,
                    device,
                    eval_examples,
                    eval_batch_size,
                    max_seq_length,
                )
                pred_labels = [id2label.get(i, "no_relation") for i in pred_ids]
                cls._write_relation_predictions(eval_docs, eval_examples, pred_labels, pred_file_path)

                gold_ids = [label2id.get(e["label"], 0) for e in eval_examples]
                eval_metrics = cls._compute_relation_metrics(pred_ids, gold_ids)

            with (output_dir_path / "label_list.json").open("w", encoding="utf-8") as f:
                json.dump(label_list, f)

            result: Dict[str, Any] = {
                "output_dir": str(output_dir_path),
                "prediction_file": str(pred_file_path),
                "log_file": str(log_path),
                "task": task,
                "model": model,
            }
            result.update(eval_metrics)
            return result
        finally:
            logger.removeHandler(file_handler)
            file_handler.close()

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

