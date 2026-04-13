import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from ..tasks import RexKGEntityExtractionRadiology
from .base_dataset import BaseDataset

logger = logging.getLogger(__name__)


class RexKGDataset(BaseDataset):
    """Dataset wrapper for ReXKG radiology report extraction.

    Args:
        root:
            Either:
            1) absolute/relative path to a source CSV file, or
            2) directory containing a prebuilt ``rexkg-pyhealth.csv``.
        dataset_name: Optional dataset name. Defaults to ``"rexkg"``.
        config_path: Optional custom config path. Defaults to bundled config.
        cache_dir: Optional directory for cached PyHealth artifacts.
        num_workers: Number of workers used by preprocessing.
        dev: Whether to enable PyHealth dev mode.
    """

    def __init__(
        self,
        root: str,
        train_data: Optional[str] = None,
        dev_data: Optional[str] = None,
        test_data: Optional[str] = None,
        dataset_name: Optional[str] = None,
        config_path: Optional[str] = None,
        cache_dir: Optional[str] = None,
        num_workers: int = 1,
        dev: bool = False,
    ) -> None:
        input_path = Path(root).expanduser().resolve()

        self.train_data = str(Path(train_data).expanduser().resolve()) if train_data else None
        self.dev_data = str(Path(dev_data).expanduser().resolve()) if dev_data else None
        self.test_data = str(Path(test_data).expanduser().resolve()) if test_data else None
        self.split_data_path = str(input_path) if input_path.suffix.lower() in {".json", ".jsonl"} else None

        # Entity/relation legacy pipelines consume JSON/JSONL split files directly.
        # Support constructing a lightweight RexKGDataset wrapper for those files.
        if self.split_data_path is not None:
            self.root = str(input_path.parent)
            self.dataset_name = dataset_name or "rexkg"
            self.config_path = str(config_path) if config_path is not None else None
            self.cache_dir = cache_dir
            self.num_workers = num_workers
            self.dev = dev
            return

        if config_path is None:
            logger.info("No config path provided, using default RexKG config")
            config_path = Path(__file__).parent / "configs" / "rexkg_dataset.yaml"

        if input_path.is_file():
            data_dir = input_path.parent
            source_csv = input_path
        else:
            data_dir = input_path
            source_csv = None

        prepared_csv = data_dir / "rexkg-pyhealth.csv"
        if not prepared_csv.exists():
            if source_csv is None:
                raise FileNotFoundError(
                    f"Missing prepared file: {prepared_csv}. "
                    f"Pass `root` as a CSV filepath to auto-prepare metadata."
                )
            self.prepare_metadata(root=str(data_dir), source_csv=str(source_csv))

        super().__init__(
            root=str(data_dir),
            tables=["radiology_reports"],
            dataset_name=dataset_name or "rexkg",
            config_path=config_path,
            cache_dir=cache_dir,
            num_workers=num_workers,
            dev=dev,
        )

    @staticmethod
    def prepare_metadata(root: str, source_csv: str) -> None:
        """Prepare standardized report-level CSV for PyHealth.

        Args:
            root: Directory where output ``rexkg-pyhealth.csv`` will be written.
            source_csv: Explicit source CSV file path.

        Raises:
            FileNotFoundError: If source CSV does not exist.
        """
        root_path = Path(root).expanduser().resolve()
        raw_file = Path(source_csv).expanduser().resolve()

        if not raw_file.exists() or not raw_file.is_file():
            raise FileNotFoundError(f"Provided source CSV does not exist: {raw_file}")

        df = pd.read_csv(raw_file)

        text_column_candidates = ["section_findings", "report", "section_impression"]
        text_column = next((col for col in text_column_candidates if col in df.columns), None)
        if text_column is None:
            raise ValueError(
                "Could not find a report text column. Expected one of: "
                f"{text_column_candidates}"
            )

        output = pd.DataFrame()
        output["patient_id"] = (
            df["deid_patient_id"].astype(str)
            if "deid_patient_id" in df.columns
            else [f"report_patient_{idx}" for idx in range(len(df))]
        )

        output["study_id"] = (
            df["study_id"].astype(str)
            if "study_id" in df.columns
            else df["path_to_image"].astype(str)
            if "path_to_image" in df.columns
            else [f"study_{idx}" for idx in range(len(df))]
        )

        output["report_id"] = [f"rexkg_report_{idx}" for idx in range(len(df))]
        output["text"] = df[text_column].fillna("").astype(str)
        output["report_type"] = "RADIOLOGY"

        optional_columns = ["path_to_image", "split", "section_findings", "section_impression", "report"]
        for column_name in optional_columns:
            output[column_name] = (
                df[column_name].fillna("").astype(str) if column_name in df.columns else ""
            )

        output = output[output["text"].str.strip() != ""].reset_index(drop=True)
        output.to_csv(root_path / "rexkg-pyhealth.csv", index=False)

    @property
    def default_task(self) -> RexKGEntityExtractionRadiology:
        """Return the default RexKG task for this dataset."""
        return RexKGEntityExtractionRadiology()
