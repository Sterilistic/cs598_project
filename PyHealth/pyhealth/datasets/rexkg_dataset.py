import logging
import os
from pathlib import Path
from typing import Optional

import pandas as pd

from ..tasks import RexKGEntityExtractionRadiology
from .base_dataset import BaseDataset

logger = logging.getLogger(__name__)


class RexKGDataset(BaseDataset):
    """Dataset wrapper for ReXKG radiology report extraction.

    This dataset standardizes CheXpert Plus style radiology report metadata into a
    single PyHealth event table named ``radiology_reports`` so the RexKG tasks can
    consume it directly.

    The wrapper is intentionally lightweight: each row in the source CSV becomes a
    single report-level event with text fields and identifiers preserved as event
    attributes. The expected raw input is the CheXpert Plus metadata CSV used by the
    original ReXKG pipeline, typically ``df_chexpert_plus_240401.csv``.

    Args:
        root: Directory containing the raw ReXKG/CheXpert Plus CSV.
        dataset_name: Optional dataset name. Defaults to ``"rexkg"``.
        config_path: Optional custom config path. Defaults to the bundled
            ``configs/rexkg_dataset.yaml``.
        cache_dir: Optional directory for cached PyHealth artifacts.
        num_workers: Number of workers used by PyHealth preprocessing.
        dev: Whether to enable PyHealth dev mode.

    Examples:
        >>> from pyhealth.datasets import RexKGDataset
        >>> dataset = RexKGDataset(root="/path/to/rexkg_data")
        >>> task_dataset = dataset.set_task()
    """

    def __init__(
        self,
        root: str,
        dataset_name: Optional[str] = None,
        config_path: Optional[str] = None,
        cache_dir: Optional[str] = None,
        num_workers: int = 1,
        dev: bool = False,
    ) -> None:
        if config_path is None:
            logger.info("No config path provided, using default RexKG config")
            config_path = Path(__file__).parent / "configs" / "rexkg_dataset.yaml"

        prepared_csv = os.path.join(root, "rexkg-pyhealth.csv")
        if not os.path.exists(prepared_csv):
            self.prepare_metadata(root)

        super().__init__(
            root=root,
            tables=["radiology_reports"],
            dataset_name=dataset_name or "rexkg",
            config_path=config_path,
            cache_dir=cache_dir,
            num_workers=num_workers,
            dev=dev,
        )

    @staticmethod
    def prepare_metadata(root: str) -> None:
        """Prepare a standardized report-level CSV for PyHealth.

        The method looks for the raw CheXpert Plus CSV used by ReXKG and writes a
        simplified ``rexkg-pyhealth.csv`` file that matches the bundled dataset config.

        Args:
            root: Directory containing raw ReXKG input CSV files.

        Raises:
            FileNotFoundError: If a supported raw CSV cannot be found.
        """
        possible_files = [
            "df_chexpert_plus_240401.csv",
            "chexpert_plus.csv",
            "df_chexpert_plus_onlyfindings.csv",
        ]

        raw_file = None
        for file_name in possible_files:
            candidate = os.path.join(root, file_name)
            if os.path.exists(candidate):
                raw_file = candidate
                break

        if raw_file is None:
            raise FileNotFoundError(
                f"No ReXKG source CSV found in {root}. Expected one of: {possible_files}"
            )

        df = pd.read_csv(raw_file)

        text_column_candidates = ["section_findings", "report", "section_impression"]
        text_column = next((col for col in text_column_candidates if col in df.columns), None)
        if text_column is None:
            raise ValueError(
                "Could not find a report text column. Expected one of: "
                f"{text_column_candidates}"
            )

        output = pd.DataFrame()
        if "deid_patient_id" in df.columns:
            output["patient_id"] = df["deid_patient_id"].astype(str)
        else:
            output["patient_id"] = [f"report_patient_{idx}" for idx in range(len(df))]

        if "study_id" in df.columns:
            output["study_id"] = df["study_id"].astype(str)
        elif "path_to_image" in df.columns:
            output["study_id"] = df["path_to_image"].astype(str)
        else:
            output["study_id"] = [f"study_{idx}" for idx in range(len(df))]

        output["report_id"] = [f"rexkg_report_{idx}" for idx in range(len(df))]
        output["text"] = df[text_column].fillna("").astype(str)
        output["report_type"] = "RADIOLOGY"

        optional_columns = {
            "path_to_image": "",
            "split": "",
            "section_findings": "",
            "section_impression": "",
            "report": "",
        }
        for column_name, default_value in optional_columns.items():
            if column_name in df.columns:
                output[column_name] = df[column_name].fillna(default_value).astype(str)
            else:
                output[column_name] = default_value

        output = output[output["text"].str.strip() != ""].reset_index(drop=True)
        output.to_csv(os.path.join(root, "rexkg-pyhealth.csv"), index=False)

    @property
    def default_task(self) -> RexKGEntityExtractionRadiology:
        """Return the default RexKG task for this dataset."""
        return RexKGEntityExtractionRadiology()
