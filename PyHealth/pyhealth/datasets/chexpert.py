"""CheXpert Plus dataset for PyHealth.

This module provides the CheXpertDataset class for loading and processing
CheXpert Plus chest X-ray data for machine learning tasks including
multilabel pathology classification and radiology report generation.

Dataset paper:
    Chambon, P., et al. "CheXpert Plus: Augmenting a Large Chest X-ray Dataset
    with Text Radiology Reports, Patient Demographics and Additional Image
    Formats." arXiv:2405.19538, 2024.

Dataset link:
    https://stanfordaimi.azurewebsites.net/datasets/5158c524-d3ab-4e02-96e9-6ee9efc110a1
"""

import logging
import os
from pathlib import Path
from typing import List, Optional

import pandas as pd

from .base_dataset import BaseDataset

logger = logging.getLogger(__name__)


class CheXpertDataset(BaseDataset):
    """CheXpert Plus chest X-ray dataset for multi-task clinical imaging.

    CheXpert Plus is an augmented version of the original Stanford CheXpert
    dataset, extending it with structured radiology reports, patient
    demographic metadata, and additional image formats. It contains over
    223,000 chest radiographs from more than 64,000 patients.

    The dataset supports two primary task families:
      - **Multilabel pathology classification**: predict presence/absence of
        14 radiological findings from the X-ray image.
      - **Report generation**: produce a radiology impression or findings
        section given the chest X-ray image.

    Dataset link:
        https://stanfordaimi.azurewebsites.net/datasets/5158c524-d3ab-4e02-96e9-6ee9efc110a1

    Note:
        CheXpert Plus requires registration and license agreement. After
        downloading, place the metadata CSV (``df_chexpert_plus_240401.csv``)
        and the image directory (``CheXpert-v1.0/``) together under ``root``.

    Args:
        root: Root directory of the raw data. Must contain
            ``df_chexpert_plus_240401.csv`` and the image directory referenced
            by the ``path_to_image`` column.
        tables: Optional list of additional tables to load beyond the default
            ``"chexpert"`` table.
        dataset_name: Optional name for the dataset. Defaults to
            ``"chexpert"``.
        config_path: Optional path to a custom YAML configuration file. If not
            provided, uses the bundled ``configs/chexpert.yaml``.

    Attributes:
        root: Root directory of the raw data.
        dataset_name: Name of the dataset.
        config_path: Path to the YAML configuration file.

    Examples:
        >>> from pyhealth.datasets import CheXpertDataset
        >>> dataset = CheXpertDataset(root="/path/to/chexpert")
        >>> dataset.stats()
        >>> samples = dataset.set_task()
        >>> print(samples[0])
    """

    def __init__(
        self,
        root: str,
        tables: List[str] = None,
        dataset_name: Optional[str] = None,
        config_path: Optional[str] = None,
        **kwargs,
    ) -> None:
        if config_path is None:
            logger.info("No config path provided, using default config")
            config_path = Path(__file__).parent / "configs" / "chexpert.yaml"

        pyhealth_csv = os.path.join(root, "chexpert-pyhealth.csv")
        if not os.path.exists(pyhealth_csv):
            logger.info("Preparing CheXpert metadata...")
            self.prepare_metadata(root)

        default_tables = ["chexpert"]
        tables = default_tables + (tables or [])

        super().__init__(
            root=root,
            tables=tables,
            dataset_name=dataset_name or "chexpert",
            config_path=config_path,
            **kwargs,
        )

    @staticmethod
    def prepare_metadata(root: str) -> None:
        """Prepare the standardized metadata CSV for the CheXpert dataset.

        Reads the raw ``df_chexpert_plus_240401.csv`` metadata file, selects
        the relevant columns, converts image paths to absolute paths relative
        to ``root``, and writes ``chexpert-pyhealth.csv`` to ``root``.

        Args:
            root: Root directory containing ``df_chexpert_plus_240401.csv`` and
                the image directory.

        Raises:
            FileNotFoundError: If no recognised CheXpert metadata CSV can be
                found inside ``root``.
        """
        possible_files = [
            "df_chexpert_plus_240401.csv",
            "chexpert_plus.csv",
            "CheXpert-v1.0/train.csv",
            "CheXpert-v1.0/valid.csv",
            "train.csv",
        ]

        raw_file = None
        for fname in possible_files:
            fpath = os.path.join(root, fname)
            if os.path.exists(fpath):
                raw_file = fpath
                break

        if raw_file is None:
            logger.warning(
                f"No raw CheXpert metadata file found in {root}. "
                "Please download from "
                "https://stanfordaimi.azurewebsites.net/datasets/"
                "5158c524-d3ab-4e02-96e9-6ee9efc110a1 and place "
                "df_chexpert_plus_240401.csv in the root directory."
            )
            # Write an empty placeholder so subsequent loads don't re-trigger
            pd.DataFrame(
                columns=[
                    "deid_patient_id",
                    "path_to_image",
                    "frontal_lateral",
                    "ap_pa",
                    "patient_report_date_order",
                    "age",
                    "sex",
                    "race",
                    "ethnicity",
                    "interpreter_needed",
                    "insurance_type",
                    "recent_bmi",
                    "deceased",
                    "split",
                    "section_findings",
                    "section_impression",
                    "report",
                ]
            ).to_csv(os.path.join(root, "chexpert-pyhealth.csv"), index=False)
            return

        logger.info(f"Processing CheXpert metadata file: {raw_file}")
        df = pd.read_csv(raw_file, low_memory=False)

        # ------------------------------------------------------------------
        # Column normalisation: CheXpert Plus uses these names directly.
        # The original CheXpert (train.csv / valid.csv) uses "Path" and has
        # explicit numeric pathology label columns; remap them if present.
        # ------------------------------------------------------------------
        column_mapping = {
            # Original CheXpert column names
            "Path": "path_to_image",
            "Patient ID": "deid_patient_id",
            "Sex": "sex",
            "Age": "age",
            "Frontal/Lateral": "frontal_lateral",
            "AP/PA": "ap_pa",
        }
        rename_dict = {k: v for k, v in column_mapping.items() if k in df.columns}
        df = df.rename(columns=rename_dict)

        # Ensure a patient ID column exists (use row index as fallback)
        if "deid_patient_id" not in df.columns:
            if "path_to_image" in df.columns:
                # Extract patient identifier from the path, e.g.
                # "train/patient00001/study1/view1_frontal.jpg" -> "patient00001"
                df["deid_patient_id"] = (
                    df["path_to_image"]
                    .str.split("/")
                    .str[1]
                    .fillna(df.index.astype(str))
                )
            else:
                df["deid_patient_id"] = df.index.astype(str)

        # Make image paths absolute so they can be resolved at load time without
        # knowing the root directory again
        if "path_to_image" in df.columns:
            df["path_to_image"] = df["path_to_image"].apply(
                lambda p: os.path.join(root, p) if isinstance(p, str) else p
            )

        # Select only the columns the YAML config declares, where available
        desired_cols = [
            "deid_patient_id",
            "path_to_image",
            "frontal_lateral",
            "ap_pa",
            "patient_report_date_order",
            "age",
            "sex",
            "race",
            "ethnicity",
            "interpreter_needed",
            "insurance_type",
            "recent_bmi",
            "deceased",
            "split",
            "section_findings",
            "section_impression",
            "report",
        ]
        available_cols = [c for c in desired_cols if c in df.columns]
        df_out = df[available_cols]

        output_path = os.path.join(root, "chexpert-pyhealth.csv")
        df_out.to_csv(output_path, index=False)
        logger.info(f"Saved {len(df_out)} records to {output_path}")

    @property
    def default_task(self):
        """Returns the default task for the CheXpert dataset.

        Returns:
            CheXpertImpressionGeneration: default radiology report generation
            task using the impression section as the target.
        """
        from pyhealth.tasks import CheXpertImpressionGeneration

        return CheXpertImpressionGeneration()
