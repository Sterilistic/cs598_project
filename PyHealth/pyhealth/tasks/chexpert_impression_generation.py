"""PyHealth task for radiology report impression generation using CheXpert Plus.

Dataset link:
    https://stanfordaimi.azurewebsites.net/datasets/5158c524-d3ab-4e02-96e9-6ee9efc110a1

Dataset paper: (please cite if you use this dataset)
    Chambon, P., et al. "CheXpert Plus: Augmenting a Large Chest X-ray Dataset
    with Text Radiology Reports, Patient Demographics and Additional Image
    Formats." arXiv:2405.19538, 2024.
"""

import logging
from typing import Dict, List

from pyhealth.data import Patient
from pyhealth.tasks import BaseTask

logger = logging.getLogger(__name__)


class CheXpertImpressionGeneration(BaseTask):
    """Task for generating radiology impression text from chest X-ray images.

    Given a chest X-ray image (and optionally the findings section as context),
    the model predicts the impression section of the radiology report.  This
    formulation follows the standard conditional text-generation setup used in
    the CheXpert Plus benchmark.

    Attributes:
        task_name (str): Unique name of this task.
        input_schema (Dict[str, str]): Input features consumed by this task.
        output_schema (Dict[str, str]): Output features predicted by this task.

    Examples:
        >>> from pyhealth.datasets import CheXpertDataset
        >>> from pyhealth.tasks import CheXpertImpressionGeneration
        >>> dataset = CheXpertDataset(root="/path/to/chexpert")
        >>> task = CheXpertImpressionGeneration()
        >>> samples = dataset.set_task(task)
        >>> print(samples[0])
    """

    task_name: str = "CheXpertImpressionGeneration"
    input_schema: Dict[str, str] = {"image": "image"}
    output_schema: Dict[str, str] = {"impression": "text"}

    def __call__(self, patient: Patient) -> List[Dict]:
        """Generate impression-generation samples for a single patient.

        Each visit (study) with a non-empty impression section becomes one
        sample.  Visits with missing image paths or missing impression text
        are silently skipped.

        Args:
            patient: A :class:`~pyhealth.data.Patient` object containing one
                or more ``"chexpert"`` events.

        Returns:
            List[Dict]: A list of sample dictionaries, each containing:
                - ``"image"`` (str): Absolute path to the chest X-ray image.
                - ``"impression"`` (str): Target impression text.
                - ``"findings"`` (str): Findings section text (may be empty).
                - ``"frontal_lateral"`` (str): ``"Frontal"`` or ``"Lateral"``.
                - ``"ap_pa"`` (str): ``"AP"`` or ``"PA"`` (frontal only).
                - ``"patient_id"`` (str): De-identified patient identifier.
        """
        events = patient.get_events(event_type="chexpert")

        samples = []
        for event in events:
            image_path = event["path_to_image"] if "path_to_image" in event else ""
            impression = event["section_impression"] if "section_impression" in event else ""

            # Skip samples without a valid image path or impression text
            if not image_path or not isinstance(impression, str) or not impression.strip():
                continue

            findings = event["section_findings"] if "section_findings" in event else ""
            frontal_lateral = event["frontal_lateral"] if "frontal_lateral" in event else ""
            ap_pa = event["ap_pa"] if "ap_pa" in event else ""

            samples.append(
                {
                    "image": image_path,
                    "impression": impression.strip(),
                    "findings": findings or "",
                    "frontal_lateral": frontal_lateral or "",
                    "ap_pa": ap_pa or "",
                    "patient_id": patient.patient_id,
                }
            )

        return samples
