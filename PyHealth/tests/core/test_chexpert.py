"""
Unit tests for CheXpertDataset and CheXpertImpressionGeneration.

Validation approach mirrors test_chestxray14.py:
    - Fake JPEG images and a chexpert-pyhealth.csv (with absolute image paths)
        are created in setUpClass and deleted in tearDownClass so no machine-
        specific paths are committed to the repository.
    - Tests cover: initialization, stats(), patient counts, attribute retrieval,
        default_task type, task output length, task output keys, and the edge case
        where a missing impression causes a sample to be skipped.
"""

import tempfile
import csv
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from pyhealth.datasets import CheXpertDataset
from pyhealth.tasks import CheXpertImpressionGeneration


class TestCheXpertDataset(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).parent.parent.parent / "test-resources" / "core" / "chexpert"
        (cls.root / "images").mkdir(parents=True, exist_ok=True)
        cls._generate_fake_images()
        cls._write_fixture_csv()
        cls.cache_dir = tempfile.TemporaryDirectory()
        cls.dataset = CheXpertDataset(
            root=str(cls.root), cache_dir=cls.cache_dir.name
        )
        cls.samples = cls.dataset.set_task(CheXpertImpressionGeneration())

    @classmethod
    def tearDownClass(cls):
        cls.samples.close()
        cls.cache_dir.cleanup()
        cls._delete_fake_images()
        pyhealth_csv = cls.root / "chexpert-pyhealth.csv"
        if pyhealth_csv.exists():
            pyhealth_csv.unlink()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    # Fixture rows: (deid_patient_id, image_filename, frontal_lateral, ap_pa,
    #                order, age, sex, split, section_findings, section_impression)
    FIXTURE_ROWS = [
        ("patient1", "patient1_study1_view1.jpg", "Frontal", "AP",  "1", "45", "Female", "train",
         "Lungs are clear.",        "No acute cardiopulmonary process."),
        ("patient1", "patient1_study2_view1.jpg", "Frontal", "PA",  "2", "46", "Female", "train",
         "Mild interstitial edema.", "Mild pulmonary edema."),
        ("patient2", "patient2_study1_view1.jpg", "Frontal", "AP",  "1", "62", "Male",   "valid",
         "No pleural effusion.",     "No acute findings."),
        # patient3: empty impression — must be skipped by the task
        ("patient3", "patient3_study1_view1.jpg", "Frontal", "AP",  "1", "30", "Male",   "train",
         "Bilateral infiltrates.",   ""),
    ]

    @classmethod
    def _generate_fake_images(cls):
        for row in cls.FIXTURE_ROWS:
            name = row[1]
            img = Image.fromarray(
                np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
            )
            img.save(cls.root / "images" / name)

    @classmethod
    def _write_fixture_csv(cls):
        """Write chexpert-pyhealth.csv with absolute image paths."""
        fieldnames = [
            "deid_patient_id", "path_to_image", "frontal_lateral", "ap_pa",
            "patient_report_date_order", "age", "sex", "race", "ethnicity",
            "interpreter_needed", "insurance_type", "recent_bmi", "deceased",
            "split", "section_findings", "section_impression", "report",
        ]
        out_path = cls.root / "chexpert-pyhealth.csv"
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for pid, fname, fl, ap, order, age, sex, split, findings, impression in cls.FIXTURE_ROWS:
                writer.writerow({
                    "deid_patient_id": pid,
                    "path_to_image": str(cls.root / "images" / fname),
                    "frontal_lateral": fl,
                    "ap_pa": ap,
                    "patient_report_date_order": order,
                    "age": age,
                    "sex": sex,
                    "race": "White",
                    "ethnicity": "Not Hispanic",
                    "interpreter_needed": "No",
                    "insurance_type": "Commercial",
                    "recent_bmi": "24.5",
                    "deceased": "No",
                    "split": split,
                    "section_findings": findings,
                    "section_impression": impression,
                    "report": f"Full report for {pid}",
                })

    @classmethod
    def _delete_fake_images(cls):
        for row in cls.FIXTURE_ROWS:
            p = cls.root / "images" / row[1]
            if p.exists():
                p.unlink()

    # ------------------------------------------------------------------
    # Dataset-level tests
    # ------------------------------------------------------------------

    def test_dataset_name(self):
        self.assertEqual(self.dataset.dataset_name, "chexpert")

    def test_stats(self):
        """stats() should run without raising an exception."""
        self.dataset.stats()

    def test_num_patients(self):
        """Fixture has 3 unique patient IDs."""
        self.assertEqual(len(self.dataset.unique_patient_ids), 3)

    def test_get_patient1_event_count(self):
        """patient1 has 2 studies in the fixture."""
        events = self.dataset.get_patient("patient1").get_events(
            event_type="chexpert"
        )
        self.assertEqual(len(events), 2)

    def test_get_patient2_event_count(self):
        """patient2 has 1 study in the fixture."""
        events = self.dataset.get_patient("patient2").get_events(
            event_type="chexpert"
        )
        self.assertEqual(len(events), 1)

    def test_patient2_attributes(self):
        """Verify several attribute values for patient2's single event."""
        event = self.dataset.get_patient("patient2").get_events(
            event_type="chexpert"
        )[0]
        self.assertEqual(event["frontal_lateral"], "Frontal")
        self.assertEqual(event["ap_pa"], "AP")
        self.assertEqual(event["age"], "62")
        self.assertEqual(event["sex"], "Male")
        self.assertEqual(event["split"], "valid")

    def test_default_task(self):
        self.assertIsInstance(
            self.dataset.default_task, CheXpertImpressionGeneration
        )

    # ------------------------------------------------------------------
    # Task-level tests
    # ------------------------------------------------------------------

    def test_task_sample_count(self):
        """3 of the 4 fixture rows have a non-empty impression; patient3 is skipped."""
        self.assertEqual(len(self.samples), 3)

    def test_sample_keys(self):
        """Every sample must contain the expected keys."""
        required_keys = {"image", "impression", "findings", "frontal_lateral", "ap_pa", "patient_id"}
        for sample in self.samples:
            self.assertTrue(required_keys.issubset(sample.keys()), sample.keys())

    def test_sample_impression_nonempty(self):
        """All returned samples must have a non-empty impression string."""
        for sample in self.samples:
            self.assertIsInstance(sample["impression"], str)
            self.assertGreater(len(sample["impression"].strip()), 0)

    def test_patient1_samples(self):
        """patient1 has 2 valid studies so 2 samples should appear."""
        patient1_samples = [s for s in self.samples if s["patient_id"] == "patient1"]
        self.assertEqual(len(patient1_samples), 2)

    def test_patient3_skipped(self):
        """patient3's impression is empty so it must not produce any sample."""
        patient3_samples = [s for s in self.samples if s["patient_id"] == "patient3"]
        self.assertEqual(len(patient3_samples), 0)

    def test_task_attributes(self):
        task = CheXpertImpressionGeneration()
        self.assertEqual(task.task_name, "CheXpertImpressionGeneration")
        self.assertIn("image", task.input_schema)
        self.assertIn("impression", task.output_schema)


if __name__ == "__main__":
    unittest.main()
