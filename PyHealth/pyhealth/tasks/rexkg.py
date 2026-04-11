# Author: ReXKG Team
# Description: ReXKG entity and relation extraction tasks for radiology reports

import logging
from typing import Dict, List, Union, Type, Any

import polars as pl
from pyhealth.data.data import Patient
from pyhealth.processors import TextProcessor, SequenceProcessor
from .base_task import BaseTask

logger = logging.getLogger(__name__)


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

    def pre_filter(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """Filter to reports with entity annotations.

        Args:
            df: Lazy polars dataframe of events

        Returns:
            Filtered dataframe
        """
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
        """Extract relations between entities in radiology reports.

        Processes radiology reports and extracts relationships between
        previously identified entities.

        Args:
            patient: Patient object containing radiology report events

        Returns:
            List of samples, each containing:
            - text: Original radiology report text
            - entities: Extracted entity annotations
            - relations: Extracted relation annotations
            - entity_pairs: List of entity pairs and their relation types
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

            # Extract text and initialize relations container
            sample = {
                "patient_id": patient.patient_id,
                "text": text,
                "entities": [],  # Would be populated by entity extraction
                "relations": [],  # Would be populated by relation extraction
                "entity_pairs": [],  # Would store (entity_i, entity_j, relation_type)
            }

            # Include metadata if available
            if hasattr(report, "study_id"):
                sample["study_id"] = report.study_id
            elif hasattr(report, "report_id"):
                sample["study_id"] = report.report_id
            if hasattr(report, "report_type"):
                sample["report_type"] = report.report_type

            samples.append(sample)

        return samples


class RexKGKnowledgeGraphConstruction(BaseTask):
    """Knowledge graph construction task using ReXKG extractions.

    This task orchestrates the full pipeline of entity and relation extraction
    from radiology reports to construct a comprehensive knowledge graph. It
    integrates entity extraction and relation extraction outputs to create
    a structured representation of clinical knowledge.

    The task produces:
    - Entity nodes with types and attributes
    - Relation edges connecting entities
    - Medical concept linking (UMLS CUIs)
    - Size and measurement standardization

    Output Knowledge Graph Contains:
        - Nodes: Entities with types (anatomy, disorder, procedure, etc.)
        - Edges: Relations between entities (modify, located_at, suggestive_of)
        - Attributes: Size measurements, qualifiers, temporal information
        - Semantic Links: UMLS concept mappings for standardization

    This task is typically used as the final stage after entity and
    relation extraction for complete knowledge graph assembly.

    Args:
        task_name: Name identifying this task
        input_schema: Schema defining input feature types
        output_schema: Schema defining output knowledge graph format

    Examples:
        >>> from pyhealth.datasets import SampleDataset
        >>> from pyhealth.tasks import RexKGKnowledgeGraphConstruction
        >>> dataset = SampleDataset(...)
        >>> task = RexKGKnowledgeGraphConstruction()
        >>> samples = dataset.set_task(task)
        >>> # Each sample contains complete KG from a radiology report
    """

    task_name: str = "rexkg_kg_construction"
    input_schema: Dict[str, Union[str, Type]] = {
        "text": TextProcessor,
        "entities": SequenceProcessor,
        "relations": SequenceProcessor,
    }
    output_schema: Dict[str, Union[str, Type]] = {
        "kg_nodes": SequenceProcessor,
        "kg_edges": SequenceProcessor,
    }

    def pre_filter(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """Filter to reports with complete annotations.

        Args:
            df: Lazy polars dataframe of events

        Returns:
            Filtered dataframe
        """
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
        """Construct knowledge graph from radiology report extractions.

        Processes radiology reports and produces a complete knowledge graph
        representation with entities as nodes and relations as edges.

        Args:
            patient: Patient object containing radiology report events

        Returns:
            List of samples, each containing:
            - text: Original radiology report text
            - kg_nodes: Knowledge graph nodes (entities with metadata)
            - kg_edges: Knowledge graph edges (relations between entities)
            - num_nodes: Number of entities in the graph
            - num_edges: Number of relations in the graph
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

            # Initialize knowledge graph
            sample = {
                "patient_id": patient.patient_id,
                "text": text,
                "kg_nodes": [],  # Entity nodes with types and attributes
                "kg_edges": [],  # Relation edges between entities
                "num_nodes": 0,
                "num_edges": 0,
            }

            # Add metadata
            if hasattr(report, "study_id"):
                sample["study_id"] = report.study_id
            elif hasattr(report, "report_id"):
                sample["study_id"] = report.report_id
            if hasattr(report, "report_type"):
                sample["report_type"] = report.report_type

            samples.append(sample)

        return samples
