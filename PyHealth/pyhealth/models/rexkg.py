"""ReXKG (Radiology Examination Knowledge Graph) entity and relation extraction model.

This module provides a transformer-based model for extracting entities and relations
from clinical radiology reports. ReXKG performs Named Entity Recognition (NER) and
Relation Extraction (RE) to construct knowledge graphs from unstructured text.

Key Features:
    - Transformer-based architecture for entity and relation extraction
    - Supports multi-task learning (NER + RE)
    - Compatible with pretrained clinical language models
    - Outputs structured entities and relations for KG construction
    - Integrates with PyHealth's multimodal pipeline

Entity Types:
    - anatomy: Anatomical terms and body regions
    - disorder_present: Confirmed disorders/diseases
    - disorder_notpresent: Excluded or ruled-out disorders
    - concept: Medical/clinical concepts
    - procedures: Medical procedures
    - devices_present: Medical devices present
    - devices_notpresent: Medical devices absent
    - size: Size/measurement-related terms

Relation Types:
    - modify: Modifying relationships between entities
    - located_at: Spatial location relationships
    - suggestive_of: Diagnostic suggestion relationships

Example:
    >>> from pyhealth.datasets import SampleDataset
    >>> from pyhealth.models import RexKG
    >>> dataset = SampleDataset(...)
    >>> model = RexKG(dataset=dataset, embedding_dim=256)
    >>> # Forward pass for batch
    >>> batch = {"text": text_tensor}
    >>> output = model(**batch)
"""

from typing import Optional, Dict, List, Any, Tuple
import logging

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

from .base_model import BaseModel
from ..datasets import SampleDataset

logger = logging.getLogger(__name__)


class RexKG(BaseModel):
    """Radiology Examination Knowledge Graph entity and relation extraction model.

    ReXKG extracts entities and their relationships from clinical radiology reports
    using a transformer-based architecture. It supports both Named Entity Recognition (NER)
    and Relation Extraction (RE) tasks for knowledge graph construction.

    The model uses a pretrained clinical language model (default: Bio_ClinicalBERT)
    as the encoder and adds task-specific heads for entity classification and
    relation extraction.

    Args:
        dataset (SampleDataset): The dataset containing text inputs and entity/relation
            labels.
        embedding_dim (int): Dimension of the transformer embeddings. Default is 768
            (Bio_ClinicalBERT default).
        num_entity_types (int): Number of entity types to classify (includes BIO tags).
            Default is 25 (8 entity types * 3 BIO tags + 1 O tag).
        num_relation_types (int): Number of relation types to classify (includes
            'no_relation' class). Default is 4 (3 relation types + 1 no_relation).
        pretrained_model (str): Hugging Face model identifier for pretrained encoder.
            Default is "emilyalsentzer/Bio_ClinicalBERT".
        freeze_encoder (bool): Whether to freeze pretrained encoder weights.
            Default is True for efficient fine-tuning.
        dropout_rate (float): Dropout rate for regularization. Default is 0.1.
        use_crf (bool): Whether to use CRF layer for sequence labeling. Default is False.

    Attributes:
        encoder: Transformer encoder from pretrained model
        entity_head: Classification head for entity extraction (NER)
        relation_head: Classification head for relation extraction (RE)
        dropout: Dropout layer for regularization

    Examples:
        >>> from pyhealth.datasets import SampleDataset
        >>> from pyhealth.models import RexKG
        >>> # Initialize with a dataset
        >>> dataset = SampleDataset(...)
        >>> model = RexKG(
        ...     dataset=dataset,
        ...     embedding_dim=768,
        ...     pretrained_model="emilyalsentzer/Bio_ClinicalBERT"
        ... )
        >>> # Training forward pass
        >>> batch = {
        ...     "text": input_ids,  # Shape: [batch_size, seq_len]
        ...     "entity_labels": entity_ids,  # Shape: [batch_size, seq_len]
        ...     "relation_labels": relation_ids,  # Shape: [batch_size, max_pairs]
        ... }
        >>> output = model(**batch)
        >>> # output contains:
        >>> # - entity_logits: [batch_size, seq_len, num_entity_types]
        >>> # - relation_logits: [batch_size, num_entity_pairs, num_relation_types]
        >>> # - loss (if labels provided): scalar tensor
        >>> # - y_prob: predicted probabilities
    """

    def __init__(
        self,
        dataset: SampleDataset,
        embedding_dim: int = 768,
        num_entity_types: int = 25,
        num_relation_types: int = 4,
        pretrained_model: str = "emilyalsentzer/Bio_ClinicalBERT",
        freeze_encoder: bool = True,
        dropout_rate: float = 0.1,
        use_crf: bool = False,
        **kwargs,
    ):
        """Initialize RexKG model.

        Args:
            dataset: PyHealth dataset with text inputs
            embedding_dim: Dimension of encoder embeddings
            num_entity_types: Number of entity type classes
            num_relation_types: Number of relation type classes
            pretrained_model: HuggingFace model identifier
            freeze_encoder: Whether to freeze encoder weights
            dropout_rate: Dropout probability
            use_crf: Whether to use CRF for sequence tagging
            **kwargs: Additional arguments (unused, for compatibility)
        """
        super(RexKG, self).__init__(dataset=dataset)
        
        self.embedding_dim = embedding_dim
        self.num_entity_types = num_entity_types
        self.num_relation_types = num_relation_types
        self.use_crf = use_crf

        # Load pretrained encoder
        try:
            self.encoder = AutoModel.from_pretrained(pretrained_model)
            logger.info(f"Loaded pretrained model: {pretrained_model}")
        except Exception as e:
            logger.warning(
                f"Failed to load {pretrained_model}, using BERT-base-uncased: {e}"
            )
            self.encoder = AutoModel.from_pretrained("bert-base-uncased")

        # Freeze encoder if specified
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        # Entity extraction head (NER)
        self.entity_head = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(embedding_dim // 2, num_entity_types),
        )

        # Relation extraction head (RE)
        # Takes concatenated pair embeddings as input
        self.relation_head = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(embedding_dim, num_relation_types),
        )

        self.dropout = nn.Dropout(dropout_rate)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        """Forward pass for entity and relation extraction.

        Args:
            **kwargs: Dictionary containing:
                - text: Input text tensor or (input_ids, attention_mask) tuple
                - entity_labels (optional): Ground truth entity labels
                - relation_labels (optional): Ground truth relation labels

        Returns:
            Dictionary with keys:
                - entity_logits: Entity prediction logits [batch, seq_len, num_entity_types]
                - relation_logits: Relation prediction logits [batch, num_pairs, num_relation_types]
                - logit: Concatenated logits for compatibility
                - loss (optional): Total loss if labels provided
                - y_prob (optional): Softmax probabilities
                - y_true (optional): Ground truth labels if provided
        """
        # Extract input
        text_input = kwargs.get("text")
        entity_labels = kwargs.get("entity_labels")
        relation_labels = kwargs.get("relation_labels")

        # Handle different input formats
        if isinstance(text_input, (tuple, list)):
            input_ids, attention_mask = text_input[0], text_input[1]
        else:
            input_ids = text_input
            attention_mask = None

        # Encoder forward pass
        encoder_output = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        hidden_states = encoder_output.last_hidden_state  # [batch, seq_len, embedding_dim]

        # Entity extraction
        entity_logits = self.entity_head(self.dropout(hidden_states))

        # Relation extraction (simplified: use [CLS] + first/last representations)
        batch_size = hidden_states.shape[0]
        # Simplified relation extraction using concatenated representations
        relation_logits = self._extract_relations(hidden_states, entity_logits)

        output = {
            "entity_logits": entity_logits,
            "relation_logits": relation_logits,
            "logit": entity_logits,  # For compatibility with BaseModel
        }

        # Calculate loss if labels provided
        if entity_labels is not None:
            entity_loss = self.loss_fn(
                entity_logits.view(-1, self.num_entity_types),
                entity_labels.view(-1),
            )
            losses = [entity_loss]

            if relation_labels is not None:
                relation_loss = self.loss_fn(
                    relation_logits.view(-1, self.num_relation_types),
                    relation_labels.view(-1),
                )
                losses.append(relation_loss)

            output["loss"] = sum(losses) / len(losses)
            output["y_true"] = entity_labels

        # Add probabilities
        output["y_prob"] = torch.softmax(entity_logits, dim=-1)

        return output

    def _extract_relations(
        self,
        hidden_states: torch.Tensor,
        entity_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Extract relations from entity representations.

        Simplified relation extraction using entity representations.
        In practice, this would extract entity pairs and compute relations.

        Args:
            hidden_states: Encoder hidden states [batch, seq_len, embedding_dim]
            entity_logits: Entity predictions [batch, seq_len, num_entity_types]

        Returns:
            Relation logits [batch, max_pairs, num_relation_types]
        """
        batch_size = hidden_states.shape[0]
        seq_len = hidden_states.shape[1]

        # Simplified: take top entity pairs (would be more sophisticated in practice)
        # For now, create dummy relation logits
        max_pairs = min(seq_len * (seq_len - 1) // 2, 32)  # Cap at reasonable number
        
        # Create pair representations from entity positions
        # This is a placeholder - real implementation would select entity pairs
        relation_reps = torch.zeros(
            batch_size, max_pairs, self.embedding_dim * 2,
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        
        # Fill with some representations (in practice, would select specific pairs)
        for b in range(batch_size):
            for p in range(min(max_pairs, seq_len - 1)):
                i, j = p % seq_len, (p + 1) % seq_len
                relation_reps[b, p] = torch.cat(
                    [hidden_states[b, i], hidden_states[b, j]], dim=-1
                )

        relation_logits = self.relation_head(relation_reps)
        return relation_logits

    def forward_from_embedding(self, **kwargs) -> Dict[str, torch.Tensor]:
        """Forward pass from embeddings (for interpretability).

        This method allows gradient-based interpretability methods to work
        with the model by accepting pre-computed embeddings as input.

        Args:
            **kwargs: Dictionary with embedding tensors

        Returns:
            Model output dictionary
        """
        # For interpretability, forward directly through heads
        embeddings = kwargs.get("embeddings")
        if embeddings is None:
            return self.forward(**kwargs)

        entity_logits = self.entity_head(self.dropout(embeddings))
        return {
            "entity_logits": entity_logits,
            "logit": entity_logits,
            "y_prob": torch.softmax(entity_logits, dim=-1),
        }
