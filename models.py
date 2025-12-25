"""
Core data structures for RAG-Anything framework.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set, Tuple
from enum import Enum
from pathlib import Path
import numpy as np


class ModalityType(str, Enum):
    """Content modality types."""
    TEXT = "text"
    IMAGE = "image"
    TABLE = "table"
    EQUATION = "equation"


class EntityType(str, Enum):
    """Entity types for knowledge graph."""
    CONCEPT = "concept"
    OBJECT = "object"
    PERSON = "person"
    ORGANIZATION = "organization"
    METRIC = "metric"
    VALUE = "value"
    MULTIMODAL_ANCHOR = "multimodal_anchor"
    OTHER = "other"


class RelationType(str, Enum):
    """Relationship types for knowledge graph."""
    BELONGS_TO = "belongs_to"
    DESCRIBES = "describes"
    RELATES_TO = "relates_to"
    CAUSES = "causes"
    COMPARES = "compares"
    CONTAINS = "contains"
    PART_OF = "part_of"
    REFERENCES = "references"
    OTHER = "other"


@dataclass
class ContentUnit:
    """
    Atomic knowledge unit after document decomposition.
    Corresponds to c_j = (t_j, x_j) in the paper.
    """
    id: str
    modality_type: ModalityType
    content: Any  # Raw content (text, image path, table dict, equation latex)
    metadata: Dict[str, Any] = field(default_factory=dict)
    position: int = 0  # Position in document
    page_num: Optional[int] = None
    document_id: Optional[str] = None
    
    # Context information
    context_before: Optional[str] = None
    context_after: Optional[str] = None
    caption: Optional[str] = None
    
    # Generated descriptions (for non-text modalities)
    description: Optional[str] = None  # d_chunk_j in paper
    entity_summary: Optional[Dict[str, Any]] = None  # e_entity_j in paper
    
    def __hash__(self):
        return hash(self.id)
    
    def __eq__(self, other):
        if not isinstance(other, ContentUnit):
            return False
        return self.id == other.id
    
    def get_context_window(self, all_units: List["ContentUnit"], delta: int = 3) -> List["ContentUnit"]:
        """
        Get neighborhood context window C_j = {c_k | |k - j| <= δ}.
        
        Args:
            all_units: All content units in document
            delta: Context window size
        
        Returns:
            List of content units in context window
        """
        # Find current position in list
        try:
            idx = all_units.index(self)
        except ValueError:
            return [self]
        
        start = max(0, idx - delta)
        end = min(len(all_units), idx + delta + 1)
        return all_units[start:end]


@dataclass
class Entity:
    """Knowledge graph entity node."""
    id: str
    name: str
    type: EntityType
    attributes: Dict[str, Any] = field(default_factory=dict)
    source_unit_id: Optional[str] = None  # Link back to content unit
    embedding: Optional[np.ndarray] = None
    
    def __hash__(self):
        return hash(self.id)
    
    def __eq__(self, other):
        if not isinstance(other, Entity):
            return False
        return self.id == other.id


@dataclass
class Relation:
    """Knowledge graph relationship edge."""
    id: str
    source: Entity
    target: Entity
    type: RelationType
    attributes: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    embedding: Optional[np.ndarray] = None
    
    def __hash__(self):
        return hash(self.id)
    
    def __eq__(self, other):
        if not isinstance(other, Relation):
            return False
        return self.id == other.id


@dataclass
class KnowledgeGraph:
    """
    Knowledge graph structure.
    Corresponds to G = (V, E) in the paper.
    """
    entities: Set[Entity] = field(default_factory=set)
    relations: Set[Relation] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_entity(self, entity: Entity) -> None:
        """Add entity to graph."""
        self.entities.add(entity)
    
    def add_relation(self, relation: Relation) -> None:
        """Add relation to graph."""
        self.relations.add(relation)
        # Ensure entities are in graph
        self.entities.add(relation.source)
        self.entities.add(relation.target)
    
    def get_entity_by_name(self, name: str) -> Optional[Entity]:
        """Find entity by name."""
        for entity in self.entities:
            if entity.name.lower() == name.lower():
                return entity
        return None
    
    def get_neighbors(self, entity: Entity, max_hops: int = 1) -> Set[Entity]:
        """
        Get neighboring entities within max_hops.
        
        Args:
            entity: Source entity
            max_hops: Maximum number of hops
        
        Returns:
            Set of neighboring entities
        """
        if max_hops == 0:
            return {entity}
        
        neighbors = {entity}
        current_level = {entity}
        
        for _ in range(max_hops):
            next_level = set()
            for e in current_level:
                # Find all relations involving this entity
                for rel in self.relations:
                    if rel.source == e:
                        next_level.add(rel.target)
                    elif rel.target == e:
                        next_level.add(rel.source)
            neighbors.update(next_level)
            current_level = next_level
        
        return neighbors
    
    def merge(self, other: "KnowledgeGraph") -> None:
        """Merge another knowledge graph into this one."""
        self.entities.update(other.entities)
        self.relations.update(other.relations)


@dataclass
class EmbeddingTable:
    """
    Dense representation table for all components.
    Corresponds to T in the paper.
    """
    embeddings: Dict[str, np.ndarray] = field(default_factory=dict)
    metadata: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    dimension: int = 3072
    
    def add(self, component_id: str, embedding: np.ndarray, metadata: Dict[str, Any] = None) -> None:
        """Add embedding for a component."""
        self.embeddings[component_id] = embedding
        if metadata:
            self.metadata[component_id] = metadata
    
    def get(self, component_id: str) -> Optional[np.ndarray]:
        """Get embedding for a component."""
        return self.embeddings.get(component_id)
    
    def search(self, query_embedding: np.ndarray, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Search for similar embeddings.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
        
        Returns:
            List of (component_id, similarity_score) tuples
        """
        if not self.embeddings:
            return []
        
        # Compute cosine similarities
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        similarities = []
        
        for comp_id, emb in self.embeddings.items():
            emb_norm = emb / np.linalg.norm(emb)
            similarity = np.dot(query_norm, emb_norm)
            similarities.append((comp_id, float(similarity)))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]


@dataclass
class Index:
    """
    Complete retrieval index.
    Corresponds to I = (G, T) in the paper.
    """
    graph: KnowledgeGraph
    embedding_table: EmbeddingTable
    content_units: Dict[str, ContentUnit] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def save(self, path: Path) -> None:
        """
        Save index to disk.
        
        Args:
            path: Directory to save index
        """
        # TODO: Implement serialization
        # - Save graph structure (networkx)
        # - Save embeddings (numpy)
        # - Save content units (pickle)
        pass
    
    @classmethod
    def load(cls, path: Path) -> "Index":
        """
        Load index from disk.
        
        Args:
            path: Directory containing saved index
        
        Returns:
            Index instance
        """
        # TODO: Implement deserialization
        pass


@dataclass
class RetrievalResult:
    """Result from retrieval process."""
    content_unit: ContentUnit
    score: float
    source: str  # "structural" or "semantic"
    explanation: Optional[str] = None
    
    def __lt__(self, other):
        """For sorting by score."""
        return self.score < other.score


@dataclass
class QueryResult:
    """Final query result with answer."""
    query: str
    answer: str
    retrieved_units: List[RetrievalResult]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_sources(self) -> List[str]:
        """Get list of source content unit IDs."""
        return [result.content_unit.id for result in self.retrieved_units]