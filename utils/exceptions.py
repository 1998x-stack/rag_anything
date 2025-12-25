"""
Custom exception hierarchy for RAG-Anything framework.
All exceptions inherit from RAGAnythingException for consistent error handling.
"""


class RAGAnythingException(Exception):
    """Base exception for all RAG-Anything errors."""
    
    def __init__(self, message: str, details: dict = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)
    
    def __str__(self):
        if self.details:
            details_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            return f"{self.message} ({details_str})"
        return self.message


class ParsingError(RAGAnythingException):
    """Raised when document parsing fails."""
    pass


class TextParsingError(ParsingError):
    """Raised when text extraction fails."""
    pass


class ImageParsingError(ParsingError):
    """Raised when image extraction fails."""
    pass


class TableParsingError(ParsingError):
    """Raised when table parsing fails."""
    pass


class EquationParsingError(ParsingError):
    """Raised when equation recognition fails."""
    pass


class GraphBuildError(RAGAnythingException):
    """Raised when knowledge graph construction fails."""
    pass


class EntityExtractionError(GraphBuildError):
    """Raised when entity extraction fails."""
    pass


class RelationExtractionError(GraphBuildError):
    """Raised when relation extraction fails."""
    pass


class GraphFusionError(GraphBuildError):
    """Raised when graph fusion/alignment fails."""
    pass


class EmbeddingError(RAGAnythingException):
    """Raised when embedding generation fails."""
    pass


class EncoderError(EmbeddingError):
    """Raised when encoder fails."""
    pass


class VectorStoreError(EmbeddingError):
    """Raised when vector store operations fail."""
    pass


class RetrievalError(RAGAnythingException):
    """Raised when retrieval fails."""
    pass


class StructuralNavigationError(RetrievalError):
    """Raised when graph navigation fails."""
    pass


class SemanticMatchingError(RetrievalError):
    """Raised when semantic search fails."""
    pass


class RankingError(RetrievalError):
    """Raised when result ranking fails."""
    pass


class SynthesisError(RAGAnythingException):
    """Raised when answer generation fails."""
    pass


class VLMError(RAGAnythingException):
    """Raised when Vision-Language Model API fails."""
    pass


class LLMError(RAGAnythingException):
    """Raised when Large Language Model API fails."""
    pass


class ConfigurationError(RAGAnythingException):
    """Raised when configuration is invalid."""
    pass


class IndexingError(RAGAnythingException):
    """Raised when document indexing fails."""
    pass


class QueryError(RAGAnythingException):
    """Raised when query processing fails."""
    pass


class ValidationError(RAGAnythingException):
    """Raised when data validation fails."""
    pass


# Exception mapping for easier error handling
EXCEPTION_MAP = {
    "parsing": ParsingError,
    "text_parsing": TextParsingError,
    "image_parsing": ImageParsingError,
    "table_parsing": TableParsingError,
    "equation_parsing": EquationParsingError,
    "graph_build": GraphBuildError,
    "entity_extraction": EntityExtractionError,
    "relation_extraction": RelationExtractionError,
    "graph_fusion": GraphFusionError,
    "embedding": EmbeddingError,
    "encoder": EncoderError,
    "vector_store": VectorStoreError,
    "retrieval": RetrievalError,
    "structural_navigation": StructuralNavigationError,
    "semantic_matching": SemanticMatchingError,
    "ranking": RankingError,
    "synthesis": SynthesisError,
    "vlm": VLMError,
    "llm": LLMError,
    "configuration": ConfigurationError,
    "indexing": IndexingError,
    "query": QueryError,
    "validation": ValidationError,
}


def get_exception_class(error_type: str) -> type:
    """
    Get exception class by error type string.
    
    Args:
        error_type: Error type identifier
    
    Returns:
        Exception class
    """
    return EXCEPTION_MAP.get(error_type, RAGAnythingException)