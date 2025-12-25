"""
Configuration management for RAG-Anything framework.
Uses Pydantic for validation and environment variable support.
"""

import os
from pathlib import Path
from typing import Dict, Optional, Literal
from enum import Enum

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings
import yaml


class ModalityType(str, Enum):
    """Supported content modalities."""
    TEXT = "text"
    IMAGE = "image"
    TABLE = "table"
    EQUATION = "equation"


class ModelProvider(str, Enum):
    """Supported model providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    LOCAL = "local"


class VLMConfig(BaseModel):
    """Vision-Language Model configuration."""
    provider: ModelProvider = ModelProvider.OPENAI
    model: str = "gpt-4o-mini"
    api_key: Optional[str] = Field(default=None, description="API key from env")
    max_tokens: int = Field(default=1000, ge=1)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    
    @field_validator('api_key', mode='before')
    @classmethod
    def get_api_key(cls, v, info):
        """Get API key from environment if not provided."""
        if v is None:
            provider = info.data.get('provider')
            if provider == ModelProvider.OPENAI:
                return os.getenv('OPENAI_API_KEY')
            elif provider == ModelProvider.ANTHROPIC:
                return os.getenv('ANTHROPIC_API_KEY')
        return v


class LLMConfig(BaseModel):
    """Large Language Model configuration."""
    provider: ModelProvider = ModelProvider.OPENAI
    model: str = "gpt-4o-mini"
    api_key: Optional[str] = None
    max_tokens: int = Field(default=2000, ge=1)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    
    @field_validator('api_key', mode='before')
    @classmethod
    def get_api_key(cls, v, info):
        if v is None:
            provider = info.data.get('provider')
            if provider == ModelProvider.OPENAI:
                return os.getenv('OPENAI_API_KEY')
            elif provider == ModelProvider.ANTHROPIC:
                return os.getenv('ANTHROPIC_API_KEY')
        return v


class EmbeddingConfig(BaseModel):
    """Embedding model configuration."""
    provider: ModelProvider = ModelProvider.OPENAI
    model: str = "text-embedding-3-large"
    api_key: Optional[str] = None
    dimensions: int = Field(default=3072, ge=1)
    batch_size: int = Field(default=32, ge=1)
    
    @field_validator('api_key', mode='before')
    @classmethod
    def get_api_key(cls, v, info):
        if v is None:
            provider = info.data.get('provider')
            if provider == ModelProvider.OPENAI:
                return os.getenv('OPENAI_API_KEY')
        return v


class GraphConfig(BaseModel):
    """Knowledge graph construction configuration."""
    context_window_size: int = Field(
        default=3,
        ge=0,
        description="δ parameter: neighborhood size for context-aware processing"
    )
    max_hops: int = Field(
        default=2,
        ge=1,
        description="Maximum hops for graph navigation"
    )
    entity_token_limit: int = Field(
        default=20000,
        ge=1000,
        description="Combined entity-relation token limit"
    )
    chunk_token_limit: int = Field(
        default=12000,
        ge=1000,
        description="Chunk token limit"
    )
    min_entity_confidence: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Minimum confidence for entity extraction"
    )
    enable_entity_linking: bool = Field(
        default=True,
        description="Enable cross-document entity linking"
    )


class RetrievalConfig(BaseModel):
    """Retrieval configuration."""
    top_k_semantic: int = Field(
        default=10,
        ge=1,
        description="Top-K for semantic similarity search"
    )
    top_k_structural: int = Field(
        default=5,
        ge=1,
        description="Top-K for structural navigation"
    )
    fusion_weights: Dict[str, float] = Field(
        default={
            "structural": 0.4,
            "semantic": 0.4,
            "modality": 0.2
        },
        description="Weights for multi-signal fusion"
    )
    enable_reranking: bool = Field(
        default=True,
        description="Enable cross-modal reranking"
    )
    reranker_model: Optional[str] = Field(
        default="bge-reranker-v2-m3",
        description="Reranker model name"
    )
    
    @field_validator('fusion_weights')
    @classmethod
    def validate_weights(cls, v):
        """Ensure fusion weights sum to 1.0."""
        total = sum(v.values())
        if not 0.99 <= total <= 1.01:  # Allow small floating point errors
            raise ValueError(f"Fusion weights must sum to 1.0, got {total}")
        return v


class ParserConfig(BaseModel):
    """Document parser configuration."""
    image_resolution: int = Field(
        default=144,
        ge=72,
        description="DPI for image rendering"
    )
    image_max_size: int = Field(
        default=2048,
        ge=512,
        description="Maximum image dimension"
    )
    table_engine: Literal["camelot", "tabula", "pdfplumber"] = Field(
        default="camelot",
        description="Table extraction engine"
    )
    equation_engine: Literal["pix2tex", "mathpix"] = Field(
        default="pix2tex",
        description="Equation OCR engine"
    )
    max_pages_per_doc: int = Field(
        default=50,
        ge=1,
        description="Maximum pages to process per document"
    )
    enable_ocr: bool = Field(
        default=True,
        description="Enable OCR for scanned documents"
    )


class VectorStoreConfig(BaseModel):
    """Vector store configuration."""
    backend: Literal["faiss", "chromadb", "milvus"] = Field(
        default="faiss",
        description="Vector store backend"
    )
    index_type: Literal["flat", "ivf", "hnsw"] = Field(
        default="flat",
        description="Index type for FAISS"
    )
    metric: Literal["cosine", "l2", "ip"] = Field(
        default="cosine",
        description="Distance metric"
    )
    persist_dir: Optional[Path] = Field(
        default=None,
        description="Directory to persist vector store"
    )


class RAGConfig(BaseSettings):
    """Main RAG-Anything configuration."""
    
    # Model configurations
    vlm: VLMConfig = Field(default_factory=VLMConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    
    # Component configurations
    graph: GraphConfig = Field(default_factory=GraphConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    parser: ParserConfig = Field(default_factory=ParserConfig)
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    
    # General settings
    cache_dir: Path = Field(
        default=Path.home() / ".cache" / "rag_anything",
        description="Cache directory for models and data"
    )
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_file: Optional[Path] = None
    num_workers: int = Field(
        default=4,
        ge=1,
        description="Number of parallel workers"
    )
    
    class Config:
        env_prefix = "RAG_"
        env_nested_delimiter = "__"
    
    def model_post_init(self, __context):
        """Create cache directory if it doesn't exist."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def from_yaml(cls, yaml_path: Path) -> "RAGConfig":
        """
        Load configuration from YAML file.
        
        Args:
            yaml_path: Path to YAML configuration file
        
        Returns:
            RAGConfig instance
        """
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)
    
    def to_yaml(self, yaml_path: Path) -> None:
        """
        Save configuration to YAML file.
        
        Args:
            yaml_path: Path to save YAML configuration
        """
        config_dict = self.model_dump(exclude={'vlm': {'api_key'}, 'llm': {'api_key'}, 'embedding': {'api_key'}})
        with open(yaml_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)


# Default configuration instance
default_config = RAGConfig()


if __name__ == "__main__":
    # Example usage
    config = RAGConfig()
    print(config.model_dump_json(indent=2))
    
    # Save to YAML
    config.to_yaml(Path("config.yaml"))
    
    # Load from YAML
    loaded_config = RAGConfig.from_yaml(Path("config.yaml"))
    print(loaded_config)