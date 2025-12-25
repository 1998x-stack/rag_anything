"""
Abstract base parser interface for document parsing.
All modality-specific parsers inherit from this class.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Any, Optional
import traceback

from loguru import logger

from ..config.settings import ParserConfig
from ..utils.exceptions import ParsingError
from ..utils.logger import log_exception
from . import data_models


class BaseParser(ABC):
    """
    Abstract base class for document parsers.
    Implements Template Method pattern for consistent parsing flow.
    """
    
    def __init__(self, config: ParserConfig):
        """
        Initialize parser with configuration.
        
        Args:
            config: Parser configuration
        """
        self.config = config
        self.logger = logger.bind(parser=self.__class__.__name__)
    
    def parse(self, file_path: Path, document_id: str = None) -> List[data_models.ContentUnit]:
        """
        Parse document and extract content units.
        Template method that defines the parsing workflow.
        
        Args:
            file_path: Path to document file
            document_id: Optional document identifier
        
        Returns:
            List of ContentUnit objects
        
        Raises:
            ParsingError: If parsing fails
        """
        self.logger.info(f"Starting to parse: {file_path}")
        
        try:
            # Step 1: Validate input
            self._validate_input(file_path)
            
            # Step 2: Load document
            doc = self._load_document(file_path)
            
            # Step 3: Extract content units (modality-specific)
            content_units = self._extract_content_units(doc, document_id or str(file_path))
            
            # Step 4: Extract metadata for each unit
            for unit in content_units:
                unit.metadata.update(self._extract_metadata(unit, doc))
            
            # Step 5: Post-process units
            content_units = self._post_process(content_units)
            
            self.logger.success(f"Parsed {len(content_units)} content units from {file_path}")
            return content_units
            
        except Exception as e:
            log_exception(e, context=f"{self.__class__.__name__}.parse")
            raise ParsingError(
                f"Failed to parse {file_path}",
                details={"file": str(file_path), "error": str(e)}
            ) from e
    
    def _validate_input(self, file_path: Path) -> None:
        """
        Validate input file.
        
        Args:
            file_path: Path to validate
        
        Raises:
            ParsingError: If validation fails
        """
        if not file_path.exists():
            raise ParsingError(
                f"File does not exist: {file_path}",
                details={"file": str(file_path)}
            )
        
        if not file_path.is_file():
            raise ParsingError(
                f"Path is not a file: {file_path}",
                details={"file": str(file_path)}
            )
    
    @abstractmethod
    def _load_document(self, file_path: Path) -> Any:
        """
        Load document from file.
        Must be implemented by subclasses.
        
        Args:
            file_path: Path to document
        
        Returns:
            Document object (format depends on parser type)
        """
        pass
    
    @abstractmethod
    def _extract_content_units(self, doc: Any, document_id: str) -> List[data_models.ContentUnit]:
        """
        Extract content units from document.
        Core parsing logic - must be implemented by subclasses.
        
        Args:
            doc: Loaded document object
            document_id: Document identifier
        
        Returns:
            List of ContentUnit objects
        """
        pass
    
    def _extract_metadata(self, unit: data_models.ContentUnit, doc: Any) -> Dict[str, Any]:
        """
        Extract metadata for a content unit.
        Can be overridden by subclasses for specific metadata.
        
        Args:
            unit: Content unit
            doc: Document object
        
        Returns:
            Metadata dictionary
        """
        metadata = {
            "parser": self.__class__.__name__,
            "modality": unit.modality_type.value
        }
        return metadata
    
    def _post_process(self, content_units: List[data_models.ContentUnit]) -> List[data_models.ContentUnit]:
        """
        Post-process extracted content units.
        Can be overridden by subclasses for specific processing.
        
        Args:
            content_units: List of content units
        
        Returns:
            Processed content units
        """
        # Add position information
        for i, unit in enumerate(content_units):
            unit.position = i
        
        # Add context information (before/after)
        for i, unit in enumerate(content_units):
            if i > 0:
                prev_unit = content_units[i - 1]
                if prev_unit.modality_type == data_models.ModalityType.TEXT:
                    unit.context_before = str(prev_unit.content)[:500]  # First 500 chars
            
            if i < len(content_units) - 1:
                next_unit = content_units[i + 1]
                if next_unit.modality_type == data_models.ModalityType.TEXT:
                    unit.context_after = str(next_unit.content)[:500]
        
        return content_units
    
    def get_supported_extensions(self) -> List[str]:
        """
        Get list of supported file extensions.
        Can be overridden by subclasses.
        
        Returns:
            List of extensions (e.g., ['.pdf', '.docx'])
        """
        return []
    
    def __repr__(self):
        return f"{self.__class__.__name__}(config={self.config})"