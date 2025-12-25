"""
Text parser for extracting hierarchical text content.
Handles paragraphs, headings, lists, and sections.
"""

import re
from pathlib import Path
from typing import List, Any, Dict
import traceback

from loguru import logger

from .base_parser import BaseParser
from ..config.settings import ParserConfig
from ..utils.exceptions import TextParsingError
from ..utils.logger import log_exception
from . import data_models


class TextParser(BaseParser):
    """
    Parser for text content extraction.
    Segments text into paragraphs and preserves hierarchical structure.
    """
    
    def __init__(self, config: ParserConfig):
        super().__init__(config)
        self.min_paragraph_length = 10  # Minimum chars for a paragraph
        self.logger = logger.bind(parser="TextParser")
    
    def _load_document(self, file_path: Path) -> str:
        """
        Load text document.
        
        Args:
            file_path: Path to text file
        
        Returns:
            Text content as string
        """
        try:
            # Try multiple encodings
            encodings = ['utf-8', 'latin-1', 'cp1252']
            
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        return f.read()
                except UnicodeDecodeError:
                    continue
            
            raise TextParsingError(
                f"Could not decode file with any encoding: {encodings}",
                details={"file": str(file_path)}
            )
            
        except Exception as e:
            log_exception(e, context="TextParser._load_document")
            raise TextParsingError(
                f"Failed to load text document",
                details={"file": str(file_path), "error": str(e)}
            ) from e
    
    def _extract_content_units(self, doc: str, document_id: str) -> List[data_models.ContentUnit]:
        """
        Extract text content units (paragraphs).
        
        Args:
            doc: Text content
            document_id: Document identifier
        
        Returns:
            List of ContentUnit objects
        """
        content_units = []
        
        # Split into paragraphs
        paragraphs = self._segment_paragraphs(doc)
        
        for i, paragraph in enumerate(paragraphs):
            if len(paragraph.strip()) < self.min_paragraph_length:
                continue
            
            # Create content unit
            unit = data_models.ContentUnit(
                id=f"{document_id}_text_{i}",
                modality_type=data_models.ModalityType.TEXT,
                content=paragraph.strip(),
                document_id=document_id,
                metadata={
                    "paragraph_index": i,
                    "char_count": len(paragraph),
                    "word_count": len(paragraph.split())
                }
            )
            
            # Detect if this is a heading
            if self._is_heading(paragraph):
                unit.metadata["is_heading"] = True
                unit.metadata["heading_level"] = self._get_heading_level(paragraph)
            
            content_units.append(unit)
        
        self.logger.info(f"Extracted {len(content_units)} text paragraphs")
        return content_units
    
    def _segment_paragraphs(self, text: str) -> List[str]:
        """
        Segment text into paragraphs.
        
        Args:
            text: Full text content
        
        Returns:
            List of paragraph strings
        """
        # Split on double newlines or single newlines followed by indentation
        # This handles various paragraph formats
        
        # First, normalize newlines
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        
        # Split on double newlines
        paragraphs = re.split(r'\n\s*\n', text)
        
        # Clean up each paragraph
        cleaned_paragraphs = []
        for para in paragraphs:
            # Remove extra whitespace
            para = ' '.join(para.split())
            if para:
                cleaned_paragraphs.append(para)
        
        return cleaned_paragraphs
    
    def _is_heading(self, text: str) -> bool:
        """
        Detect if text is a heading.
        
        Args:
            text: Text to check
        
        Returns:
            True if heading, False otherwise
        """
        text = text.strip()
        
        # Heuristics for heading detection:
        # 1. Shorter than 200 characters
        # 2. No period at the end (unless it's an abbreviation)
        # 3. First letter is capitalized
        # 4. Contains typical heading patterns (numbers, "Chapter", etc.)
        
        if len(text) > 200:
            return False
        
        # Check for numbering patterns (1., 1.1, Chapter 1, etc.)
        if re.match(r'^(\d+\.)+\s', text) or re.match(r'^(Chapter|Section|Part)\s+\d+', text, re.IGNORECASE):
            return True
        
        # Check if all caps (common in headings)
        if text.isupper() and len(text.split()) <= 10:
            return True
        
        # Check if no sentence-ending punctuation
        if not text.endswith(('.', '!', '?', ';', ':')):
            if len(text.split()) <= 15:  # Short and no punctuation
                return True
        
        return False
    
    def _get_heading_level(self, text: str) -> int:
        """
        Estimate heading level (1-6).
        
        Args:
            text: Heading text
        
        Returns:
            Heading level (1 = highest)
        """
        text = text.strip()
        
        # Check for numbered headings (1., 1.1., 1.1.1., etc.)
        match = re.match(r'^((\d+\.)+)', text)
        if match:
            dots = match.group(1).count('.')
            return min(dots, 6)
        
        # Check for chapter/part indicators
        if re.match(r'^(Chapter|Part)', text, re.IGNORECASE):
            return 1
        if re.match(r'^Section', text, re.IGNORECASE):
            return 2
        
        # Check formatting clues
        if text.isupper():
            return 1
        
        # Default to level 2 for other headings
        return 2
    
    def _extract_metadata(self, unit: data_models.ContentUnit, doc: str) -> Dict[str, Any]:
        """
        Extract metadata for text unit.
        
        Args:
            unit: Content unit
            doc: Full document text
        
        Returns:
            Metadata dictionary
        """
        metadata = super()._extract_metadata(unit, doc)
        
        # Add text-specific metadata
        text = str(unit.content)
        
        metadata.update({
            "length": len(text),
            "sentences": text.count('.') + text.count('!') + text.count('?'),
            "has_citations": bool(re.search(r'\[\d+\]|\(\w+\s+\d{4}\)', text)),
            "has_urls": bool(re.search(r'https?://', text)),
        })
        
        return metadata
    
    def get_supported_extensions(self) -> List[str]:
        """Get supported file extensions."""
        return ['.txt', '.md', '.rst']


# Example usage
if __name__ == "__main__":
    from ..config.settings import RAGConfig
    from ..utils.logger import setup_logger
    
    setup_logger(level="DEBUG")
    
    config = RAGConfig()
    parser = TextParser(config.parser)
    
    # Test with sample text
    sample_file = Path("sample.txt")
    if sample_file.exists():
        units = parser.parse(sample_file)
        print(f"Extracted {len(units)} content units")
        for unit in units[:3]:
            print(f"\nUnit ID: {unit.id}")
            print(f"Content: {unit.content[:100]}...")
            print(f"Metadata: {unit.metadata}")