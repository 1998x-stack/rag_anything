"""
Vision-Language Model interface for multimodal analysis.
Supports OpenAI GPT-4V and Anthropic Claude.
"""

import json
import base64
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from io import BytesIO
import traceback

from PIL import Image
from loguru import logger

from ..config.settings import VLMConfig, ModelProvider
from ..config.prompts import get_prompt
from ..utils.exceptions import VLMError
from ..utils.logger import log_exception


class VLMInterface:
    """
    Unified interface for Vision-Language Models.
    Abstracts differences between providers (OpenAI, Anthropic).
    """
    
    def __init__(self, config: VLMConfig):
        """
        Initialize VLM interface.
        
        Args:
            config: VLM configuration
        """
        self.config = config
        self.provider = config.provider
        self.logger = logger.bind(module="VLMInterface")
        
        # Initialize provider-specific client
        if self.provider == ModelProvider.OPENAI:
            try:
                import openai
                self.client = openai.OpenAI(api_key=config.api_key)
            except ImportError:
                raise VLMError("OpenAI package not installed. Run: pip install openai")
        elif self.provider == ModelProvider.ANTHROPIC:
            try:
                import anthropic
                self.client = anthropic.Anthropic(api_key=config.api_key)
            except ImportError:
                raise VLMError("Anthropic package not installed. Run: pip install anthropic")
        else:
            raise VLMError(f"Unsupported VLM provider: {self.provider}")
        
        self.logger.info(f"VLM initialized with provider={self.provider}, model={config.model}")
    
    def generate(
        self,
        prompt: str,
        images: Optional[List[Union[str, Path, Image.Image]]] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        json_mode: bool = True
    ) -> str:
        """
        Generate response from VLM.
        
        Args:
            prompt: Text prompt
            images: Optional list of images (paths or PIL Images)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            json_mode: Force JSON output format
        
        Returns:
            Generated text response
        
        Raises:
            VLMError: If generation fails
        """
        try:
            max_tokens = max_tokens or self.config.max_tokens
            temperature = temperature if temperature is not None else self.config.temperature
            
            # Process images
            image_data = []
            if images:
                image_data = [self._process_image(img) for img in images]
            
            # Generate based on provider
            if self.provider == ModelProvider.OPENAI:
                response = self._generate_openai(prompt, image_data, max_tokens, temperature, json_mode)
            elif self.provider == ModelProvider.ANTHROPIC:
                response = self._generate_anthropic(prompt, image_data, max_tokens, temperature, json_mode)
            else:
                raise VLMError(f"Provider not implemented: {self.provider}")
            
            return response
            
        except Exception as e:
            log_exception(e, context="VLMInterface.generate")
            raise VLMError(
                "Failed to generate VLM response",
                details={"error": str(e), "provider": self.provider}
            ) from e
    
    def _generate_openai(
        self,
        prompt: str,
        images: List[Dict],
        max_tokens: int,
        temperature: float,
        json_mode: bool
    ) -> str:
        """
        Generate response using OpenAI API.
        
        Returns:
            Generated text
        """
        messages = [{"role": "user", "content": []}]
        
        # Add text
        messages[0]["content"].append({"type": "text", "text": prompt})
        
        # Add images
        for img in images:
            messages[0]["content"].append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img['base64']}"}
            })
        
        # API call
        kwargs = {
            "model": self.config.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        
        response = self.client.chat.completions.create(**kwargs)
        return response.choices[0].message.content
    
    def _generate_anthropic(
        self,
        prompt: str,
        images: List[Dict],
        max_tokens: int,
        temperature: float,
        json_mode: bool
    ) -> str:
        """
        Generate response using Anthropic API.
        
        Returns:
            Generated text
        """
        content = []
        
        # Add images first (Anthropic convention)
        for img in images:
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": img["media_type"],
                    "data": img["base64"]
                }
            })
        
        # Add text
        content.append({"type": "text", "text": prompt})
        
        # API call
        response = self.client.messages.create(
            model=self.config.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": content}]
        )
        
        return response.content[0].text
    
    def _process_image(self, image: Union[str, Path, Image.Image]) -> Dict[str, str]:
        """
        Process image to base64 format.
        
        Args:
            image: Image path or PIL Image
        
        Returns:
            Dictionary with base64 data and media type
        """
        try:
            # Load image if path
            if isinstance(image, (str, Path)):
                img = Image.open(image)
            else:
                img = image
            
            # Convert to RGB if necessary
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Encode to base64
            buffered = BytesIO()
            img.save(buffered, format="JPEG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode()
            
            return {
                "base64": img_base64,
                "media_type": "image/jpeg"
            }
            
        except Exception as e:
            log_exception(e, context="VLMInterface._process_image")
            raise VLMError(
                "Failed to process image",
                details={"error": str(e)}
            ) from e
    
    def describe_image(self, image: Union[str, Path, Image.Image], context: str = "") -> Dict[str, Any]:
        """
        Generate comprehensive image description.
        
        Args:
            image: Image to describe
            context: Surrounding textual context
        
        Returns:
            Dictionary with description and entity summary
        """
        prompt = get_prompt("vision_analysis", context=context)
        response = self.generate(prompt, images=[image], json_mode=True)
        
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            # Fallback if JSON parsing fails
            return {
                "image_description": response,
                "entity_summary": {
                    "name": "Image",
                    "type": "other",
                    "key_elements": [],
                    "description": response[:200]
                }
            }
    
    def analyze_table(self, table_content: str, context: str = "") -> Dict[str, Any]:
        """
        Analyze table content.
        
        Args:
            table_content: Table content as text/markdown
            context: Surrounding context
        
        Returns:
            Dictionary with analysis and entity summary
        """
        prompt = get_prompt("table_analysis", context=context, table_content=table_content)
        response = self.generate(prompt, json_mode=True)
        
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {
                "table_description": response,
                "entity_summary": {
                    "name": "Table",
                    "type": "data_table",
                    "headers": [],
                    "key_values": {},
                    "description": response[:200]
                }
            }
    
    def analyze_equation(self, equation_latex: str, context: str = "") -> Dict[str, Any]:
        """
        Analyze mathematical equation.
        
        Args:
            equation_latex: Equation in LaTeX format
            context: Surrounding context
        
        Returns:
            Dictionary with analysis and entity summary
        """
        prompt = get_prompt("equation_analysis", context=context, equation_latex=equation_latex)
        response = self.generate(prompt, json_mode=True)
        
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {
                "equation_description": response,
                "entity_summary": {
                    "name": "Equation",
                    "type": "formula",
                    "variables": {},
                    "description": response[:200]
                }
            }
    
    def extract_entities(self, text: str) -> Dict[str, Any]:
        """
        Extract entities and relations from text.
        
        Args:
            text: Text to analyze
        
        Returns:
            Dictionary with entities and relations
        """
        prompt = get_prompt("entity_extraction", text=text)
        response = self.generate(prompt, json_mode=True)
        
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {"entities": [], "relations": []}


# Example usage
if __name__ == "__main__":
    from ..config.settings import RAGConfig
    from ..utils.logger import setup_logger
    
    setup_logger(level="DEBUG")
    
    config = RAGConfig()
    vlm = VLMInterface(config.vlm)
    
    # Test image description
    # result = vlm.describe_image("test_image.jpg", context="This is a test image")
    # print(json.dumps(result, indent=2))