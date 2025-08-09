"""
AI-powered data enhancement and transformation for LLM training data.
"""
import os
import json
from typing import List, Dict, Any, Optional, Union, Generator, Tuple
import numpy as np
from transformers import pipeline
from openai import OpenAI
import backoff
import time
from loguru import logger

from ..core.config import get_settings

settings = get_settings()


class AIDataEnhancer:
    """AI-based data enhancement for LLM training."""
    
    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        openai_model: str = "gpt-3.5-turbo",
        hf_model: Optional[str] = None,
        use_openai: bool = True,
    ):
        """
        Initialize the AI enhancer.
        
        Args:
            openai_api_key: OpenAI API key
            openai_model: OpenAI model to use
            hf_model: Hugging Face model to use
            use_openai: Whether to use OpenAI (vs HF)
        """
        self.openai_api_key = openai_api_key or settings.OPENAI_API_KEY
        self.openai_model = openai_model
        self.hf_model = hf_model
        self.use_openai = use_openai
        
        # Initialize clients
        self.openai_client = None
        self.hf_pipeline = None
        
        if self.use_openai and self.openai_api_key:
            self._init_openai()
        elif self.hf_model:
            self._init_hf_pipeline()
    
    def _init_openai(self):
        """Initialize OpenAI client."""
        try:
            self.openai_client = OpenAI(api_key=self.openai_api_key)
            logger.info(f"OpenAI client initialized with model {self.openai_model}")
        except Exception as e:
            logger.error(f"Error initializing OpenAI client: {e}")
            self.openai_client = None
    
    def _init_hf_pipeline(self):
        """Initialize Hugging Face pipeline."""
        try:
            self.hf_pipeline = pipeline(
                "text-generation",
                model=self.hf_model,
                device_map="auto",
            )
            logger.info(f"Hugging Face pipeline initialized with model {self.hf_model}")
        except Exception as e:
            logger.error(f"Error initializing Hugging Face pipeline: {e}")
            self.hf_pipeline = None
    
    @backoff.on_exception(backoff.expo, Exception, max_tries=3)
    def _generate_with_openai(self, prompt: str) -> str:
        """
        Generate text with OpenAI.
        
        Args:
            prompt: Input prompt
            
        Returns:
            Generated text
        """
        if not self.openai_client:
            logger.error("OpenAI client not initialized")
            return ""
        
        try:
            response = self.openai_client.chat.completions.create(
                model=self.openai_model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that enhances and cleans text data for LLM training."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1000,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Error generating with OpenAI: {e}")
            raise
    
    def _generate_with_hf(self, prompt: str) -> str:
        """
        Generate text with Hugging Face.
        
        Args:
            prompt: Input prompt
            
        Returns:
            Generated text
        """
        if not self.hf_pipeline:
            logger.error("Hugging Face pipeline not initialized")
            return ""
        
        try:
            outputs = self.hf_pipeline(
                prompt,
                max_new_tokens=1000,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )
            return outputs[0]["generated_text"].strip()
        except Exception as e:
            logger.error(f"Error generating with Hugging Face: {e}")
            return ""
    
    def generate(self, prompt: str) -> str:
        """
        Generate text with AI.
        
        Args:
            prompt: Input prompt
            
        Returns:
            Generated text
        """
        if self.use_openai and self.openai_client:
            return self._generate_with_openai(prompt)
        elif self.hf_pipeline:
            return self._generate_with_hf(prompt)
        else:
            logger.error("No AI generator available")
            return ""
    
    def enhance_text(self, text: str, task: str = "clean") -> str:
        """
        Enhance text with AI.
        
        Args:
            text: Input text
            task: Enhancement task (clean, rewrite, summarize, etc.)
            
        Returns:
            Enhanced text
        """
        prompts = {
            "clean": f"Clean the following text for language model training. Fix grammar, spelling, and formatting issues without changing the meaning:\n\n{text}",
            "rewrite": f"Rewrite the following text to make it more clear and concise, while preserving the key information:\n\n{text}",
            "summarize": f"Provide a concise summary of the following text:\n\n{text}",
            "expand": f"Expand on the following text with additional relevant details and explanations:\n\n{text}",
            "format": f"Format the following text with proper paragraphs, bullet points where appropriate, and clean spacing:\n\n{text}",
        }
        
        prompt = prompts.get(task, prompts["clean"])
        return self.generate(prompt)
    
    def classify_text(self, text: str, categories: List[str]) -> Dict[str, float]:
        """
        Classify text into categories.
        
        Args:
            text: Input text
            categories: List of categories
            
        Returns:
            Dictionary mapping categories to confidence scores
        """
        categories_str = ", ".join(categories)
        prompt = f"Classify the following text into one or more of these categories: {categories_str}. Return ONLY a JSON object mapping each category to a confidence score between 0 and 1:\n\n{text}"
        
        response = self.generate(prompt)
        
        # Try to parse JSON from response
        try:
            # Extract JSON part from response if needed
            json_start = response.find("{")
            json_end = response.rfind("}")
            if json_start >= 0 and json_end >= 0:
                json_str = response[json_start:json_end+1]
                result = json.loads(json_str)
                return result
        except Exception as e:
            logger.error(f"Error parsing classification result: {e}")
        
        # Fallback: return empty scores
        return {category: 0.0 for category in categories}
    
    def extract_metadata(self, text: str, fields: List[str]) -> Dict[str, Any]:
        """
        Extract metadata from text.
        
        Args:
            text: Input text
            fields: List of fields to extract
            
        Returns:
            Dictionary mapping fields to extracted values
        """
        fields_str = ", ".join(fields)
        prompt = f"Extract the following information from the text as a JSON object: {fields_str}. Return ONLY a valid JSON object with these fields:\n\n{text}"
        
        response = self.generate(prompt)
        
        # Try to parse JSON from response
        try:
            # Extract JSON part from response if needed
            json_start = response.find("{")
            json_end = response.rfind("}")
            if json_start >= 0 and json_end >= 0:
                json_str = response[json_start:json_end+1]
                result = json.loads(json_str)
                return result
        except Exception as e:
            logger.error(f"Error parsing metadata extraction result: {e}")
        
        # Fallback: return empty metadata
        return {field: None for field in fields}
    
    def filter_quality(self, text: str) -> Tuple[bool, float, str]:
        """
        Assess text quality for LLM training.
        
        Args:
            text: Input text
            
        Returns:
            Tuple of (is_high_quality, quality_score, reason)
        """
        prompt = """
        Evaluate the quality of the following text for language model training. Consider:
        1. Coherence and readability
        2. Grammar and spelling
        3. Factual accuracy (if applicable)
        4. Information density
        5. Potential for harmful, biased, or inappropriate content
        
        Return ONLY a JSON object with three fields:
        - is_high_quality: boolean
        - quality_score: float between 0 and 1
        - reason: brief explanation
        
        Text to evaluate:
        
        """ + text
        
        response = self.generate(prompt)
        
        # Try to parse JSON from response
        try:
            # Extract JSON part from response if needed
            json_start = response.find("{")
            json_end = response.rfind("}")
            if json_start >= 0 and json_end >= 0:
                json_str = response[json_start:json_end+1]
                result = json.loads(json_str)
                return (
                    result.get("is_high_quality", False),
                    result.get("quality_score", 0.0),
                    result.get("reason", "No reason provided")
                )
        except Exception as e:
            logger.error(f"Error parsing quality assessment result: {e}")
        
        # Fallback
        return False, 0.0, "Error processing quality assessment"
    
    def generate_variants(self, text: str, num_variants: int = 3) -> List[str]:
        """
        Generate variants of text for data augmentation.
        
        Args:
            text: Input text
            num_variants: Number of variants to generate
            
        Returns:
            List of text variants
        """
        prompt = f"""
        Generate {num_variants} different versions of the following text. Each version should preserve the core meaning but use different wording, structure, or examples. Return ONLY a JSON array containing the variants.
        
        Original text:
        {text}
        """
        
        response = self.generate(prompt)
        
        # Try to parse JSON from response
        try:
            # Extract JSON part from response if needed
            json_start = response.find("[")
            json_end = response.rfind("]")
            if json_start >= 0 and json_end >= 0:
                json_str = response[json_start:json_end+1]
                result = json.loads(json_str)
                return result
        except Exception as e:
            logger.error(f"Error parsing variants result: {e}")
        
        # Fallback: split by newlines and hope for the best
        variants = [line.strip() for line in response.split("\n") if line.strip()]
        return variants[:num_variants] if variants else []


def enhance_dataset(
    data: List[Dict[str, Any]],
    content_field: str = 'content',
    output_field: str = 'enhanced_content',
    task: str = "clean",
    openai_api_key: Optional[str] = None,
    openai_model: str = "gpt-3.5-turbo",
    max_items: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Enhance a dataset with AI.
    
    Args:
        data: Input dataset
        content_field: Field containing text content
        output_field: Field to store enhanced content
        task: Enhancement task
        openai_api_key: OpenAI API key
        openai_model: OpenAI model to use
        max_items: Maximum number of items to process
        
    Returns:
        Enhanced dataset
    """
    enhancer = AIDataEnhancer(
        openai_api_key=openai_api_key,
        openai_model=openai_model,
        use_openai=True,
    )
    
    # Process only a subset if specified
    if max_items is not None:
        process_data = data[:max_items]
    else:
        process_data = data
    
    for i, item in enumerate(process_data):
        if content_field in item and item[content_field]:
            text = item[content_field]
            
            # Skip very short or long texts
            if len(text) < 20 or len(text) > 10000:
                item[output_field] = text
                continue
            
            # Apply enhancement
            try:
                enhanced_text = enhancer.enhance_text(text, task=task)
                item[output_field] = enhanced_text
                logger.info(f"Enhanced item {i+1}/{len(process_data)}")
            except Exception as e:
                logger.error(f"Error enhancing item {i+1}: {e}")
                item[output_field] = text  # Use original text as fallback
            
            # Add a small delay to avoid rate limits
            time.sleep(0.5)
    
    return data
