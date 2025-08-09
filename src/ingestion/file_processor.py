"""
File processing utilities for the data curation system.
"""
import os
import csv
import json
import yaml
import re
from typing import List, Dict, Any, Optional, Union, Generator, BinaryIO, TextIO
from pathlib import Path
import pandas as pd
import chardet
from loguru import logger
import magic
import hashlib
import xml.etree.ElementTree as ET
from io import StringIO, BytesIO
import zipfile
import gzip
import shutil

from ..core.config import get_settings

settings = get_settings()


class FileProcessor:
    """Processor for various file types."""
    
    def __init__(self, file_path: Union[str, Path], encoding: Optional[str] = None):
        """
        Initialize the file processor.
        
        Args:
            file_path: Path to the file
            encoding: File encoding (auto-detected if not provided)
        """
        self.file_path = Path(file_path)
        self.mime_type = self._detect_mime_type()
        self.encoding = encoding or self._detect_encoding()
        self.file_size = self.file_path.stat().st_size if self.file_path.exists() else 0
    
    def _detect_mime_type(self) -> str:
        """
        Detect the MIME type of the file.
        
        Returns:
            MIME type string
        """
        try:
            return magic.from_file(str(self.file_path), mime=True)
        except Exception as e:
            logger.error(f"Error detecting MIME type: {e}")
            # Fallback to extension-based detection
            extension = self.file_path.suffix.lower()
            mime_map = {
                '.txt': 'text/plain',
                '.csv': 'text/csv',
                '.json': 'application/json',
                '.jsonl': 'application/jsonl',
                '.xml': 'application/xml',
                '.html': 'text/html',
                '.md': 'text/markdown',
                '.yaml': 'application/yaml',
                '.yml': 'application/yaml',
                '.pdf': 'application/pdf',
                '.zip': 'application/zip',
                '.gz': 'application/gzip',
            }
            return mime_map.get(extension, 'application/octet-stream')
    
    def _detect_encoding(self) -> str:
        """
        Detect the encoding of the file.
        
        Returns:
            Encoding string
        """
        # Don't try to detect encoding for binary files
        if not self.mime_type.startswith('text/') and not self.mime_type in [
            'application/json', 'application/jsonl', 'application/xml',
            'application/yaml', 'text/markdown'
        ]:
            return 'binary'
        
        try:
            # Read a sample of the file to detect encoding
            with open(self.file_path, 'rb') as f:
                sample = f.read(min(self.file_size, 10_000))  # Read up to 10KB
            
            result = chardet.detect(sample)
            confidence = result.get('confidence', 0)
            encoding = result.get('encoding', 'utf-8')
            
            if confidence < 0.7:
                logger.warning(f"Low confidence ({confidence}) in encoding detection: {encoding}")
            
            return encoding or 'utf-8'
        except Exception as e:
            logger.error(f"Error detecting encoding: {e}")
            return 'utf-8'
    
    def read_text(self) -> str:
        """
        Read the file as text.
        
        Returns:
            File content as text
        """
        try:
            with open(self.file_path, 'r', encoding=self.encoding) as f:
                return f.read()
        except UnicodeDecodeError as e:
            logger.error(f"Unicode decode error with {self.encoding}: {e}")
            # Try with a different encoding
            with open(self.file_path, 'r', encoding='latin1') as f:
                return f.read()
    
    def read_csv(self, delimiter: str = ',') -> List[Dict[str, str]]:
        """
        Read the file as CSV.
        
        Args:
            delimiter: CSV delimiter
            
        Returns:
            List of dictionaries (rows)
        """
        try:
            with open(self.file_path, 'r', encoding=self.encoding, newline='') as f:
                reader = csv.DictReader(f, delimiter=delimiter)
                return list(reader)
        except Exception as e:
            logger.error(f"Error reading CSV: {e}")
            return []
    
    def read_json(self) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Read the file as JSON.
        
        Returns:
            Parsed JSON data
        """
        try:
            with open(self.file_path, 'r', encoding=self.encoding) as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error reading JSON: {e}")
            return {}
    
    def read_jsonl(self) -> List[Dict[str, Any]]:
        """
        Read the file as JSONL (JSON Lines).
        
        Returns:
            List of parsed JSON objects
        """
        results = []
        try:
            with open(self.file_path, 'r', encoding=self.encoding) as f:
                for line in f:
                    line = line.strip()
                    if line:  # Skip empty lines
                        try:
                            data = json.loads(line)
                            results.append(data)
                        except json.JSONDecodeError as e:
                            logger.warning(f"Error parsing JSON line: {e}")
        except Exception as e:
            logger.error(f"Error reading JSONL: {e}")
        
        return results
    
    def read_yaml(self) -> Dict[str, Any]:
        """
        Read the file as YAML.
        
        Returns:
            Parsed YAML data
        """
        try:
            with open(self.file_path, 'r', encoding=self.encoding) as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Error reading YAML: {e}")
            return {}
    
    def read_xml(self) -> Dict[str, Any]:
        """
        Read the file as XML and convert to dict.
        
        Returns:
            Dictionary representation of XML
        """
        try:
            tree = ET.parse(self.file_path)
            root = tree.getroot()
            
            def xml_to_dict(element):
                result = {}
                for child in element:
                    child_data = xml_to_dict(child)
                    if child.tag in result:
                        if isinstance(result[child.tag], list):
                            result[child.tag].append(child_data)
                        else:
                            result[child.tag] = [result[child.tag], child_data]
                    else:
                        result[child.tag] = child_data
                
                if element.text and element.text.strip():
                    if not result:
                        return element.text.strip()
                    else:
                        result['_text'] = element.text.strip()
                
                if element.attrib:
                    result['_attributes'] = element.attrib
                
                return result
            
            return {root.tag: xml_to_dict(root)}
        except Exception as e:
            logger.error(f"Error reading XML: {e}")
            return {}
    
    def read_pandas(self) -> pd.DataFrame:
        """
        Read the file as a pandas DataFrame.
        
        Returns:
            Pandas DataFrame
        """
        try:
            if self.mime_type == 'text/csv':
                return pd.read_csv(self.file_path, encoding=self.encoding)
            elif self.mime_type in ['application/json', 'application/jsonl']:
                return pd.read_json(self.file_path, encoding=self.encoding, lines=('jsonl' in self.mime_type))
            elif self.mime_type == 'application/excel':
                return pd.read_excel(self.file_path)
            else:
                logger.warning(f"Unsupported file type for pandas: {self.mime_type}")
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error reading with pandas: {e}")
            return pd.DataFrame()
    
    def extract_archive(self, extract_dir: Optional[Union[str, Path]] = None) -> List[Path]:
        """
        Extract archive contents.
        
        Args:
            extract_dir: Directory to extract to (default: temp dir)
            
        Returns:
            List of extracted file paths
        """
        if extract_dir is None:
            extract_dir = settings.DATA_DIR / "tmp" / f"extract_{os.path.basename(self.file_path)}"
        
        extract_dir = Path(extract_dir)
        os.makedirs(extract_dir, exist_ok=True)
        
        extracted_files = []
        
        try:
            if self.mime_type == 'application/zip':
                with zipfile.ZipFile(self.file_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_dir)
                    extracted_files = [extract_dir / name for name in zip_ref.namelist()]
            
            elif self.mime_type == 'application/gzip':
                # Handle single file gzip
                output_path = extract_dir / self.file_path.with_suffix('').name
                with gzip.open(self.file_path, 'rb') as f_in:
                    with open(output_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                extracted_files = [output_path]
            
            else:
                logger.warning(f"Unsupported archive type: {self.mime_type}")
        
        except Exception as e:
            logger.error(f"Error extracting archive: {e}")
        
        return extracted_files
    
    def read_content(self) -> Any:
        """
        Read the file content based on its MIME type.
        
        Returns:
            Parsed file content
        """
        if self.mime_type == 'text/plain':
            return self.read_text()
        elif self.mime_type == 'text/csv':
            return self.read_csv()
        elif self.mime_type == 'application/json':
            return self.read_json()
        elif self.mime_type == 'application/jsonl':
            return self.read_jsonl()
        elif self.mime_type in ['application/yaml', 'text/yaml']:
            return self.read_yaml()
        elif self.mime_type in ['application/xml', 'text/xml']:
            return self.read_xml()
        elif self.mime_type in ['application/zip', 'application/gzip']:
            return self.extract_archive()
        else:
            logger.warning(f"Unsupported MIME type for reading: {self.mime_type}")
            # Default to text for unknown types
            if self.encoding != 'binary':
                return self.read_text()
            return None
    
    def get_file_hash(self) -> str:
        """
        Compute the SHA-256 hash of the file.
        
        Returns:
            Hex digest of the file hash
        """
        sha256 = hashlib.sha256()
        with open(self.file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                sha256.update(chunk)
        return sha256.hexdigest()


def process_file(file_path: Union[str, Path], encoding: Optional[str] = None) -> Dict[str, Any]:
    """
    Process a file and return its content with metadata.
    
    Args:
        file_path: Path to the file
        encoding: File encoding (auto-detected if not provided)
        
    Returns:
        Dictionary with file content and metadata
    """
    processor = FileProcessor(file_path, encoding)
    content = processor.read_content()
    
    return {
        'file_path': str(file_path),
        'file_name': os.path.basename(file_path),
        'file_size': processor.file_size,
        'mime_type': processor.mime_type,
        'encoding': processor.encoding,
        'file_hash': processor.get_file_hash(),
        'content': content,
    }


def process_directory(
    dir_path: Union[str, Path],
    recursive: bool = True,
    file_patterns: Optional[List[str]] = None
) -> Generator[Dict[str, Any], None, None]:
    """
    Process all files in a directory.
    
    Args:
        dir_path: Directory path
        recursive: Whether to process subdirectories
        file_patterns: List of file patterns to include (e.g., ["*.txt", "*.csv"])
        
    Yields:
        File content and metadata for each file
    """
    dir_path = Path(dir_path)
    
    if not dir_path.is_dir():
        logger.error(f"Not a directory: {dir_path}")
        return
    
    # Compile regex patterns
    if file_patterns:
        patterns = [re.compile(pattern.replace("*", ".*")) for pattern in file_patterns]
    else:
        patterns = None
    
    # Walk directory
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            file_path = Path(root) / file
            
            # Check if file matches patterns
            if patterns:
                if not any(pattern.match(file) for pattern in patterns):
                    continue
            
            try:
                result = process_file(file_path)
                yield result
            except Exception as e:
                logger.error(f"Error processing file {file_path}: {e}")
        
        if not recursive:
            break  # Don't process subdirectories
