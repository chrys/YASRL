import hashlib
import os
from pathlib import Path
from typing import List, Union
from urllib.parse import urlparse

from llama_index.core.readers.base import BaseReader as LlamaBaseReader
from llama_index.readers.file.base import SimpleDirectoryReader
from llama_index.readers.web.base import SimpleWebPageReader

from yasrl.exceptions import IndexingError
from yasrl.models import Document


class DocumentLoader:
    def detect_source_type(self, source: Union[str, List[str]]) -> str:
        if isinstance(source, list):
            if all(self._is_url(item) for item in source):
                return "url_list"
        elif isinstance(source, str):
            if self._is_url(source):
                return "url"
            elif os.path.isdir(source):
                return "directory"
            elif os.path.isfile(source):
                return "file"
        raise IndexingError(f"Unsupported source type: {source}")

    def load_documents(self, source: Union[str, List[str]]) -> List[Document]:
        source_type = self.detect_source_type(source)
        if source_type == "url":
            return self._load_from_url(source)
        elif source_type == "url_list":
            return self._load_from_url_list(source)
        elif source_type == "directory":
            return self._load_from_directory(source)
        elif source_type == "file":
            return self._load_from_file(source)
        else:
            raise IndexingError(f"Unsupported source type: {source_type}")

    def generate_document_id(self, source: str) -> str:
        return hashlib.sha256(source.encode()).hexdigest()

    def _is_url(self, source: str) -> bool:
        try:
            result = urlparse(source)
            return all([result.scheme, result.netloc])
        except ValueError:
            return False

    def _load_from_url(self, url: str) -> List[Document]:
        documents = SimpleWebPageReader(html_to_text=True).load_data([url])
        return [
            Document(
                id=self.generate_document_id(doc.metadata.get("extra_info", {}).get("Source", url)),
                text=doc.get_content(),
                metadata=doc.metadata,
            )
            for doc in documents
        ]

    def _load_from_url_list(self, urls: List[str]) -> List[Document]:
        documents = []
        for url in urls:
            documents.extend(self._load_from_url(url))
        return documents

    def _load_from_directory(self, directory_path: str) -> List[Document]:
        reader = SimpleDirectoryReader(directory_path)
        documents = reader.load_data()
        return [
            Document(
                id=self.generate_document_id(doc.metadata.get("file_path")),
                text=doc.get_content(),
                metadata=doc.metadata,
            )
            for doc in documents
        ]

    def _load_from_file(self, file_path: str) -> List[Document]:
        _, extension = os.path.splitext(file_path)
        if extension not in SimpleDirectoryReader.supported_suffix():
            raise IndexingError(f"Unsupported file type: {extension}")

        reader = SimpleDirectoryReader(input_files=[file_path])
        documents = reader.load_data()
        return [
            Document(
                id=self.generate_document_id(doc.metadata.get("file_path")),
                text=doc.get_content(),
                metadata=doc.metadata,
            )
            for doc in documents
        ]
