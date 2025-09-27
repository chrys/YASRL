import hashlib
import os
from pathlib import Path
from typing import List, Union
from urllib.parse import urlparse

from llama_index.core.readers.base import BaseReader as LlamaBaseReader
from llama_index.core import SimpleDirectoryReader
from llama_index.readers.web import SimpleWebPageReader

from yasrl.exceptions import IndexingError
from llama_index.core import Document


class DocumentLoader:
    def detect_source_type(self, source: Union[str, List[str]]) -> str:
        if isinstance(source, list):
            if all(self._is_url(item) for item in source):
                return "url_list"
        elif isinstance(source, str):
            if self._is_url(source):
                return "url"
            elif os.path.isfile(source):
                return "file"
            elif os.path.isdir(source):
                return "directory"
        raise IndexingError(f"Unsupported source type: {source}")

    def load_documents(self, source: Union[str, List[str]]) -> List[Document]:
        source_type = self.detect_source_type(source)
        if source_type == "url":
            assert isinstance(source, str)
            return self._load_from_url(source)
        elif source_type == "url_list":
            assert isinstance(source, list)
            return [doc for url in source for doc in self._load_from_url(url)]
        elif source_type == "directory":
            assert isinstance(source, str)
            return self._load_from_directory(source)
        elif source_type == "file":
            assert isinstance(source, str)
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
                id_=self.generate_document_id(doc.metadata.get("extra_info", {}).get("Source", url) or "unknown"),
                text=doc.get_content(),
                metadata=doc.metadata,
            )
            for doc in documents
        ]

    def _load_from_directory(self, directory_path: str) -> List[Document]:
        reader = SimpleDirectoryReader(directory_path)
        documents = reader.load_data()
        return [
            Document(
                id_=self.generate_document_id(doc.metadata.get("file_path") or "unknown"),
                text=doc.get_content(),
                metadata=doc.metadata,
            )
            for doc in documents
        ]


    def _load_from_file(self, file_path: str) -> List[Document]:
        _, extension = os.path.splitext(file_path)
        if extension not in SimpleDirectoryReader.supported_suffix_fn():
            raise IndexingError(f"Unsupported file type: {extension}")

        reader = SimpleDirectoryReader(input_files=[file_path])
        documents = reader.load_data()
        return [
            Document(
                id_=self.generate_document_id(doc.metadata.get("file_path") or "unknown"),
                text=doc.get_content(),
                metadata=doc.metadata,
            )
            for doc in documents
]