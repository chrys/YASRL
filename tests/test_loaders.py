import hashlib
import os
from unittest.mock import MagicMock, patch

import pytest
from llama_index.core.readers.base import BaseReader as LlamaBaseReader
from llama_index.core import SimpleDirectoryReader
from llama_index.readers.web import SimpleWebPageReader

from yasrl.exceptions import IndexingError
from yasrl.loaders import DocumentLoader
from llama_index.core import Document


@pytest.fixture
def loader():
    return DocumentLoader()


def test_detect_source_type_directory(loader, tmp_path):
    assert loader.detect_source_type(str(tmp_path)) == "directory"


def test_detect_source_type_url(loader):
    assert loader.detect_source_type("http://example.com") == "url"


def test_detect_source_type_url_list(loader):
    assert loader.detect_source_type(["http://example.com", "https://anotherexample.com"]) == "url_list"


def test_detect_source_type_unsupported(loader):
    with pytest.raises(IndexingError):
        loader.detect_source_type("unsupported_source")


def test_generate_document_id(loader):
    source = "http://example.com"
    expected_id = hashlib.sha256(source.encode()).hexdigest()
    assert loader.generate_document_id(source) == expected_id

@patch("yasrl.loaders.SimpleWebPageReader")
def test_load_documents_from_url(mock_reader, loader):
    url = "http://example.com"
    mock_doc = MagicMock()
    mock_doc.get_content.return_value = "web page content"
    mock_doc.metadata = {"extra_info": {"Source": url}}
    mock_reader.return_value.load_data.return_value = [mock_doc]

    documents = loader.load_documents(url)

    assert len(documents) == 1
    assert documents[0].text == "web page content"
    assert documents[0].id_ == loader.generate_document_id(url)


@patch("yasrl.loaders.SimpleWebPageReader")
def test_load_documents_from_url_list(mock_reader, loader):
    urls = ["http://example.com", "http://anotherexample.com"]
    mock_doc1 = MagicMock()
    mock_doc1.get_content.return_value = "web page content 1"
    mock_doc1.metadata = {"extra_info": {"Source": urls[0]}}
    mock_doc2 = MagicMock()
    mock_doc2.get_content.return_value = "web page content 2"
    mock_doc2.metadata = {"extra_info": {"Source": urls[1]}}
    mock_reader.return_value.load_data.side_effect = [[mock_doc1], [mock_doc2]]

    documents = loader.load_documents(urls)

    assert len(documents) == 2
    assert documents[0].text == "web page content 1"
    assert documents[0].id_ == loader.generate_document_id(urls[0])
    assert documents[1].text == "web page content 2"
    assert documents[1].id_ == loader.generate_document_id(urls[1])


def test_load_documents_unsupported_file_type(loader, tmp_path):
    file_path = tmp_path / "test.unsupported"
    file_path.touch()
    documents = loader.load_documents(str(file_path))
    assert len(documents) == 1
    assert documents[0].text == ""


def test_load_documents_non_existent_file(loader):
    with pytest.raises(IndexingError):
        loader.load_documents("non_existent_file.txt")