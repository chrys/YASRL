import unittest
from yasrl.models import SourceChunk, QueryResult

class TestSourceChunk(unittest.TestCase):
    def test_creation_and_access(self):
        chunk = SourceChunk(text="Hello", metadata={"source": "doc.txt"}, score=0.5)
        self.assertEqual(chunk.text, "Hello")
        self.assertEqual(chunk.metadata["source"], "doc.txt")
        self.assertEqual(chunk.score, 0.5)

    def test_validation_success(self):
        chunk = SourceChunk(text="Valid text", score=0.8)
        chunk.validate()  # Should not raise

    def test_validation_empty_text(self):
        chunk = SourceChunk(text=" ")
        with self.assertRaises(ValueError):
            chunk.validate()

    def test_validation_score_out_of_range(self):
        chunk = SourceChunk(text="Text", score=1.5)
        with self.assertRaises(ValueError):
            chunk.validate()

    def test_from_dict(self):
        data = {"text": "Dict text", "metadata": {"source": "doc"}, "score": 0.7}
        chunk = SourceChunk.from_dict(data)
        self.assertEqual(chunk.text, "Dict text")
        self.assertEqual(chunk.metadata["source"], "doc")
        self.assertEqual(chunk.score, 0.7)

class TestQueryResult(unittest.TestCase):
    def test_creation_and_access(self):
        chunk = SourceChunk(text="Chunk", metadata={"source": "doc"}, score=0.9)
        result = QueryResult(answer="Answer", source_chunks=[chunk])
        self.assertEqual(result.answer, "Answer")
        self.assertEqual(result.source_chunks[0].text, "Chunk")

    def test_validation_success(self):
        chunk = SourceChunk(text="Chunk")
        result = QueryResult(answer="Valid", source_chunks=[chunk])
        result.validate()  # Should not raise

    def test_validation_empty_answer(self):
        result = QueryResult(answer=" ", source_chunks=[])
        with self.assertRaises(ValueError):
            result.validate()

    def test_validation_invalid_source_chunks(self):
        result = QueryResult(answer="Valid", source_chunks=["not_a_chunk"])
        with self.assertRaises(ValueError):
            result.validate()

    def test_to_dict(self):
        chunk = SourceChunk(text="Chunk", metadata={"source": "doc"}, score=0.9)
        result = QueryResult(answer="Answer", source_chunks=[chunk])
        d = result.to_dict()
        self.assertEqual(d["answer"], "Answer")
        self.assertEqual(d["source_chunks"][0]["text"], "Chunk")

    def test_get_sources(self):
        chunk1 = SourceChunk(text="A", metadata={"source": "doc1"})
        chunk2 = SourceChunk(text="B", metadata={"source": "doc2"})
        chunk3 = SourceChunk(text="C", metadata={"source": "doc1"})
        result = QueryResult(answer="Ans", source_chunks=[chunk1, chunk2, chunk3])
        sources = result.get_sources()
        self.assertCountEqual(sources, ["doc1", "doc2"])

if __name__ == "__main__":
    unittest.main()
