import pytest

from ancora.chunking import chunk_document, chunk_documents
from ancora.retrieval import BM25Index, Retriever
from ancora.types import Document

LONG = " ".join(f"Esta e a frase numero {i} do documento de teste." for i in range(1, 21))


class TestChunking:
    def test_offsets_map_back_to_source(self):
        doc = Document(id="d", text=LONG)
        for chunk in chunk_document(doc, target_chars=150):
            assert doc.text[chunk.start : chunk.end].strip() == chunk.text

    def test_respects_target_size_roughly(self):
        chunks = chunk_document(Document(id="d", text=LONG), target_chars=150)
        assert len(chunks) > 1
        assert all(len(c.text) < 400 for c in chunks)

    def test_never_splits_mid_sentence(self):
        chunks = chunk_document(Document(id="d", text=LONG), target_chars=100)
        for chunk in chunks:
            assert chunk.text.rstrip().endswith(".")

    def test_overlap_repeats_tail_sentence(self):
        with_overlap = chunk_document(Document(id="d", text=LONG), target_chars=150, overlap_sentences=1)
        without = chunk_document(Document(id="d", text=LONG), target_chars=150, overlap_sentences=0)
        assert len(with_overlap) >= len(without)

    def test_short_document_is_one_chunk(self):
        chunks = chunk_document(Document(id="d", text="Uma frase curta."))
        assert len(chunks) == 1

    def test_empty_document(self):
        assert chunk_document(Document(id="d", text="")) == []
        assert chunk_document(Document(id="d", text="   ")) == []

    def test_metadata_propagates(self):
        doc = Document(id="d", text=LONG, metadata={"source": "manual.md"})
        assert all(c.metadata["source"] == "manual.md" for c in chunk_document(doc))

    def test_ids_unique(self):
        chunks = chunk_documents(
            [Document(id="a", text=LONG), Document(id="b", text=LONG)], target_chars=150
        )
        assert len({c.id for c in chunks}) == len(chunks)

    @pytest.mark.parametrize("bad", [0, -1])
    def test_invalid_target(self, bad):
        with pytest.raises(ValueError):
            chunk_document(Document(id="d", text=LONG), target_chars=bad)

    def test_document_requires_id(self):
        with pytest.raises(ValueError):
            Document(id="", text="x")


@pytest.fixture
def chunks():
    return chunk_documents(
        [
            Document(id="frete", text="O frete custa R$ 24,90 para todo o Brasil.", metadata={"source": "frete"}),
            Document(id="prazo", text="O prazo de entrega e de 5 dias uteis.", metadata={"source": "prazo"}),
            Document(id="troca", text="Trocas em ate 7 dias corridos apos o recebimento.", metadata={"source": "troca"}),
        ]
    )


class TestBM25:
    def test_ranks_relevant_chunk_first(self, chunks):
        idx = BM25Index(chunks)
        scores = idx.score("quanto custa o frete")
        assert scores.index(max(scores)) == 0

    def test_unknown_terms_score_zero(self, chunks):
        assert all(s == 0.0 for s in BM25Index(chunks).score("zzzz qqqq"))

    def test_empty_query(self, chunks):
        assert all(s == 0.0 for s in BM25Index(chunks).score(""))

    def test_empty_index(self):
        assert BM25Index([]).score("frete") == []


class TestRetriever:
    def test_returns_top_k(self, chunks):
        assert len(Retriever(chunks).search("frete prazo troca", top_k=2)) <= 2

    def test_scores_normalised(self, chunks):
        hits = Retriever(chunks).search("frete", top_k=3)
        assert hits and hits[0].score == pytest.approx(1.0)
        assert all(0.0 < h.score <= 1.0 for h in hits)

    def test_ordered_descending(self, chunks):
        hits = Retriever(chunks).search("prazo entrega dias", top_k=3)
        assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)

    def test_no_match_returns_empty(self, chunks):
        assert Retriever(chunks).search("xyzzy plugh") == []

    def test_deterministic(self, chunks):
        r = Retriever(chunks)
        a = [h.chunk.id for h in r.search("frete", top_k=3)]
        b = [h.chunk.id for h in r.search("frete", top_k=3)]
        assert a == b

    def test_dense_blend_runs(self, chunks):
        class FakeEmbedder:
            def embed(self, texts):
                return [[float(len(t)), 1.0, 0.5] for t in texts]

        hits = Retriever(chunks, embedder=FakeEmbedder(), dense_weight=0.5).search("frete", top_k=2)
        assert hits

    @pytest.mark.parametrize("bad", [-0.1, 1.1])
    def test_invalid_dense_weight(self, chunks, bad):
        class E:
            def embed(self, texts):
                return [[1.0] for _ in texts]

        with pytest.raises(ValueError):
            Retriever(chunks, embedder=E(), dense_weight=bad)
